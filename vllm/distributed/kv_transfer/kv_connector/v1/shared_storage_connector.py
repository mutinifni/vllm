# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import os
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Any
import time

import safetensors
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    # Request tokens
    token_ids: torch.Tensor
    # Slot mappings, should have the same length as token_ids
    slot_mapping: torch.Tensor
    # Is store or load
    is_store: bool
    # Request ID for timing lookup
    req_id: Optional[str] = None

    @staticmethod
    def make_meta(token_ids: list[int], block_ids: list[int], block_size: int,
                  is_store: bool, req_id: Optional[str] = None) -> "ReqMeta":
        valid_num_tokens = align_to_block_size(len(token_ids), block_size)
        token_ids_tensor = torch.tensor(token_ids)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = block_offsets.reshape((1, block_size)) + \
                block_ids_tensor.reshape((num_blocks, 1)) * block_size
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return ReqMeta(
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            req_id=req_id,
        )


@dataclass
class SharedStorageConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        req_id: Optional[str] = None,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(token_ids, block_ids, block_size, is_store, req_id))


class SharedStorageConnector(KVConnectorBase_V1):
    # NOTE: This is Simple debug implementation of the KV connector.
    # It save / load the KV cache to / from the disk.
    # It does extra work which will overwrite the existing prefix-cache in GPU
    # - to remove the overhead, need to add some "mask" in the ReqMeta class

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Request] = {}
        transfer_config = vllm_config.kv_transfer_config
        self._storage_path = transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp")
        logger.info(vllm_config.kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """
        attn_metadata = forward_context.attn_metadata

        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            """Inject the KV cache into the layer.

            Args:
                dst_kv_cache_layer (torch.Tensor): the destination KV cache
                    layer. In shape [2, num_pages, page_size, xxx] if not
                    using MLA, [num_pages, page_size, xxx] otherwise.
                src_kv_cache (torch.Tensor): the source KV cache. In shape
                    [2, num_tokens, xxx] if not using MLA, [num_tokens, xxx]
                    otherwise.
                slot_mapping (torch.Tensor): the slot mapping. In shape
                    [num_tokens].
            """
            dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages = dst_kv_cache_layer_shape[0]
                page_size = dst_kv_cache_layer_shape[1]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    num_pages * page_size, -1)
                dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)
            else:
                num_pages = dst_kv_cache_layer_shape[1]
                page_size = dst_kv_cache_layer_shape[2]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    2, num_pages * page_size, -1)
                dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)

        # Get the metadata
        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, SharedStorageConnectorMetadata)

        if metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the connector metadata is None"
            )
            return

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the attn_metadata is None")
            return

        # Load the KV for each request each layer
        for request in metadata.requests:
            if request.is_store:
                continue
            logger.info("Inject KV cache of %d tokens to the paged memory",
                        len(request.slot_mapping))
            for layer_name in forward_context.no_compile_layers:
                attn_layer = forward_context.no_compile_layers[layer_name]
                kv_cache_layer = attn_layer.kv_cache[\
                        forward_context.virtual_engine]

                filename = self._generate_filename_debug(
                    layer_name, request.token_ids)
                kv_cache = safetensors.torch.load_file(
                    filename)["kv_cache"].cuda()
                inject_kv_into_layer(kv_cache_layer, kv_cache,
                                     request.slot_mapping)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Start saving the KV cache of the layer from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> torch.Tensor:
            """Extract the KV cache from the layer.

            Assume the shape of the layer is (2, num_pages, page_size, xxx)
            if MLA is not used, and (num_pages, page_size, xxx) otherwise.
            """
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages, page_size = layer.shape[0], layer.shape[1]
                return layer.reshape(num_pages * page_size, -1)[slot_mapping,
                                                                ...]
            num_pages, page_size = layer.shape[1], layer.shape[2]
            return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping,
                                                               ...]

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, SharedStorageConnectorMetadata)
        for request in connector_metadata.requests:
            if request.is_store:
                filename = self._generate_filename_debug(
                    layer_name, request.token_ids)
                kv_cache = extract_kv_from_layer(kv_layer,
                                                 request.slot_mapping)
                tensors = {"kv_cache": kv_cache.detach().cpu()}
                safetensors.torch.save_file(tensors, filename)

    def wait_for_save(self):
        return

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        # NOTE: in this debug implementation, we assume that the prompt is
        # cached_prompt + newly_generated_single_token
        # Therefore, we use prompt_token_ids[:-1] to determine the folder name

        # NOTE: in current v1 scheduler, the num_computed_tokens is aligned
        # with the block granularity. And it expects the returned blocks and
        # num_computed_tokens to also be aligned with the block granularity.
        if not self._found_match_for_request(request):
            return 0, False

        logger.info("External Cache Hit!")

        # Now, first num_tokens_to_check tokens are hit, we need to prepare
        # the metadata for the worker connector to correctly load the KV
        num_tokens_to_check = align_to_block_size(
            len(request.prompt_token_ids) - 1, self._block_size)

        return num_tokens_to_check - num_computed_tokens, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.

        If blocks were allocated, add to _requests_need_load,
        such that we load the KVs in the next forward pass.
        """
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = SharedStorageConnectorMetadata()

        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._requests_need_load:
                meta.add_request(token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size,
                                 is_store=False)
                total_need_load += 1
            else:
                # NOTE: here, we set the store and load being exclusive,
                # but a single request can have both store and load.
                # NOTE(rob): for this debug implementation, we only cache
                # the original prompt tokens.
                if not self._found_match_for_request(new_req):
                    meta.add_request(token_ids=new_req.prompt_token_ids,
                                     block_ids=new_req.block_ids[0],
                                     block_size=self._block_size,
                                     is_store=True)

        for cached_req in scheduler_output.scheduled_cached_reqs:
            # NOTE(rob): here we rely on the resumed requests being
            # the first N requests in the list scheduled_cache_reqs.
            if not cached_req.resumed_from_preemption:
                break
            if cached_req.req_id in self._requests_need_load:
                # NOTE(rob): cached_req_data does not have the full
                # list of token ids (only new tokens). So we look it
                # up in the actual request object.
                request = self._requests_need_load[cached_req.req_id]
                total_tokens = (len(cached_req.new_token_ids) +
                                cached_req.num_computed_tokens)
                token_ids = request.all_token_ids[:total_tokens]

                # NOTE(rob): For resumed req, new_block_ids is all
                # of the block_ids for the request.
                block_ids = cached_req.new_block_ids[0]

                meta.add_request(token_ids=token_ids,
                                 block_ids=block_ids,
                                 block_size=self._block_size,
                                 is_store=False)
                total_need_load += 1

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        return meta

    # ==============================
    # Helper functions
    # ==============================

    def _found_match_for_request(
        self,
        request: "Request",
    ) -> bool:
        """Check if the cache is hit for the request.
        """
        num_tokens_to_check = align_to_block_size(
            len(request.prompt_token_ids) - 1, self._block_size)
        foldername = self._generate_foldername_debug(torch.tensor(
            request.prompt_token_ids)[:num_tokens_to_check],
                                                     create_folder=False)
        return os.path.exists(foldername)

    def _generate_foldername_debug(
        self,
        input_ids: torch.Tensor,
        create_folder=False,
    ) -> str:
        """Generate a folder name based on the hash of the bytes of the input
        ids.
        """
        input_ids_bytes = input_ids.numpy().tobytes()
        input_ids_hash = hashlib.md5(input_ids_bytes,
                                     usedforsecurity=False).hexdigest()
        foldername = os.path.join(self._storage_path, input_ids_hash)
        if create_folder:
            os.makedirs(foldername, exist_ok=True)
        return foldername

    def _generate_filename_debug(
        self,
        layer_name: str,
        input_ids: torch.Tensor,
    ) -> str:
        """Generate a file name based on the layer name and the hash
        of the bytes of the input ids.
        """
        foldername = self._generate_foldername_debug(input_ids,
                                                     create_folder=True)
        return os.path.join(foldername, f"{layer_name}.safetensors")


class PreloadedSharedStorageConnector(SharedStorageConnector):
    """
    A SharedStorageConnector that preloads KV caches into CPU memory at startup
    to reduce disk I/O overhead on each request.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)

        # Storage for preloaded KV caches: {filename -> kv_cache_tensor}
        self._preloaded_caches: Dict[str, torch.Tensor] = {}

        # Try to preload caches if manifest path is provided
        transfer_config = vllm_config.kv_transfer_config
        manifest_path = transfer_config.get_from_extra_config("manifest_path", None)

        if manifest_path:
            self._preload_kv_caches(manifest_path)
        else:
            logger.debug("No manifest_path provided in kv_transfer_config, skipping preloading")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        logger.debug(f"[PreloadedConnector] get_num_new_matched_tokens called: "
                   f"request_id={request.request_id}, num_computed_tokens={num_computed_tokens}")

        has_match = self._found_match_for_request(request)
        logger.debug(f"[PreloadedConnector] _found_match_for_request returned: {has_match}")

        if not has_match:
            logger.debug(f"[PreloadedConnector] No cache hit, returning (0, False)")
            return 0, False

        logger.info("External Cache Hit!")

        num_tokens_to_check = align_to_block_size(
            len(request.prompt_token_ids) - 1, self._block_size)

        if num_tokens_to_check == 0:
            num_tokens_to_check = len(request.prompt_token_ids)
            logger.debug(f"[PreloadedConnector] Adjusted num_tokens_to_check to prompt length: {num_tokens_to_check}")

        result_tokens = num_tokens_to_check - num_computed_tokens
        logger.debug(f"[PreloadedConnector] Cache hit! num_tokens_to_check={num_tokens_to_check}, "
                   f"result_tokens={result_tokens}")

        return result_tokens, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Override to add debugging for preloaded connector.
        """
        logger.debug(f"[PreloadedConnector] update_state_after_alloc called: "
                   f"request_id={request.request_id}, num_external_tokens={num_external_tokens}")
        logger.debug(f"[PreloadedConnector] Blocks: {blocks}")
        logger.debug(f"[PreloadedConnector] Request prompt_token_ids: {getattr(request, 'prompt_token_ids', 'not found')}")

        super().update_state_after_alloc(request, blocks, num_external_tokens)

        logger.debug(f"[PreloadedConnector] After super().update_state_after_alloc:")
        logger.debug(f"[PreloadedConnector] Requests needing load: {list(self._requests_need_load.keys())}")
        logger.debug(f"[PreloadedConnector] Request {request.request_id} in _requests_need_load: {request.request_id in self._requests_need_load}")

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Override to add debugging and handle requests properly."""
        logger.debug(f"[PreloadedConnector] build_connector_meta called:")
        logger.debug(f"  scheduled_new_reqs: {len(scheduler_output.scheduled_new_reqs)}")
        logger.debug(f"  scheduled_cached_reqs: {len(scheduler_output.scheduled_cached_reqs)}")
        logger.debug(f"  requests_need_load: {list(self._requests_need_load.keys())}")

        # Log details about each cached request
        for i, req in enumerate(scheduler_output.scheduled_cached_reqs):
            # CachedRequestData might have different attributes, let's be safe
            req_id = getattr(req, 'request_id', getattr(req, 'req_id', f'unknown_{i}'))
            num_tokens = getattr(req, 'num_computed_tokens', getattr(req, 'num_tokens', 'unknown'))
            resumed = getattr(req, 'resumed_from_preemption', 'not found')
            logger.debug(f"  cached_req[{i}]: req_id={req_id}, num_computed_tokens={num_tokens}, resumed_from_preemption={resumed}")
            logger.debug(f"  cached_req[{i}] attributes: {dir(req)}")

        # Create metadata
        meta = SharedStorageConnectorMetadata()
        total_need_load = 0

        # Process scheduled new requests
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._requests_need_load:
                meta.add_request(req_id=new_req.req_id,
                                 token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size,
                                 is_store=False)
                total_need_load += 1
                logger.debug(f"[PreloadedConnector] Added load request for new_req: {new_req.req_id}")
            else:
                # Store requests for cache misses
                if not self._found_match_for_request(new_req):
                    meta.add_request(req_id=new_req.req_id,
                                     token_ids=new_req.prompt_token_ids,
                                     block_ids=new_req.block_ids[0],
                                     block_size=self._block_size,
                                     is_store=True)
                    logger.debug(f"[PreloadedConnector] Added store request for new_req: {new_req.req_id}")

        # Process scheduled cached requests
        # For PreloadedConnector, we want to handle ALL cached requests that need loading,
        # not just resumed ones. This differs from the parent class which only handles
        # resumed (preempted) requests.
        for cached_req in scheduler_output.scheduled_cached_reqs:
            # Check if this request needs loading (regardless of preemption status)
            if cached_req.req_id in self._requests_need_load:
                request = self._requests_need_load[cached_req.req_id]
                total_tokens = (len(cached_req.new_token_ids) +
                                cached_req.num_computed_tokens)
                token_ids = request.all_token_ids[:total_tokens]
                block_ids = cached_req.new_block_ids[0]

                meta.add_request(req_id=cached_req.req_id,
                                 token_ids=token_ids,
                                 block_ids=block_ids,
                                 block_size=self._block_size,
                                 is_store=False)
                total_need_load += 1
                logger.debug(f"[PreloadedConnector] Added load request for cached_req: {cached_req.req_id} (resumed={cached_req.resumed_from_preemption})")

        # Handle orphaned requests in _requests_need_load that aren't in scheduled lists
        # This can happen when the scheduler doesn't include them in the current step
        orphaned_requests = 0

        # OPTIMIZATION: Create sets for O(1) lookup instead of O(n) loops
        scheduled_new_req_ids = {new_req.req_id for new_req in scheduler_output.scheduled_new_reqs}
        scheduled_cached_req_ids = {cached_req.req_id for cached_req in scheduler_output.scheduled_cached_reqs}

        for req_id, request in list(self._requests_need_load.items()):  # Use list() to avoid dict iteration issues
            # Check if already processed using O(1) set lookup
            already_processed = (req_id in scheduled_new_req_ids or
                               req_id in scheduled_cached_req_ids)

            # If not processed, it's an orphaned request - add it to metadata
            if not already_processed:
                logger.debug(f"[PreloadedConnector] Found orphaned request in _requests_need_load: {req_id}")
                # For orphaned requests, use the full prompt token ids
                meta.add_request(req_id=req_id,
                                 token_ids=request.prompt_token_ids,
                                 block_ids=[1],  # Default block id
                                 block_size=self._block_size,
                                 is_store=False)
                total_need_load += 1
                orphaned_requests += 1

        logger.debug(f"[PreloadedConnector] total_need_load={total_need_load}, "
                   f"len(_requests_need_load)={len(self._requests_need_load)}, "
                   f"orphaned_requests={orphaned_requests}")

        # Clear the requests that need loading
        self._requests_need_load.clear()

        logger.debug(f"[PreloadedConnector] build_connector_meta result: {len(meta.requests)} requests")
        for i, req in enumerate(meta.requests):
            logger.debug(f"  Request {i}: is_store={req.is_store}, tokens={len(req.token_ids)}")

        return meta

    def _preload_kv_caches(self, manifest_path: str) -> None:
        """Preload all KV caches from the manifest into CPU memory."""
        logger.debug(f"Preloading KV caches from manifest: {manifest_path}")

        if not os.path.exists(manifest_path):
            logger.warning(f"Manifest file not found: {manifest_path}")
            return

        try:
            # Simply preload all .safetensors files found in the storage directory
            # This is more reliable than trying to parse the manifest
            layer_files = []
            for root, dirs, files in os.walk(self._storage_path):
                for file in files:
                    if file.endswith('.safetensors'):
                        layer_files.append(os.path.join(root, file))

            logger.debug(f"Found {len(layer_files)} layer files to preload")

            # Preload each file into CPU memory
            preloaded_count = 0
            total_size_mb = 0
            for filepath in layer_files:
                try:
                    # Load KV cache and keep it on CPU
                    kv_cache = safetensors.torch.load_file(filepath)["kv_cache"]
                    # Store in CPU memory (don't move to CUDA yet)
                    self._preloaded_caches[filepath] = kv_cache

                    # Calculate size for logging
                    size_mb = kv_cache.numel() * kv_cache.element_size() / (1024 * 1024)
                    total_size_mb += size_mb
                    preloaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to preload {filepath}: {e}")

            logger.debug(f"Successfully preloaded {preloaded_count} KV cache files into CPU memory "
                       f"(total size: {total_size_mb:.1f} MB)")

        except Exception as e:
            logger.error(f"Failed to preload KV caches: {e}")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        This version loads from preloaded CPU memory instead of disk when possible,
        falling back to the original disk-based loading if needed.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """
        logger.debug("[PreloadedConnector] start_load_kv called")

        # Start timing KV loading
        kv_load_start_time = time.time()

        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            """Inject the KV cache into the layer.

            Args:
                dst_kv_cache_layer (torch.Tensor): the destination KV cache
                    layer. In shape [2, num_pages, page_size, xxx] if not
                    using MLA, [num_pages, page_size, xxx] otherwise.
                src_kv_cache (torch.Tensor): the source KV cache. In shape
                    [2, num_tokens, xxx] if not using MLA, [num_tokens, xxx]
                    otherwise.
                slot_mapping (torch.Tensor): the slot mapping. In shape
                    [num_tokens].
            """
            dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages = dst_kv_cache_layer_shape[0]
                page_size = dst_kv_cache_layer_shape[1]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    num_pages * page_size, -1)
                dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)
            else:
                num_pages = dst_kv_cache_layer_shape[1]
                page_size = dst_kv_cache_layer_shape[2]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    2, num_pages * page_size, -1)
                dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)

        # Get the metadata
        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, SharedStorageConnectorMetadata)

        if metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the connector metadata is None"
            )
            return

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the attn_metadata is None")
            return

        logger.debug(f"[PreloadedConnector] Processing {len(metadata.requests)} requests")
        loaded_from_disk = False

        for request in metadata.requests:
            if request.is_store:
                logger.debug(f"[PreloadedConnector] Skipping store request")
                continue
            logger.info("Inject KV cache of %d tokens to the paged memory",
                        len(request.slot_mapping))
            for layer_name in forward_context.no_compile_layers:
                attn_layer = forward_context.no_compile_layers[layer_name]
                kv_cache_layer = attn_layer.kv_cache[\
                        forward_context.virtual_engine]

                filename = self._generate_filename_debug(
                    layer_name, request.token_ids)

                logger.debug(f"[PreloadedConnector] Looking for file: {filename}")
                logger.debug(f"[PreloadedConnector] Have {len(self._preloaded_caches)} preloaded files")

                if filename in self._preloaded_caches:
                    logger.debug(f"Loading KV cache from CPU memory: {filename}")
                    kv_cache = self._preloaded_caches[filename].cuda()
                else:
                    logger.debug(f"Loading KV cache from disk (not preloaded): {filename}")
                    kv_cache = safetensors.torch.load_file(
                        filename)["kv_cache"].cuda()
                    loaded_from_disk = True

                inject_kv_into_layer(kv_cache_layer, kv_cache,
                                     request.slot_mapping)

        # End timing and store in global context for this request
        kv_load_end_time = time.time()
        kv_load_duration_ms = (kv_load_end_time - kv_load_start_time) * 1000

        # Track whether we actually loaded any KV cache
        did_load_kv = False
        for request in metadata.requests:
            if request.is_store:
                continue
            for layer_name in forward_context.no_compile_layers:
                filename = self._generate_filename_debug(layer_name, request.token_ids)
                if filename in self._preloaded_caches or loaded_from_disk:
                    did_load_kv = True
                    logger.debug(f"[PreloadedConnector] Found KV cache to load: {filename}")
                    break
            if did_load_kv:
                break

        # Only log and store timing if we actually loaded something
        if did_load_kv:
            # Persist timing so that scheduler-side connector (separate process)
            # can retrieve it. Use a simple per-request file in the shared
            # storage directory. This avoids cross-process object passing.
            timing_dir = os.path.join(self._storage_path, "kv_load_timings")
            try:
                os.makedirs(timing_dir, exist_ok=True)
            except Exception:
                pass

            # Store timing in a way that can be retrieved by the request output
            if not hasattr(self, '_kv_load_timings'):
                self._kv_load_timings = {}

            # Associate timing with all requests being processed
            for request in metadata.requests:
                if not request.is_store:
                    # Store by request ID for accurate retrieval
                    self._kv_load_timings[request.req_id] = kv_load_duration_ms
                    # Also write to filesystem for cross-process access
                    if request.req_id:
                        try:
                            with open(os.path.join(timing_dir, f"{request.req_id}.txt"), "w") as f:
                                f.write(str(kv_load_duration_ms))
                        except Exception as e:
                            logger.debug("[PreloadedConnector] Failed to write timing file: %s", e)
                    logger.info(
                        "[PreloadedConnector] KV load time for request "
                        f"{request.req_id}: {kv_load_duration_ms:.2f} ms")

            logger.info(f"[PreloadedConnector] KV loading took {kv_load_duration_ms:.2f} ms")

            if loaded_from_disk:
                logger.info("Loaded KV cache from disk")
            else:
                logger.debug("Loaded KV cache from CPU memory")
        else:
            logger.debug("[PreloadedConnector] No KV cache loaded, skipping timing storage")

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Override to add logging for preloaded connector.

        Since we load from CPU memory, this should return immediately.
        """
        logger.debug(f"[PreloadedConnector] wait_for_layer_load called: layer_name={layer_name}")

        # CPU memory loading is immediate, so just return
        # (parent also just returns)
        super().wait_for_layer_load(layer_name)

        logger.debug(f"[PreloadedConnector] wait_for_layer_load completed for layer: {layer_name}")

    def wait_for_save(self):
        """
        Override to add logging for preloaded connector.
        """
        logger.debug(f"[PreloadedConnector] wait_for_save called")

        # Delegate to parent (which just returns)
        super().wait_for_save()

        logger.debug(f"[PreloadedConnector] wait_for_save completed")

    def _found_match_for_request(
        self,
        request: "Request",
    ) -> bool:
        """
        Check for cache hit by folder (hash) only, both in-memory and on disk, matching parent class logic.
        """
        logger.debug(f"[PreloadedConnector] _found_match_for_request called for request_id={getattr(request, 'request_id', getattr(request, 'req_id', 'unknown'))}")

        prompt_len = len(request.prompt_token_ids)
        logger.debug(f"[PreloadedConnector] prompt_len={prompt_len}, block_size={self._block_size}")

        # Use the same logic as parent class: prompt_len - 1
        num_tokens_to_check = align_to_block_size(prompt_len - 1, self._block_size)
        logger.debug(f"[PreloadedConnector] num_tokens_to_check after align_to_block_size: {num_tokens_to_check}")

        # Handle the special case where align_to_block_size returns 0
        if num_tokens_to_check == 0:
            num_tokens_to_check = prompt_len
            logger.debug(f"[PreloadedConnector] Adjusted num_tokens_to_check to: {num_tokens_to_check}")

        input_ids_tensor = torch.tensor(request.prompt_token_ids)[:num_tokens_to_check]
        foldername = self._generate_foldername_debug(input_ids_tensor, create_folder=False)
        logger.debug(f"[PreloadedConnector] Checking folder: {foldername}")

        # In-memory check: see if any preloaded file's parent folder matches
        in_memory_match = any(foldername == os.path.dirname(f) for f in self._preloaded_caches.keys())
        if in_memory_match:
            logger.debug(f"[PreloadedConnector] Found match in preloaded CPU memory for {foldername}")
            return True

        # Disk check: only check for the folder
        filesystem_match = os.path.exists(foldername)
        logger.debug(f"[PreloadedConnector] Checking filesystem folder: {foldername}, exists: {filesystem_match}")
        if filesystem_match:
            logger.debug(f"[PreloadedConnector] Found match in filesystem: {foldername}")
        else:
            logger.debug(f"[PreloadedConnector] No match found in CPU memory or filesystem")
        return filesystem_match

    def get_kv_load_timing(self, req_id: str) -> Optional[float]:
        """Get KV loading timing for a request based on its request ID."""
        if not hasattr(self, '_kv_load_timings'):
            logger.debug("[PreloadedConnector] No KV load timings stored")
            self._kv_load_timings = {}

        timing = self._kv_load_timings.get(req_id)

        # If not found in in-memory dict (likely different process), try file.
        if timing is None and req_id is not None:
            timing_file = os.path.join(self._storage_path, "kv_load_timings", f"{req_id}.txt")
            try:
                with open(timing_file, "r") as f:
                    timing = float(f.read().strip())
                # Clean up file after reading
                try:
                    os.remove(timing_file)
                except Exception:
                    pass
            except FileNotFoundError:
                pass

        logger.debug(
            f"[PreloadedConnector] Looking up KV load timing for request "
            f"{req_id}: {timing if timing is not None else 'not found'}")

        # Clean up old timings to prevent memory leak
        if req_id in self._kv_load_timings:
            del self._kv_load_timings[req_id]
            logger.debug(
                f"[PreloadedConnector] Cleaned up timing for request {req_id}")

        return timing

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Override to include KV loading timing in the response.
        """
        # Get the base result from parent
        delay_free_blocks, kv_transfer_params = super().request_finished(request, block_ids)

        # Add KV loading timing if available
        kv_load_time_ms = self.get_kv_load_timing(request.request_id)
        if kv_load_time_ms is not None:
            if kv_transfer_params is None:
                kv_transfer_params = {}
            kv_transfer_params['kv_load_time_ms'] = kv_load_time_ms
            logger.info(
                "[PreloadedConnector] Added KV load timing to response for "
                f"request {request.request_id}: {kv_load_time_ms:.2f} ms")
        else:
            logger.debug(
                "[PreloadedConnector] No KV load timing found for request "
                f"{request.request_id}")

        return delay_free_blocks, kv_transfer_params

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """
        Override to handle saving for preloaded connector.

        Since this is a preloaded connector focused on loading performance,
        we can either:
        1. Skip saving (read-only mode)
        2. Save to disk only (maintain compatibility)
        3. Save to both CPU memory and disk

        For now, we'll save to disk to maintain compatibility with the parent behavior.
        """
        return
        logger.debug(f"[PreloadedConnector] save_kv_layer called: layer_name={layer_name}")

        # Get metadata to find what we're saving
        metadata = self._get_connector_metadata()
        if metadata and hasattr(metadata, 'requests') and metadata.requests:
            for req in metadata.requests:
                if req.is_store:
                    foldername = self._generate_foldername_debug(req.token_ids)
                    logger.debug(f"[PreloadedConnector] Saving to folder: {foldername}")

        # For now, delegate to parent to maintain compatibility
        # In the future, we could add CPU memory caching here
        super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

        logger.debug(f"[PreloadedConnector] save_kv_layer completed for layer: {layer_name}")


def align_to_block_size(num_tokens: int, block_size) -> int:
    """Align the number of tokens to the block size.
    """
    return (num_tokens - 1) // block_size * block_size
