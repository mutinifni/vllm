#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# NOTE: We reuse the same argument definitions as `vllm serve` by importing
# EngineArgs.add_cli_args (and AsyncEngineArgs.add_cli_args) which are also
# used under the hood by `vllm/entrypoints/cli/serve.py`.  This guarantees
# parity between cache-generation and server startup, and means any new
# engine/server option added to vLLM will automatically be picked up here
# without manual maintenance.

import argparse
import os
import json
import torch
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs  # provides add_cli_args & from_cli_args
from vllm.transformers_utils.tokenizer import get_tokenizer
from benchmark_dataset import (
    AIMODataset,
    ASRDataset,
    BurstGPTDataset,
    ConversationDataset,
    CustomDataset,
    HuggingFaceDataset,
    InstructCoderDataset,
    MTBenchDataset,
    NextEditPredictionDataset,
    RandomDataset,
    SampleRequest,
    ShareGPTDataset,
    SonnetDataset,
    VisionArenaDataset,
)
# Use vLLM's FlexibleArgumentParser so EngineArgs.add_cli_args can add
# arguments marked as `deprecated` without argparse raising errors.
from vllm.utils import FlexibleArgumentParser

def _serialize_attn_metadata(attn_metadata):
    """Convert AttentionMetadata to a JSON-serializable dict."""
    if attn_metadata is None:
        return None
    if hasattr(attn_metadata, 'asdict_zerocopy'):
        d = attn_metadata.asdict_zerocopy()
    else:
        d = dict(attn_metadata.__dict__)
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.cpu().tolist()
    return d

def prefill(args, engine_args):
    os.makedirs(args.output_dir, exist_ok=True)
    local_storage_path = os.path.join(args.output_dir, "local_storage")
    os.makedirs(local_storage_path, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "manifest.json")

    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = get_tokenizer(
        tokenizer_id,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )

    if args.dataset_name == "custom":
        dataset = CustomDataset(dataset_path=args.dataset_path)
        input_requests = dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.custom_output_len,
            skip_chat_template=args.custom_skip_chat_template,
        )
    elif args.dataset_name == "sonnet":
        dataset = SonnetDataset(dataset_path=args.dataset_path)
        input_requests = dataset.sample(
            num_requests=args.num_prompts,
            input_len=args.sonnet_input_len,
            output_len=args.sonnet_output_len,
            prefix_len=args.sonnet_prefix_len,
            tokenizer=tokenizer,
            return_prompt_formatted=True,
        )
    elif args.dataset_name == "hf":
        if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = VisionArenaDataset
            args.hf_split = "train"
            args.hf_subset = None
        elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = InstructCoderDataset
            args.hf_split = "train"
        elif args.dataset_path in MTBenchDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = MTBenchDataset
            args.hf_split = "train"
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = ConversationDataset
        elif args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS:
            dataset_class = AIMODataset
            args.hf_split = "train"
        elif args.dataset_path in NextEditPredictionDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = NextEditPredictionDataset
            args.hf_split = "train"
        elif args.dataset_path in ASRDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = ASRDataset
            args.hf_split = "train"
        else:
            raise ValueError(f"Unsupported dataset path: {args.dataset_path}")
        input_requests = dataset_class(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            random_seed=args.seed,
        ).sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.hf_output_len,
        )
    else:
        dataset_mapping = {
            "sharegpt": lambda: ShareGPTDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                output_len=args.sharegpt_output_len,
            ),
            "burstgpt": lambda: BurstGPTDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(tokenizer=tokenizer, num_requests=args.num_prompts),
            "random": lambda: RandomDataset(dataset_path=args.dataset_path).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.random_prefix_len,
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                range_ratio=args.random_range_ratio,
            ),
        }
        try:
            input_requests = dataset_mapping[args.dataset_name]()
        except KeyError as err:
            raise ValueError(f"Unknown dataset: {args.dataset_name}") from err

    print(f"[Prefill] Using {len(input_requests)} prompts from dataset '{args.dataset_name}'.")

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
    llm = LLM(
        model=args.model,
        # Tokenizer options
        tokenizer=args.tokenizer,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,

        # Parallelism / memory options
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        dtype=args.dtype,
        quantization=args.quantization,
        download_dir=args.download_dir,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        block_size=args.block_size,
        # Disaggregated prefill requires full KV cache; disable chunked prefill.
        enable_chunked_prefill=False,

        # Keep the KV cache settings pointing to the shared storage so the
        # generated caches can later be consumed by the decode-only server.
        kv_transfer_config=KVTransferConfig(
            kv_connector="SharedStorageConnector",
            #kv_role="kv_both",
            kv_role="kv_producer",
            kv_connector_extra_config={"shared_storage_path": local_storage_path},
        ),
    )

    manifest = []
    prompts = [req.prompt for req in input_requests]
    outputs = llm.generate(prompts, sampling_params)
    for i, (output, req) in enumerate(zip(outputs, input_requests)):
        prompt = output.prompt
        attn_metadata = getattr(output, "attn_metadata", None)
        manifest.append({
            "prompt_id": i,
            "prompt": prompt,
            "prompt_len": req.prompt_len,
            "output_len": req.expected_output_len,
            "attn_metadata": _serialize_attn_metadata(attn_metadata),
        })
        print(f"[Prefill] Prompt {i}: {prompt[:60]!r}...")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Prefill] Saved manifest to {manifest_path}")
    print(f"[Prefill] KV caches saved in {local_storage_path}")

def main():
    # ------------------------------------------------------------------
    # Engine/server args – exactly the same set as accepted by `vllm serve`.
    # ------------------------------------------------------------------
    parser = EngineArgs.add_cli_args(
        FlexibleArgumentParser(
            description="Generate KV caches and manifest for disaggregated decode benchmarking. All engine/server options from 'vllm serve -h' are accepted here as-is."
        )
    )

    # ------------------------------------------------------------------
    # Additional dataset / script-specific args.
    # ------------------------------------------------------------------
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save KV caches and manifest.")
    parser.add_argument("--num-prompts", type=int, default=100, help="Number of prompts to prefill.")
    parser.add_argument("--dataset-name", type=str, default="sharegpt", choices=["sharegpt", "burstgpt", "sonnet", "random", "hf", "custom"], help="Name of the dataset to benchmark on.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to the dataset or HF dataset ID.")
    parser.add_argument("--custom-output-len", type=int, default=256, help="Custom dataset: output tokens per request.")
    parser.add_argument("--custom-skip-chat-template", action="store_true", help="Custom dataset: skip chat template.")
    parser.add_argument("--sonnet-input-len", type=int, default=550, help="Sonnet dataset: input tokens per request.")
    parser.add_argument("--sonnet-output-len", type=int, default=150, help="Sonnet dataset: output tokens per request.")
    parser.add_argument("--sonnet-prefix-len", type=int, default=200, help="Sonnet dataset: prefix tokens per request.")
    parser.add_argument("--sharegpt-output-len", type=int, default=None, help="ShareGPT: output length override.")
    parser.add_argument("--random-input-len", type=int, default=1024, help="Random dataset: input tokens per request.")
    parser.add_argument("--random-output-len", type=int, default=128, help="Random dataset: output tokens per request.")
    parser.add_argument("--random-range-ratio", type=float, default=0.0, help="Random dataset: range ratio for input/output length.")
    parser.add_argument("--random-prefix-len", type=int, default=0, help="Random dataset: fixed prefix tokens.")
    parser.add_argument("--hf-subset", type=str, default=None, help="HF dataset: subset.")
    parser.add_argument("--hf-split", type=str, default=None, help="HF dataset: split.")
    parser.add_argument("--hf-output-len", type=int, default=None, help="HF dataset: output length override.")
    args = parser.parse_args()

    # EngineArgs.from_cli_args will ignore any unknown (dataset-specific)
    # arguments, so we create a shallow copy without them when constructing
    # the engine. We still keep the full `args` for dataset handling.
    engine_cli_namespace = argparse.Namespace(**{k: v for k, v in vars(args).items() if k in EngineArgs.__dataclass_fields__})
    engine_args = EngineArgs.from_cli_args(engine_cli_namespace)

    # Pass engine_args to prefill for unified config
    prefill(args, engine_args)

if __name__ == "__main__":
    main()