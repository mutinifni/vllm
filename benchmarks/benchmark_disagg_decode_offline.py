# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import sys
import json
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
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
import requests
import time
import subprocess
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch  # Add this import for tensor serialization

def _serialize_attn_metadata(attn_metadata):
    """Convert AttentionMetadata to a JSON-serializable dict."""
    if attn_metadata is None:
        return None
    # Use asdict_zerocopy if available, else fallback to __dict__
    if hasattr(attn_metadata, 'asdict_zerocopy'):
        d = attn_metadata.asdict_zerocopy()
    else:
        d = dict(attn_metadata.__dict__)
    # Convert torch.Tensors to lists
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.cpu().tolist()
    return d

def _deserialize_attn_metadata(d):
    """Convert dict back to AttentionMetadata if needed (not implemented)."""
    # For now, just return the dict; actual object construction would require class info
    return d

def decode_benchmark(args):
    """
    1. Read manifest
    2. Instantiate LLM with kv_transfer_config (SharedStorageConnector, kv_both, local_storage)
    3. Call .generate() with all prompts at once (continuous batching)
    4. Collect and report metrics
    """
    import time
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    with open(args.manifest, "r") as f:
        manifest = json.load(f)
    manifest = manifest[:args.num_requests]
    prompts = [entry["prompt"] for entry in manifest]
    attn_metadatas = [
        _deserialize_attn_metadata(entry.get("attn_metadata")) for entry in manifest
    ]

    print(f"[Decode] Using {len(prompts)} prompts from manifest.")

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=args.max_tokens)
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        kv_transfer_config=KVTransferConfig(
            kv_connector="SharedStorageConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": args.kv_cache_dir},
        ),
    )

    start_time = time.time()
    # Try to pass attn_metadatas if supported, else fallback
    try:
        outputs = llm.generate(prompts, sampling_params, attn_metadatas=attn_metadatas)
    except TypeError:
        outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start_time

    latencies = []
    for i, output in enumerate(outputs):
        latency = output.finished_time - output.start_time if hasattr(output, 'finished_time') and hasattr(output, 'start_time') else None
        latencies.append(latency if latency is not None else 0.0)
        print(f"[Decode] Prompt {i}: latency={latencies[-1]:.3f}s, output={output.outputs[0].text[:40]!r}")

    print("\n==== Decode Benchmark Results ====")
    print(f"Total requests: {len(prompts)}")
    print(f"Total time: {total_time:.2f}s")
    if latencies:
        print(f"Mean latency per request: {sum(latencies)/len(latencies):.3f}s")
        print(f"Median latency per request: {sorted(latencies)[len(latencies)//2]:.3f}s")
        print(f"Throughput: {len(prompts)/total_time:.2f} requests/sec")
    print("===============================\n")

def main():
    parser = argparse.ArgumentParser(description="Benchmark decode-only vLLM instance with prefilled KV caches.")
    # Only keep decode benchmark subcommand
    decode_parser = parser.add_argument_group("decode benchmark options")
    parser.add_argument("--kv-cache-dir", type=str, required=True, help="Path to local_storage dir from prefill phase.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest file from prefill phase.")
    parser.add_argument("--batch-size", type=int, default=1, help="(Unused, for API compatibility)")
    parser.add_argument("--num-requests", type=int, default=100, help="Number of decode requests to send.")
    parser.add_argument("--model", type=str, required=True, help="Model name (for LLM instantiation).")
    parser.add_argument("--max-tokens", type=int, default=16, help="Number of tokens to generate in decode phase.")
    parser.add_argument("--max-concurrency", type=int, default=None, help="Maximum number of concurrent decode requests.")
    args = parser.parse_args()
    decode_benchmark(args)

if __name__ == "__main__":
    main()