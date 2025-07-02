#!/usr/bin/env bash
# Benchmark the decode-only server with PreloadedSharedStorageConnector (online requests) - KV cache load time exclusion.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"


python3 benchmarks/benchmark_decode_latency_no_kv.py \
  --server-url http://localhost:8002 \
  --manifest test_prefill_output_preloaded/manifest.json \
  --model /home/ppatel-ext-l/models/mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --num-requests 10 \
  --max-tokens 16 \
  --request-rate 5 \
  --ignore-eos \
  --save-result \
  --result-dir results/ \
  --result-filename kvexclusion-decode-results.json \
  --stream
  #--model /home/ppatel-ext-l/models/meta-llama/Llama-2-7b-hf \
  #--num-requests 500 \
  #--max-tokens 128 \