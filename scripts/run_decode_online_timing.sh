#!/usr/bin/env bash
# Benchmark the decode-only server with KV loading time separation (online requests).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"


python3 benchmarks/benchmark_disagg_decode_timing.py \
  --server-url http://localhost:8002 \
  --manifest test_prefill_output_preloaded/manifest.json \
  --model /home/ppatel-ext-l/models/mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --num-requests 10 \
  --max-tokens 16 \
  --timing-mode server \
  --request-rate 5 \
  --stream-options "include_usage=true" \
  --ignore-eos \
  --save-result \
  --result-dir results/ \
  --result-filename timing-decode-results.json
  #--model /home/ppatel-ext-l/models/meta-llama/Llama-2-7b-hf \
