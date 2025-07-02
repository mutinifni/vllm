#!/usr/bin/env bash
# Benchmark the decode-only server (online requests).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"


python3 benchmarks/benchmark_disagg_decode_online.py \
  --server-url http://localhost:8001 \
  --manifest test_prefill_output/manifest.json \
  --model /home/ppatel-ext-l/models/meta-llama/Llama-2-7b-hf \
  --num-requests 10 \
  --max-tokens 128 \
  --request-rate 3 \
  --ignore-eos \
  --save-result \
  --result-dir results/
  #--model /home/ppatel-ext-l/models/mistralai/Mixtral-8x7B-Instruct-v0.1 \

