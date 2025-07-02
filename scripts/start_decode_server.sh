#!/usr/bin/env bash
# Start the decode-only server that consumes the prefilled KV cache.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"


python3 benchmarks/benchmark_disagg_decode_server.py \
        --model /home/ppatel-ext-l/models/meta-llama/Llama-2-7b-hf \
  		--kv-cache-dir test_prefill_output/local_storage \
  		--tensor-parallel-size 2 \
  		--port 8001 \
        #--model /home/ppatel-ext-l/models/mistralai/Mixtral-8x7B-Instruct-v0.1 \

