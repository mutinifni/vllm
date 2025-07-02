#!/usr/bin/env bash
# Start the decode-only server with PreloadedSharedStorageConnector that consumes the prefilled KV cache.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"


python3 benchmarks/benchmark_disagg_decode_server_preloaded.py \
        --model /home/ppatel-ext-l/models/mistralai/Mixtral-8x7B-Instruct-v0.1 \
  		--kv-cache-dir test_prefill_output_preloaded/local_storage \
  		--manifest test_prefill_output_preloaded/manifest.json \
  		--tensor-parallel-size 4 \
  		--port 8002 \
        --disable-log-requests
        #--model /home/ppatel-ext-l/models/meta-llama/Llama-2-7b-hf \
