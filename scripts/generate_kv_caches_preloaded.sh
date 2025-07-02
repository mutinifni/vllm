#!/usr/bin/env bash
# Generate KV caches & manifest for disaggregated decode benchmarking with PreloadedSharedStorageConnector.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Creates ./test_prefill_output_preloaded/{local_storage,manifest.json}

python3 benchmarks/generate_kv_cache_manifest.py \
	--model /home/ppatel-ext-l/models/mistralai/Mixtral-8x7B-Instruct-v0.1 \
	--dataset-name random \
	--num-prompts 500 \
	--random-input-len 1024 \
	--output-dir test_prefill_output_preloaded \
	--tensor-parallel-size 4 \
    --no-enable-chunked-prefill
	#--model /home/ppatel-ext-l/models/meta-llama/Llama-2-7b-hf \
