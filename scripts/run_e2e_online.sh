#!/usr/bin/env bash
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Benchmark the end-to-end vLLM server started by start_e2e_server.sh.
# Uses the same workload as earlier disaggregated benchmark (random dataset,
# 1024 prompts).
# ---------------------------------------------------------------------------

python3 benchmarks/benchmark_serving.py \
  --backend vllm \
  --base-url http://localhost:8001 \
  --model /home/ppatel-ext-l/models/mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --dataset-name random \
  --num-prompts 500 \
  --random-input-len 1 \
  --random-output-len 128 \
  --request-rate 5 \
  --ignore-eos \
  --save-result \
  --result-dir results/
  #--model /home/ppatel-ext-l/models/meta-llama/Llama-2-7b-hf \
