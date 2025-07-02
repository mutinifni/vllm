#!/usr/bin/env bash
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Launch a standard vLLM OpenAI-compatible server (prefill + decode together).
# Adjust model path, TP size, and port as needed.
# ---------------------------------------------------------------------------

python3 -m vllm.entrypoints.openai.api_server \
  --model /home/ppatel-ext-l/models/mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --tensor-parallel-size 4 \
  --port 8001
  #--model /home/ppatel-ext-l/models/meta-llama/Llama-2-7b-hf \
