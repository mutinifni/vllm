#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Start a vLLM OpenAI-compatible server for *decode-only* latency benchmarking.
#
# Usage (override any env variable as needed):
#   MODEL=/path/to/model TP_SIZE=4 PORT=8001 ./scripts/start_decode_server.sh
#
# Environment variables (all optional):
#   MODEL   – HF model name or local path.
#             Default: /home/ppatel-ext-l/models/mistralai/Mixtral-8x7B-Instruct-v0.1
#   TP_SIZE – Tensor parallel size.               Default: 4
#   PORT    – HTTP port for the OpenAI endpoint.  Default: 8001
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Ensure repo root on PYTHONPATH so sitecustomize.py is discoverable in forked processes
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MODEL="${MODEL:-/home/ppatel-ext-l/models/mistralai/Mixtral-8x7B-Instruct-v0.1}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-8001}"
export VLLM_BATCH_ITER_PROF="1"

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size "$TP_SIZE" \
  --disable-log-requests \
  --port "$PORT"

