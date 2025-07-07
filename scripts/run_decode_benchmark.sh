#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the steady-state decode latency benchmark against a running vLLM server.
#
# Usage (override env vars as desired):
#   MODEL=/home/ppatel-ext-l/models/mistralai/Mixtral-8x7B-Instruct-v0.1 \
#   BATCH_SIZE=32 INPUT_LEN=1024 OUTPUT_LEN=128 PORT=8001 \
#     ./scripts/run_decode_benchmark.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Configurable parameters via environment variables
# ---------------------------------------------------------------------------
BACKEND="${BACKEND:-vllm}"
MODEL="${MODEL:-/home/ppatel-ext-l/models/mistralai/Mixtral-8x7B-Instruct-v0.1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"  # leave empty to default to MODEL
BATCH_SIZE="${BATCH_SIZE:-32}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
SKIP_INITIAL_TOKENS="${SKIP_INITIAL_TOKENS:-2}"
WARMUP_BATCHES="${WARMUP_BATCHES:-1}"
MEASURE_BATCHES="${MEASURE_BATCHES:-5}"
BETWEEN_BATCH_SLEEP="${BETWEEN_BATCH_SLEEP:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
ENDPOINT="${ENDPOINT:-/v1/completions}"

python3 benchmarks/benchmark_decode_only.py \
  --backend "$BACKEND" \
  --model "$MODEL" \
  ${SERVED_MODEL_NAME:+--served-model-name "$SERVED_MODEL_NAME"} \
  --batch-size "$BATCH_SIZE" \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --skip-initial-tokens "$SKIP_INITIAL_TOKENS" \
  --warmup-batches "$WARMUP_BATCHES" \
  --measure-batches "$MEASURE_BATCHES" \
  --between-batch-sleep "$BETWEEN_BATCH_SLEEP" \
  --host "$HOST" \
  --port "$PORT" \
  --endpoint "$ENDPOINT"
