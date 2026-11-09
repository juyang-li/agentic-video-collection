#!/usr/bin/env bash
set -euo pipefail

STAGE3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERAL_DIR="$(cd "$STAGE3_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$GENERAL_DIR/.." && pwd)"
QWEN_PYTHON_BIN="${QWEN_PYTHON_BIN:-/home/users/astar/cfar/stuchengyou/jcy/.conda/envs/xks_lance/bin/python}"
SYSTEM_PYTHON_BIN="${SYSTEM_PYTHON_BIN:-python}"
BENCHMARK_DIR="$STAGE3_DIR/artifacts/cascade_benchmark_v1"
LABEL_DIR="$GENERAL_DIR/2_vlm_filtering/artifacts/systematic_binary_benchmark_v1/labels"
GEMINI_DIR="$GENERAL_DIR/2_vlm_filtering/artifacts/systematic_binary_benchmark_v1/gemini_flash_eval"
QWEN_SCORES="$BENCHMARK_DIR/qwen72b_prompt_v12_scores.csv"

if [[ -z "${LIJUYANG_ICLR:-}" ]]; then
  echo "LIJUYANG_ICLR is not set" >&2
  exit 2
fi
if [[ ! -x "$QWEN_PYTHON_BIN" ]]; then
  echo "QWEN_PYTHON_BIN is not executable: $QWEN_PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$BENCHMARK_DIR"
RESUME_ARGS=()
if [[ -f "$QWEN_SCORES" ]]; then
  RESUME_ARGS=(--resume)
fi

cd "$PROJECT_DIR"
"$QWEN_PYTHON_BIN" "$STAGE3_DIR/run_qwen72b_prompt_v12.py" \
  --input "$LABEL_DIR/explicit_binary_v12_input.csv" \
  --output "$QWEN_SCORES" \
  --summary "$BENCHMARK_DIR/qwen72b_prompt_v12_summary.json" \
  --api-key-env LIJUYANG_ICLR \
  --provider-order Parasail \
  --concurrency 4 \
  "${RESUME_ARGS[@]}"

"$SYSTEM_PYTHON_BIN" "$STAGE3_DIR/compose_and_evaluate_cascade.py" \
  --manifest "$LABEL_DIR/explicit_binary_v12_input.csv" \
  --labels "$LABEL_DIR/explicit_binary_dataset.csv" \
  --qwen-scores "$QWEN_SCORES" \
  --gemini-scores "$GEMINI_DIR/train_v1_scores.csv" \
  --gemini-scores "$GEMINI_DIR/validation_v1_scores.csv" \
  --gemini-scores "$GEMINI_DIR/test_selected_v1_scores.csv" \
  --output "$BENCHMARK_DIR/cascade_benchmark_rows.csv" \
  --report "$BENCHMARK_DIR/cascade_benchmark_report.json"
