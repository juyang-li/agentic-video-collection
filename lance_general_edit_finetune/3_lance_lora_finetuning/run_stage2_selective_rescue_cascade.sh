#!/usr/bin/env bash
set -euo pipefail

STAGE3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$STAGE3_DIR/../.." && pwd)"
QWEN_PYTHON_BIN="${QWEN_PYTHON_BIN:-/home/users/astar/cfar/stuchengyou/jcy/.conda/envs/xks_lance/bin/python}"
SYSTEM_PYTHON_BIN="${SYSTEM_PYTHON_BIN:-python}"
INPUT="${1:?usage: run_stage2_selective_rescue_cascade.sh INPUT_CSV RUN_DIR}"
RUN_DIR="${2:?usage: run_stage2_selective_rescue_cascade.sh INPUT_CSV RUN_DIR}"
QWEN_INPUT="$RUN_DIR/qwen_called_input.csv"
QWEN_CALLED="$RUN_DIR/qwen72b_called_scores.csv"
QWEN_SKIPPED="$RUN_DIR/qwen72b_skipped_scores.csv"
QWEN_SCORES="$RUN_DIR/qwen72b_prompt_v12_scores.csv"
GEMINI_INPUT="$RUN_DIR/gemini_route.csv"
GEMINI_SCORES="$RUN_DIR/gemini38_scores.csv"

if [[ -z "${LIJUYANG_ICLR:-}" ]]; then
  echo "LIJUYANG_ICLR is not set" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
cd "$PROJECT_DIR"

"$SYSTEM_PYTHON_BIN" "$STAGE3_DIR/prepare_selective_qwen_input.py" \
  --manifest "$INPUT" \
  --qwen-input "$QWEN_INPUT" \
  --skipped-scores "$QWEN_SKIPPED" \
  --summary "$RUN_DIR/selective_qwen_summary.json"

for QWEN_ATTEMPT in 1 2 3; do
  QWEN_RESUME=()
  if [[ -f "$QWEN_CALLED" ]]; then QWEN_RESUME=(--resume); fi
  "$QWEN_PYTHON_BIN" "$STAGE3_DIR/run_qwen72b_prompt_v12.py" \
    --input "$QWEN_INPUT" \
    --output "$QWEN_CALLED" \
    --summary "$RUN_DIR/qwen72b_called_summary.json" \
    --api-key-env LIJUYANG_ICLR \
    --provider-order Parasail \
    --concurrency 1 \
    --max-attempts 8 \
    --timeout 60 \
    "${QWEN_RESUME[@]}"
  QWEN_PARSE_ERRORS="$("$QWEN_PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["parse_errors"])' "$RUN_DIR/qwen72b_called_summary.json")"
  if [[ "$QWEN_PARSE_ERRORS" == "0" ]]; then break; fi
  echo "Qwen pass $QWEN_ATTEMPT left $QWEN_PARSE_ERRORS errors; retrying successful-ID-safe resume" >&2
done

"$SYSTEM_PYTHON_BIN" "$STAGE3_DIR/merge_stage2_scores.py" \
  --input "$QWEN_CALLED" \
  --input "$QWEN_SKIPPED" \
  --output "$QWEN_SCORES"

"$SYSTEM_PYTHON_BIN" "$STAGE3_DIR/prepare_gemini_route.py" \
  --manifest "$INPUT" \
  --qwen-scores "$QWEN_SCORES" \
  --output "$GEMINI_INPUT" \
  --summary "$RUN_DIR/gemini_route_summary.json"

for GEMINI_ATTEMPT in 1 2 3; do
  GEMINI_RESUME=()
  if [[ -f "$GEMINI_SCORES" ]]; then GEMINI_RESUME=(--resume); fi
  "$QWEN_PYTHON_BIN" "$PROJECT_DIR/lance_general_edit_finetune/2_vlm_filtering/scripts/run_openrouter_gemini_flash_judge.py" \
    --input "$GEMINI_INPUT" \
    --output "$GEMINI_SCORES" \
    --summary "$RUN_DIR/gemini38_summary.json" \
    --api-key-env LIJUYANG_ICLR \
    --reasoning-effort minimal \
    --max-tokens 1024 \
    --concurrency 8 \
    --timeout 60 \
    "${GEMINI_RESUME[@]}"
  GEMINI_PARSE_ERRORS="$("$QWEN_PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["parse_errors"])' "$RUN_DIR/gemini38_summary.json")"
  if [[ "$GEMINI_PARSE_ERRORS" == "0" ]]; then break; fi
  echo "Gemini pass $GEMINI_ATTEMPT left $GEMINI_PARSE_ERRORS parse errors; retrying successful-ID-safe resume" >&2
done

"$SYSTEM_PYTHON_BIN" "$STAGE3_DIR/compose_production_cascade.py" \
  --manifest "$INPUT" \
  --qwen-scores "$QWEN_SCORES" \
  --gemini-scores "$GEMINI_SCORES" \
  --output "$RUN_DIR/cascade_rows.csv" \
  --summary "$RUN_DIR/cascade_summary.json"
