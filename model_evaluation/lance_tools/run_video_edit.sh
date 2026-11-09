#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
    echo "Usage: run_video_edit.sh EVAL_ROOT RUNTIME_ROOT MODEL_PATH CONFIG_PATH RUN_NAME" >&2
    exit 2
fi

EVAL_ROOT=$(realpath "$1")
RUNTIME_ROOT=$(realpath "$2")
MODEL_PATH=$(realpath "$3")
CONFIG_PATH=$(realpath "$4")
RUN_NAME=$5
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_PREFIX=${ENV_PREFIX:-/scratch/prj0000000275/jcy/.conda/envs/xks_lance}
OUTPUT_DIR="$EVAL_ROOT/outputs_raw/$RUN_NAME"
LOG_PATH="$EVAL_ROOT/logs/$RUN_NAME.log"
PENDING_CONFIG="$EVAL_ROOT/configs/${RUN_NAME}_pending.json"
TECHNICAL_CSV="$EVAL_ROOT/technical/${RUN_NAME}.csv"

for path in "$RUNTIME_ROOT/inference_lance.sh" "$MODEL_PATH/model.safetensors" "$CONFIG_PATH"; do
    if [[ ! -s "$path" ]]; then
        echo "Missing required input: $path" >&2
        exit 2
    fi
done

mkdir -p "$OUTPUT_DIR" "$EVAL_ROOT/logs" "$EVAL_ROOT/configs" "$EVAL_ROOT/technical"
export PATH="$ENV_PREFIX/bin:$PATH"
export HF_HOME=/scratch/prj0000000275/jcy/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$ENV_PREFIX/bin/python" "$SCRIPT_DIR/prepare_pending_config.py" \
    "$CONFIG_PATH" "$OUTPUT_DIR" "$PENDING_CONFIG" | tee -a "$LOG_PATH"

PENDING_COUNT=$("$ENV_PREFIX/bin/python" -c \
    'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' \
    "$PENDING_CONFIG")
if [[ "$PENDING_COUNT" -gt 0 ]]; then
    "$SCRIPT_DIR/launch_video_edit_seed42.sh" \
        "$RUNTIME_ROOT" "$MODEL_PATH" "$PENDING_CONFIG" "$OUTPUT_DIR" \
        2>&1 | tee -a "$LOG_PATH"
fi

"$ENV_PREFIX/bin/python" "$SCRIPT_DIR/verify_outputs.py" \
    "$CONFIG_PATH" "$OUTPUT_DIR" --inventory "$TECHNICAL_CSV" | tee -a "$LOG_PATH"
