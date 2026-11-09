#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
    echo "Usage: run_lora_video_edit.sh EVAL_ROOT RUNTIME_ROOT MODEL_PATH ADAPTER_CHECKPOINT CONFIG_PATH RUN_NAME" >&2
    exit 2
fi

EVAL_ROOT=$(realpath "$1")
RUNTIME_ROOT=$(realpath "$2")
MODEL_PATH=$(realpath "$3")
ADAPTER_CHECKPOINT=$(realpath "$4")
CONFIG_PATH=$(realpath "$5")
RUN_NAME=$6
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGE_ROOT=$(realpath "$SCRIPT_DIR/..")
TOOLS_ROOT=$(realpath "$STAGE_ROOT/../../model_evaluation/lance_tools")
ENV_PREFIX=${ENV_PREFIX:-/scratch/prj0000000275/jcy/.conda/envs/xks_lance}
OUTPUT_DIR="$EVAL_ROOT/outputs_raw/$RUN_NAME"
LOG_PATH="$EVAL_ROOT/logs/$RUN_NAME.log"
PENDING_CONFIG="$EVAL_ROOT/configs/${RUN_NAME}_pending.json"
TECHNICAL_CSV="$EVAL_ROOT/technical/${RUN_NAME}.csv"

for path in \
    "$SCRIPT_DIR/launch_lora_video_edit_seed42.sh" \
    "$TOOLS_ROOT/prepare_pending_config.py" \
    "$TOOLS_ROOT/verify_outputs.py" \
    "$RUNTIME_ROOT/datamake_lora_inference_lance.py" \
    "$MODEL_PATH/model.safetensors" \
    "$ADAPTER_CHECKPOINT/checkpoint_complete.json" \
    "$ADAPTER_CHECKPOINT/adapter.safetensors" \
    "$CONFIG_PATH"; do
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

"$ENV_PREFIX/bin/python" "$TOOLS_ROOT/prepare_pending_config.py" \
    "$CONFIG_PATH" "$OUTPUT_DIR" "$PENDING_CONFIG" --requeue-invalid | tee -a "$LOG_PATH"

PENDING_COUNT=$("$ENV_PREFIX/bin/python" -c \
    'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' \
    "$PENDING_CONFIG")
if [[ "$PENDING_COUNT" -gt 0 ]]; then
    "$SCRIPT_DIR/launch_lora_video_edit_seed42.sh" \
        "$RUNTIME_ROOT" "$MODEL_PATH" "$ADAPTER_CHECKPOINT" \
        "$PENDING_CONFIG" "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_PATH"
fi

"$ENV_PREFIX/bin/python" "$TOOLS_ROOT/verify_outputs.py" \
    "$CONFIG_PATH" "$OUTPUT_DIR" --inventory "$TECHNICAL_CSV" | tee -a "$LOG_PATH"
