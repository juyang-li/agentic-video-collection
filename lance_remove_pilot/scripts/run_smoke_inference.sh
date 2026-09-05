#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PILOT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNTIME_ROOT="$PILOT_ROOT/runtime/upstream"
ENV_PREFIX=${ENV_PREFIX:-/scratch/prj0000000275/jcy/.conda/envs/xks_lance}

RUN_NAME=${RUN_NAME:-smoke_mixed}
EXPECTED_STEPS=${EXPECTED_STEPS:-3}
CHECKPOINT_STEP=$((EXPECTED_STEPS - 1))
CHECKPOINT_DIR=$(printf '%s/runs/%s/checkpoints/%07d' "$PILOT_ROOT" "$RUN_NAME" "$CHECKPOINT_STEP")
OUTPUT_DIR=$(printf '%s/runs/%s/inference_step_%07d' "$PILOT_ROOT" "$RUN_NAME" "$CHECKPOINT_STEP")
CONFIG_PATH="$PILOT_ROOT/configs/smoke_inference.json"
LOG_PATH=$(printf '%s/runs/%s/inference_step_%07d.log' "$PILOT_ROOT" "$RUN_NAME" "$CHECKPOINT_STEP")

if [[ ! -s "$CHECKPOINT_DIR/model.safetensors" ]]; then
    echo "Missing fine-tuned checkpoint: $CHECKPOINT_DIR/model.safetensors" >&2
    exit 2
fi
if [[ ! -s "$PILOT_ROOT/media/insave/insave-11855_source.mp4" ]]; then
    echo "Missing fixed inference input video" >&2
    exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite inference output: $OUTPUT_DIR" >&2
    exit 2
fi

export PATH="$ENV_PREFIX/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$RUNTIME_ROOT"
bash inference_lance.sh \
    --NUM_GPUS 1 \
    --TASK_NAME video_edit \
    --MODEL_PATH "$CHECKPOINT_DIR" \
    --CONFIG_PATH "$CONFIG_PATH" \
    --SAVE_PATH_GEN "$OUTPUT_DIR" \
    --VALIDATION_NUM_TIMESTEPS 12 \
    --VALIDATION_TIMESTEP_SHIFT 3.5 \
    --VALIDATION_DATA_SEED 42 \
    --CFG_TEXT_SCALE 4.0 \
    --NUM_FRAMES 50 \
    --VIDEO_HEIGHT 480 \
    --VIDEO_WIDTH 480 \
    --RESOLUTION video_480p \
    --USE_KVCACHE true \
    --ENHANCE_PROMPT false \
    2>&1 | tee "$LOG_PATH"

"$ENV_PREFIX/bin/python" "$PILOT_ROOT/scripts/verify_smoke_inference.py" "$OUTPUT_DIR"
