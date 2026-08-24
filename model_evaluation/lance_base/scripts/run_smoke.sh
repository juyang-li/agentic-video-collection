#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT="/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/lance_base"
UPSTREAM="$EVAL_ROOT/runtime/upstream"
ENV_PREFIX="/scratch/prj0000000275/jcy/.conda/envs/xks_lance"
OUTPUT_DIR="$EVAL_ROOT/outputs_raw/smoke"
CONFIG_PATH="$EVAL_ROOT/configs/smoke.json"

export PATH="$ENV_PREFIX/bin:$PATH"
export HF_HOME="/scratch/prj0000000275/jcy/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIR" "$EVAL_ROOT/logs"
exec > >(tee "$EVAL_ROOT/logs/smoke.log") 2>&1

cd "$UPSTREAM"

python -c 'import torch; print({"cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})'

bash inference_lance.sh \
  --TASK_NAME video_edit \
  --MODEL_PATH downloads/Lance_3B_Video \
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
  --ENHANCE_PROMPT false

shopt -s nullglob
outputs=("$OUTPUT_DIR"/*.mp4)
if [ "${#outputs[@]}" -ne 1 ]; then
  echo "Expected exactly one smoke MP4, found ${#outputs[@]}" >&2
  exit 1
fi
