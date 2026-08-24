#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: run_benchmark.sh CONFIG_PATH RUN_TAG" >&2
  exit 2
fi

EVAL_ROOT="/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/lance_base"
UPSTREAM="$EVAL_ROOT/runtime/upstream"
ENV_PREFIX="/scratch/prj0000000275/jcy/.conda/envs/xks_lance"
CONFIG_PATH="$1"
RUN_TAG="$2"
OUTPUT_DIR="$EVAL_ROOT/outputs_raw/$RUN_TAG"

export PATH="$ENV_PREFIX/bin:$PATH"
export HF_HOME="/scratch/prj0000000275/jcy/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIR" "$EVAL_ROOT/logs"
if find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.mp4' -print -quit | grep -q .; then
  echo "Refusing to reuse non-empty output directory: $OUTPUT_DIR" >&2
  exit 1
fi
exec > >(tee "$EVAL_ROOT/logs/$RUN_TAG.log") 2>&1

cd "$UPSTREAM"
bash inference_lance.sh \
  --TASK_NAME video_edit \
  --MODEL_PATH downloads/Lance_3B_Video \
  --CONFIG_PATH "$CONFIG_PATH" \
  --SAVE_PATH_GEN "$OUTPUT_DIR" \
  --VALIDATION_NUM_TIMESTEPS 30 \
  --VALIDATION_TIMESTEP_SHIFT 3.5 \
  --VALIDATION_DATA_SEED 42 \
  --CFG_TEXT_SCALE 4.0 \
  --NUM_FRAMES 50 \
  --VIDEO_HEIGHT 480 \
  --VIDEO_WIDTH 480 \
  --RESOLUTION video_480p \
  --USE_KVCACHE true \
  --ENHANCE_PROMPT false

python "$EVAL_ROOT/scripts/verify_outputs.py" "$CONFIG_PATH" "$OUTPUT_DIR"
