#!/usr/bin/env bash
set -euo pipefail

runtime_root=${BERNINI_RUNTIME_ROOT:-/tmp/omni_bernini_20260926}
inputs=${1:?usage: run_inference.sh TASKS_JSON GPU_COUNT ULYSSES}
gpu_count=${2:?usage: run_inference.sh TASKS_JSON GPU_COUNT ULYSSES}
ulysses=${3:?usage: run_inference.sh TASKS_JSON GPU_COUNT ULYSSES}
width=${BERNINI_WIDTH:-848}
height=${BERNINI_HEIGHT:-480}

export HF_HOME="$runtime_root/hf-cache"
export TORCH_HOME="$runtime_root/torch-cache"
export TRITON_CACHE_DIR="$runtime_root/triton-cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$runtime_root/Bernini${PYTHONPATH:+:$PYTHONPATH}"

cd "$runtime_root/Bernini"
"$runtime_root/env/bin/torchrun" --standalone --nproc-per-node "$gpu_count" \
  infer_multi_gpu.py \
  --config "$runtime_root/Bernini-R-Diffusers" \
  --ulysses "$ulysses" \
  --inputs "$inputs" \
  --guidance_mode v2v_apg \
  --num_frames 81 --height "$height" --width "$width" --max_image_size 848 \
  --num_inference_steps 40 --flow_shift 5.0 --seed 42 --fps 16
