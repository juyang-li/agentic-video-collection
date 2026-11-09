#!/usr/bin/env bash
set -euo pipefail

# Run on the allocated H200 node. All rebuildable state stays on node-local disk.
runtime_root=${BERNINI_RUNTIME_ROOT:-/tmp/omni_bernini_20260926}
repo_dir="$runtime_root/Bernini"
env_dir="$runtime_root/env"
model_dir="$runtime_root/Bernini-R-Diffusers"

export PIP_CACHE_DIR="$runtime_root/pip-cache"
export HF_HOME="$runtime_root/hf-cache"
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT=1800
export CUDA_HOME=/apps/cuda/12.6.2
export PATH="$CUDA_HOME/bin:$PATH"

if [[ ! -d "$repo_dir/.git" ]]; then
  git clone --filter=blob:none https://github.com/bytedance/Bernini.git "$repo_dir"
fi
git -C "$repo_dir" fetch --depth 1 origin e6c2cf11843470b08491745947abcba9c3c6b079
git -C "$repo_dir" checkout --detach e6c2cf11843470b08491745947abcba9c3c6b079
if [[ ! -x "$env_dir/bin/python" ]]; then
  conda create -y -p "$env_dir" python=3.11.2 pip
fi
"$env_dir/bin/pip" install --upgrade pip setuptools wheel
"$env_dir/bin/pip" install --extra-index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1+cu126 torchvision==0.22.1+cu126
"$env_dir/bin/pip" install diffusers==0.35.2 accelerate==0.34.2 \
  transformers==4.57.3 safetensors einops numpy Pillow tqdm ftfy ninja scipy \
  decord imageio imageio-ffmpeg openai gradio==6.15.0
MAX_JOBS=12 "$env_dir/bin/pip" install --no-build-isolation flash-attn==2.8.3
"$env_dir/bin/pip" install --no-deps \
  git+https://github.com/ByteDance-Seed/VeOmni.git@v0.1.11
"$env_dir/bin/hf" download ByteDance/Bernini-R-Diffusers \
  --revision de8c4621d3ac75cc33efe3db8deaed2023e9ac8c \
  --local-dir "$model_dir"
touch "$runtime_root/.setup_complete"
