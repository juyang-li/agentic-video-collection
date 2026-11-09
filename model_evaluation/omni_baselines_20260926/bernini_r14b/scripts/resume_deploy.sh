#!/usr/bin/env bash
set -euo pipefail

runtime_root=/tmp/omni_bernini_20260926
env_dir="$runtime_root/env"
model_dir="$runtime_root/Bernini-R-Diffusers"

export PIP_CACHE_DIR="$runtime_root/pip-cache"
export HF_HOME="$runtime_root/hf-cache"
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT=1800
export CUDA_HOME=/apps/cuda/12.6.2
export PATH="$CUDA_HOME/bin:$PATH"

"$env_dir/bin/pip" install --no-deps \
  git+https://github.com/ByteDance-Seed/VeOmni.git@v0.1.11
"$env_dir/bin/hf" download ByteDance/Bernini-R-Diffusers \
  --revision de8c4621d3ac75cc33efe3db8deaed2023e9ac8c \
  --local-dir "$model_dir"
"$env_dir/bin/python" - <<'PY'
import flash_attn, torch, transformers, diffusers, veomni
print("READY", torch.__version__, transformers.__version__, diffusers.__version__)
PY
