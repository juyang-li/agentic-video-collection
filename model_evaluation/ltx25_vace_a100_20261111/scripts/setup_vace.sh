#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue && "${SLURM_JOB_ID:-}" == 199052 ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
export PYTHONNOUSERSITE=1
unset PYTHONPATH HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
mkdir -p "$TASK/runtime"
if [[ ! -x "$TASK/runtime/vace/bin/python" ]]; then
  "$NATIVE/runtime/envs/trainer/bin/python" -m venv --without-pip "$TASK/runtime/vace"
  "$TASK/runtime/vace/bin/python" - <<'PY'
import sysconfig
from pathlib import Path
site=Path(sysconfig.get_paths()['purelib'])
# Read the existing verified torch build; install VACE dependencies separately.
(site/'native_torch.pth').write_text("import site; site.addsitedir('/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/runtime/envs/trainer/lib/python3.12/site-packages')\n")
PY
fi
if [[ ! -f "$TASK/runtime/vace_ready.json" ]]; then
  /apps/miniforge/24.11.3/bin/python -m pip --python "$TASK/runtime/vace/bin/python" install \
    'diffusers==0.35.2' 'transformers==4.57.1' 'huggingface_hub==0.36.0' 'accelerate==1.11.0' \
    'numpy==2.2.6' 'pillow==11.3.0' 'opencv-python==4.12.0.88' \
    'hydra-core==1.3.2' 'iopath==0.1.10' 'onnxruntime-gpu==1.23.2' \
    av imageio imageio-ffmpeg pycocotools ftfy sentencepiece scipy 'matplotlib==3.10.7'
  "$TASK/runtime/vace/bin/python" "$TASK/scripts/setup_vace.py"
fi
