#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v2_96
TRAINER=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/runtime/envs/trainer
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY PYTHONPATH HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HOME="$TASK/runtime/hf" TORCH_HOME="$TASK/runtime/torch" VBENCH_CACHE_DIR="$TASK/runtime/vbench_models"
"$TRAINER/bin/python" -m venv --system-site-packages "$TASK/runtime/vbench"
"$TASK/runtime/vbench/bin/python" -c 'import site,sys; from pathlib import Path; Path(site.getsitepackages()[0],"trainer.pth").write_text(sys.argv[1]+"\n")' "$TRAINER/lib/python3.12/site-packages"
"$TASK/runtime/vbench/bin/pip" install --no-deps pyiqa==0.1.14.1 openai-clip==1.0.1 decord==0.6.0 timm==1.0.12 easydict ftfy wcwidth addict lmdb yapf filterpy facexlib einops omegaconf scipy scikit-image tensorboard sentencepiece transformers==4.44.2 tokenizers==0.19.1 huggingface_hub==0.34.4 setuptools==80.9.0
# Import the unmodified pinned official source directly. setup.py hard-codes
# CUDA<=12.1 and requires a GPU even for installation; no metric code is changed.
"$TASK/runtime/vbench/bin/python" -c 'import site,sys; from pathlib import Path; Path(site.getsitepackages()[0],"vbench_source.pth").write_text(sys.argv[1]+"\n")' "$TASK/runtime/VBench"
"$TASK/runtime/vbench/bin/python" "$TASK/scripts/vbench_weights.py"
mkdir -p "$TORCH_HOME/hub/checkpoints"
ln -sf "$VBENCH_CACHE_DIR/dino_model/dino_vitbase16_pretrain.pth" "$TORCH_HOME/hub/checkpoints/dino_vitbase16_pretrain.pth"
"$TASK/runtime/vbench/bin/python" - <<'PY'
import json, subprocess, importlib
from pathlib import Path
from vbench.utils import init_submodules
root=Path(__import__('os').environ['VBENCH_CACHE_DIR']).parents[1]
dims=['subject_consistency','background_consistency','motion_smoothness','dynamic_degree','aesthetic_quality','imaging_quality']
for d in dims:
    importlib.import_module('vbench.'+d)
    init_submodules([d],local=True)
record={'revision':subprocess.check_output(['git','-C',str(root/'runtime/VBench'),'rev-parse','HEAD'],text=True).strip(),'dimensions':dims,'pip_freeze':subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True)}
(root/'metadata/vbench_install.json').write_text(json.dumps(record,indent=2))
PY
