#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue && "${SLURM_JOB_ID:-}" == 199052 ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111
IFS=',' read -r -a allocated_gpu_ids <<< "${CUDA_VISIBLE_DEVICES:?}"
export CUDA_VISIBLE_DEVICES="${allocated_gpu_ids[0]}" PYTHONNOUSERSITE=1
"$TASK/runtime/vace/bin/python" - <<'PYGPU'
import torch
assert torch.cuda.device_count()==1
free,total=torch.cuda.mem_get_info(0)
assert total-free < 2*1024**3, 'Allocated repair GPU is already busy; aborting without loading models'
PYGPU
stage="${A100_CONDITION_STAGE:-smoke}"
args=()
for sid in "$@"; do args+=(--only "$sid"); done
if [[ "${#args[@]}" == 0 ]]; then args=(--only identity_01); fi
exec "$TASK/runtime/vace/bin/python" "$TASK/scripts/prepare_conditions.py" --stage "$stage" --shard 0 "${args[@]}"
