#!/usr/bin/env bash
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == testqueue && -n ${SLURM_JOB_ID:-} ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_distilled_v7_pilot10
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
exec 9>"$TASK/logs/worker.lock"
flock -n 9 || exit 65
# Slurm grants one exclusive GPU to this step; also cooperate with node-local UUID locks.
uuid=$(nvidia-smi --id="${CUDA_VISIBLE_DEVICES:?}" --query-gpu=uuid --format=csv,noheader)
[[ $uuid == GPU-* && $uuid != *$'\n'* ]] || exit 66
exec 8>"/tmp/ltx25_pilot_${uuid}.lock"
flock -n 8 || exit 67
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/worker.py"
