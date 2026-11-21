#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue && "${SLURM_JOB_ID:-}" == 199810 ]] || exit 64
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/recover_history.py"
