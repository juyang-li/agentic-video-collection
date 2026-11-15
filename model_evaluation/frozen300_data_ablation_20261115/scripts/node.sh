#!/bin/bash
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == testqueue && ${SLURM_JOB_ID:-} == 205167 ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/frozen300_data_ablation_20261115
source "$TASK/scripts/env.sh"
exec "$TASK/../a849_external_vbench_v2_96/runtime/vbench/bin/python" -u "$TASK/scripts/node.py"
