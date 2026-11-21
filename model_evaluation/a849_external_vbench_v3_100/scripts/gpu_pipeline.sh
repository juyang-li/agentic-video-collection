#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_100
while [[ ! -f "$TASK/metadata/pipeline_ready.json" ]];do sleep 30;done
flock "$TASK/runtime/gpu_generation.lock" bash "$TASK/scripts/generate100_locked.sh"
