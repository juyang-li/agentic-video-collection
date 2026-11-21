#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_96
bash "$TASK/scripts/run.sh" generate
bash "$TASK/scripts/run.sh" score
bash "$TASK/scripts/run.sh" static

# User requested keeping this allocation after evaluation while discussing results.
bash "$TASK/scripts/hold_allocation.sh"
