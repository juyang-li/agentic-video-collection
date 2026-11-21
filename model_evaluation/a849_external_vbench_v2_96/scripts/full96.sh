#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v2_96
PY=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python
bash "$TASK/scripts/run.sh" all
bash "$TASK/scripts/score.sh" all
"$PY" "$TASK/scripts/aggregate.py" --stage all
"$PY" "$TASK/scripts/review.py" --stage all
