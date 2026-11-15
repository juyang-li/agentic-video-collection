#!/bin/bash
set -euo pipefail
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/frozen300_vbench_20261115
source "$TASK/scripts/env.sh"
[[ -z ${SLURM_JOB_ID:-} ]] || exit 64
exec 9>"$TASK/logs/prepare.lock"
flock -n 9
"$OLD/runtime/vbench/bin/python" "$TASK/scripts/prepare.py"
exec "$OLD/runtime/vbench/bin/python" "$TASK/scripts/score.py" raw
