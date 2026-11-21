#!/usr/bin/env bash
set -uo pipefail
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_distilled_v7_pilot10
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
exec 7>"$TASK/logs/controller.lock"
flock -n 7 || exit 68
bash "$TASK/scripts/run_gpu.sh" > "$TASK/logs/worker.log" 2>&1
rc=$?
bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/finalize.py" --exit-code "$rc" > "$TASK/logs/package.log" 2>&1
exit "$rc"
