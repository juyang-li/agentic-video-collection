#!/usr/bin/env bash
set -euo pipefail
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112
# Archive only prior terminated-server in-flight IDs, retain completed outputs.
python "$TASK/h3/scripts/archive_interrupted.py"
exec bash "$TASK/h3/scripts/run.sh"
