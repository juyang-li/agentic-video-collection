#!/bin/bash
set -euo pipefail
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/frozen300_data_ablation_20261115
source "$TASK/scripts/env.sh"
exec "$TASK/../frozen300_gt_metrics_20261115/runtime/bin/python" -u "$TASK/scripts/supervisor.py"
