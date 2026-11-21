#!/bin/bash
set -euo pipefail
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/frozen300_gt_metrics_20261115
PY=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/env/bin/python
"$PY" -m venv --system-site-packages "$TASK/runtime"
"$TASK/runtime/bin/python" -m pip install --disable-pip-version-check lpips==0.1.4 scikit-image==0.26.0
