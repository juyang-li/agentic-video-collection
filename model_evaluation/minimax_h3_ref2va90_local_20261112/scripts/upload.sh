#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue && "${SLURM_JOB_ID:-}" == 199810 ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112
NAME=minimax_h3_ref2va90_local_scroll_20261112
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY PYTHONPATH
"$TASK/runtime/env/bin/python" "$TASK/scripts/upload_baidu_xpan.py" \
 --local "$TASK/review/$NAME.zip" --remote "/apps/bypy/lance_review/$NAME.zip" \
 --state "$TASK/review/$NAME.baidu_upload.json" \
 --bypy-config /scratch/prj0000000275/jcy/ljy/video_collection/dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json

"$TASK/runtime/env/bin/python" "$TASK/scripts/finalize.py"
