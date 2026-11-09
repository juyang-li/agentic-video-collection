#!/usr/bin/env bash
set -euo pipefail

PROJECT=/scratch/prj0000000275/jcy/ljy/video_collection/lance_general_edit_finetune
STAGE1="$PROJECT/1_specialist_filtering"
STAGE2="$PROJECT/2_vlm_filtering"
RUN="$STAGE2/artifacts/vlm_run_v1"
PYTHON_BIN=/home/users/astar/cfar/stuchengyou/jcy/.conda/envs/xks_lance/bin/python
BUNDLE="$RUN/calibration_review_100_bundle"
ARCHIVE="$RUN/general_edit_stage1_2_stage2_calibration_100_v1_bundle.tar"

"$PYTHON_BIN" "$STAGE1/scripts/build_review_bundle.py" \
  --manifest "$RUN/calibration_review_100.csv" \
  --output "$BUNDLE" \
  --expected-size 100

tar -cf "$ARCHIVE" -C "$RUN" "$(basename "$BUNDLE")"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"

"$PYTHON_BIN" "$STAGE1/scripts/upload_baidu_xpan.py" \
  --local "$ARCHIVE" \
  --remote /apps/bypy/lance_review/general_edit_stage1_2_stage2_calibration_100_v1_bundle.tar \
  --state "$RUN/.baidu_xpan_calibration_100_upload.json"

