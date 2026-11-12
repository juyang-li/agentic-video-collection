#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue && "${SLURM_JOB_ID:-}" == 199052 ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/package.py" --kind results
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
bash "$NATIVE/scripts/common/run_trainer_python.sh" /scratch/prj0000000275/jcy/ljy/video_collection/lance_general_edit_finetune/1_specialist_filtering/scripts/upload_baidu_xpan.py \
  --local "$TASK/review/a100_results_scroll_20261111.zip" \
  --remote /apps/bypy/lance_review/a100_results_scroll_20261111.zip \
  --state "$TASK/review/a100_results_scroll_20261111.baidu_upload.json" \
  --bypy-config /scratch/prj0000000275/jcy/ljy/video_collection/dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json
