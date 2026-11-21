#!/usr/bin/env bash
#SBATCH --partition=testqueue
#SBATCH --account=prj0000000275
#SBATCH --job-name=a849-96-upload
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v2_96/logs/upload_results_%j.log
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v2_96
PY=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python
"$PY" /scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/scripts/upload_baidu_xpan.py \
  --local "$TASK/review/a849_external_vbench_v2_96_scroll.zip" \
  --remote /apps/bypy/lance_review/a849_external_vbench_v2_96_scroll.zip \
  --state "$TASK/metadata/upload_results_state.json" \
  --bypy-config /scratch/prj0000000275/jcy/ljy/video_collection/dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json
