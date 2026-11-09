#!/usr/bin/env bash
#SBATCH --job-name=lance-rect-full
#SBATCH --account=prj0000000275
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/prj0000000275/jcy/ljy/video_collection/lance_general_edit_finetune/2_vlm_filtering/logs/rect-full-%j.log

set -euo pipefail

PROJECT=/scratch/prj0000000275/jcy/ljy/video_collection/lance_general_edit_finetune
STAGE1="$PROJECT/1_specialist_filtering"
STAGE2="$PROJECT/2_vlm_filtering"
RUN="$STAGE2/artifacts/vlm_run_v4"
PYTHON_BIN=/home/users/astar/cfar/stuchengyou/jcy/.conda/envs/xks_lance/bin/python

export HF_HOME=/scratch/prj0000000275/jcy/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME=/home/users/astar/cfar/stuchengyou/jcy/.cache/torch
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$RUN" "$STAGE2/logs"

"$PYTHON_BIN" "$STAGE2/scripts/run_qwen_rect_judge.py" \
  --input "$STAGE1/artifacts/filter_run_v2/stage1_next_input.csv" \
  --output "$RUN/qwen_rect_pair_scores.csv" \
  --summary "$RUN/qwen_rect_pair_score_summary.json" \
  --resume \
  --local-files-only
