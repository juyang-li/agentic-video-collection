#!/usr/bin/env bash
#SBATCH --job-name=lance-rect-smoke
#SBATCH --account=prj0000000275
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/prj0000000275/jcy/ljy/video_collection/lance_general_edit_finetune/2_vlm_filtering/logs/rect-smoke-%j.log

set -euo pipefail

PROJECT=/scratch/prj0000000275/jcy/ljy/video_collection/lance_general_edit_finetune
STAGE2="$PROJECT/2_vlm_filtering"
OLD_RUN="$STAGE2/artifacts/vlm_run_v1"
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
  --input "$RUN/current_scope_smoke_36.csv" \
  --output "$RUN/current_scope_smoke_36_scores.csv" \
  --summary "$RUN/current_scope_smoke_36_score_summary.json" \
  --local-files-only

"$PYTHON_BIN" "$STAGE2/scripts/summarize_qwen_rect.py" \
  --scores "$RUN/current_scope_smoke_36_scores.csv" \
  --labels "$OLD_RUN/historical_pair_labels.csv" \
  --baseline-scores "$OLD_RUN/historical_qwen_scores.csv" \
  --current-scope-only \
  --output "$RUN/current_scope_smoke_36_evaluation.json"
