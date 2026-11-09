#!/bin/bash
#SBATCH --job-name=vefx-4b-smoke
#SBATCH --account=prj0000000275
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/prj0000000275/jcy/ljy/video_collection/lance_general_edit_finetune/2_vlm_filtering/logs/vefx-4b-smoke-%j.log

set -euo pipefail

PROJECT=/scratch/prj0000000275/jcy/ljy/video_collection/lance_general_edit_finetune
PYTHON="$PROJECT/2_vlm_filtering/.venv-vefx/bin/python"
export HF_HOME=/scratch/prj0000000275/jcy/.cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1

cd "$PROJECT"
"$PYTHON" 2_vlm_filtering/scripts/run_vefx_reward.py \
  --input 2_vlm_filtering/artifacts/model_bakeoff_v1/current_scope_63.csv \
  --output 2_vlm_filtering/artifacts/model_bakeoff_v1/vefx_4b_smoke_scores.csv \
  --summary 2_vlm_filtering/artifacts/model_bakeoff_v1/vefx_4b_smoke_summary.json \
  --model xiangbog/VEFX-Reward-4B \
  --sample-id pilot-goku_add-31d1e44c2b2846e7 \
  --sample-id pilot-goku_add-ab74b2cb060e74e4 \
  --overwrite
