#!/usr/bin/env bash
#SBATCH --job-name=lance-filter-calib
#SBATCH --account=prj0000000275
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/prj0000000275/jcy/ljy/video_collection/lance_general_edit_finetune/2_vlm_filtering/logs/calibration-%j.log

set -euo pipefail

PROJECT=/scratch/prj0000000275/jcy/ljy/video_collection/lance_general_edit_finetune
STAGE1="$PROJECT/1_specialist_filtering"
STAGE2="$PROJECT/2_vlm_filtering"
RUN="$STAGE2/artifacts/vlm_run_v1"
PYTHON_BIN=/home/users/astar/cfar/stuchengyou/jcy/.conda/envs/xks_lance/bin/python

export HF_HOME=/scratch/prj0000000275/jcy/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME=/home/users/astar/cfar/stuchengyou/jcy/.cache/torch
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export YOLO_CONFIG_DIR="$STAGE2/artifacts/ultralytics_config"
export MPLCONFIGDIR="$STAGE2/artifacts/matplotlib"

mkdir -p "$RUN" "$STAGE2/logs"

"$PYTHON_BIN" "$STAGE1/scripts/run_stage1_2_robot.py" \
  --input "$RUN/calibration_robot_pool_120.csv" \
  --output "$RUN/smoke_robot_scores.csv" \
  --summary "$RUN/smoke_robot_summary.json" \
  --yolo-model "$STAGE2/artifacts/models/yolov8s-worldv2.pt" \
  --limit 10

"$PYTHON_BIN" "$STAGE2/scripts/run_qwen_judge.py" \
  --input "$RUN/calibration_pool_240.csv" \
  --output "$RUN/smoke_qwen_scores.csv" \
  --summary "$RUN/smoke_qwen_summary.json" \
  --limit 10 \
  --local-files-only

"$PYTHON_BIN" "$STAGE2/scripts/verify_model_scores.py" \
  --stage1-2 "$RUN/smoke_robot_scores.csv" \
  --qwen "$RUN/smoke_qwen_scores.csv"

"$PYTHON_BIN" "$STAGE1/scripts/run_stage1_2_robot.py" \
  --input "$RUN/calibration_robot_pool_120.csv" \
  --output "$RUN/calibration_robot_scores.csv" \
  --summary "$RUN/calibration_robot_score_summary.json" \
  --yolo-model "$STAGE2/artifacts/models/yolov8s-worldv2.pt" \
  --resume

"$PYTHON_BIN" "$STAGE2/scripts/run_qwen_judge.py" \
  --input "$RUN/historical_pair_labels.csv" \
  --output "$RUN/historical_qwen_scores.csv" \
  --summary "$RUN/historical_qwen_score_summary.json" \
  --resume \
  --local-files-only

"$PYTHON_BIN" "$STAGE2/scripts/run_qwen_judge.py" \
  --input "$RUN/calibration_pool_240.csv" \
  --output "$RUN/calibration_qwen_scores.csv" \
  --summary "$RUN/calibration_qwen_score_summary.json" \
  --resume \
  --local-files-only

"$PYTHON_BIN" "$STAGE2/scripts/evaluate_qwen_labels.py" --current-scope-only
"$PYTHON_BIN" "$STAGE2/scripts/build_calibration_review.py"
