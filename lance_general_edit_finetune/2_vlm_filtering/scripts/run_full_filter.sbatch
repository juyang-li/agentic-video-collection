#!/usr/bin/env bash
#SBATCH --job-name=lance-filter-full
#SBATCH --account=prj0000000275
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/prj0000000275/jcy/ljy/video_collection/lance_general_edit_finetune/2_vlm_filtering/logs/full-%j.log

set -euo pipefail

PROJECT=/scratch/prj0000000275/jcy/ljy/video_collection/lance_general_edit_finetune
STAGE1="$PROJECT/1_specialist_filtering"
STAGE2="$PROJECT/2_vlm_filtering"
STAGE1_RUN="$STAGE1/artifacts/filter_run_v2"
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

"$PYTHON_BIN" "$STAGE1/scripts/run_stage1_2_robot.py" \
  --input "$STAGE1_RUN/stage1_2_robot_queue.csv" \
  --output "$STAGE1_RUN/stage1_2_robot_scores.csv" \
  --summary "$STAGE1_RUN/stage1_2_robot_score_summary.json" \
  --yolo-model "$STAGE2/artifacts/models/yolov8s-worldv2.pt" \
  --resume

"$PYTHON_BIN" "$STAGE1/scripts/calibrate_stage1_2.py" \
  --review "$RUN/general_edit_stage1_2_stage2_calibration_100_v1_review.csv"

"$PYTHON_BIN" "$STAGE2/scripts/build_stage2_input.py"

"$PYTHON_BIN" "$STAGE2/scripts/run_qwen_judge.py" \
  --input "$RUN/stage2_input.csv" \
  --output "$RUN/qwen_pair_scores.csv" \
  --summary "$RUN/qwen_pair_score_summary.json" \
  --resume \
  --local-files-only

"$PYTHON_BIN" "$STAGE2/scripts/calibrate_qwen_judge.py" \
  --scores "$RUN/historical_qwen_scores.csv" \
  --scores "$RUN/calibration_qwen_scores.csv" \
  --scores "$RUN/qwen_pair_scores.csv" \
  --labels "$RUN/historical_pair_labels.csv" \
  --prospective-review "$RUN/general_edit_stage1_2_stage2_calibration_100_v1_review.csv"

"$PYTHON_BIN" "$STAGE2/scripts/apply_qwen_thresholds.py"
"$PYTHON_BIN" "$STAGE2/scripts/finalize_stage2.py"
