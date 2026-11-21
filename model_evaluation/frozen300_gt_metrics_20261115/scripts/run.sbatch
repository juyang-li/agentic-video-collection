#!/bin/bash
#SBATCH --account=prj0000000275
#SBATCH --partition=testqueue
#SBATCH --job-name=frozen300_gt_metrics
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/frozen300_gt_metrics_20261115/logs/slurm_%j.log
set -euo pipefail
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/frozen300_gt_metrics_20261115
exec 9>"$TASK/logs/gpu.lock"
flock -n 9
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_HOME=/scratch/prj0000000275/jcy/.cache/torch
export LD_LIBRARY_PATH=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONUNBUFFERED=1
exec "$TASK/runtime/bin/python" "$TASK/scripts/evaluate.py"
