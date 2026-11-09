#!/usr/bin/env bash
#SBATCH --job-name=ivebench-lance-resume
#SBATCH --account=prj0000000275
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=56
#SBATCH --mem=1000G
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ivebench_lance/artifacts/resume-%j.log

set -euo pipefail

ROOT=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ivebench_lance
PYTHON=/home/users/astar/cfar/stuchengyou/jcy/.conda/envs/xks_lance/bin/python

cd /scratch/prj0000000275/jcy/ljy
exec "$PYTHON" "$ROOT/scripts/queue_remaining_stages.py"
