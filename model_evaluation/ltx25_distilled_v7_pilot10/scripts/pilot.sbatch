#!/usr/bin/env bash
#SBATCH --account=prj0000000275
#SBATCH --partition=testqueue
#SBATCH --job-name=ltx_distilled_v7_pilot10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_distilled_v7_pilot10/logs/slurm_%j.log
set -euo pipefail
exec bash /scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_distilled_v7_pilot10/scripts/controller.sh
