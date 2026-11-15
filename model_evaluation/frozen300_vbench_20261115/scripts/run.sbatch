#!/bin/bash
#SBATCH --account=prj0000000275
#SBATCH --partition=testqueue
#SBATCH --job-name=frozen300_vbench
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=56
#SBATCH --mem=384G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/frozen300_vbench_20261115/logs/slurm_%j.log
exec bash /scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/frozen300_vbench_20261115/scripts/gpu.sh
