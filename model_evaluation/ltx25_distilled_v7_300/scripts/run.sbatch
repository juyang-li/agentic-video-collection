#!/usr/bin/env bash
#SBATCH --account=prj0000000275
#SBATCH --partition=testqueue
#SBATCH --job-name=ltx_distilled_v7_300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_distilled_v7_300/logs/slurm_%j.log
set -euo pipefail
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_distilled_v7_300
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
[[ ${SLURM_JOB_PARTITION:-} == testqueue ]] || exit 64
mkdir -p "$TASK/outputs"
export PYTHONDONTWRITEBYTECODE=1
exec bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/driver.py"
