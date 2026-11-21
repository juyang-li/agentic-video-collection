#!/usr/bin/env bash
#SBATCH --partition=testqueue
#SBATCH --account=prj0000000275
#SBATCH --job-name=a849_external
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:4
#SBATCH --mem=1000G
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v1/logs/slurm_%j.log
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
bash /scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v1/scripts/run.sh "${1:?cache|conditions|smoke|all}"
