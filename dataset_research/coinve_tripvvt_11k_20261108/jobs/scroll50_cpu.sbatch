#!/bin/bash
#SBATCH --job-name=cointrip-scroll50
#SBATCH --partition=testqueue
#SBATCH --qos=highpriority
#SBATCH --account=prj0000000275
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/prj0000000275/jcy/ljy/video_collection/dataset_research/coinve_tripvvt_11k_20261108/logs/scroll50-%j.log
set -euo pipefail
if [[ "${SLURM_JOB_PARTITION:-}" != "testqueue" ]]; then
  echo "ERROR: This project only permits partition testqueue." >&2
  exit 64
fi
cd /scratch/prj0000000275/jcy/ljy
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/apps/miniforge/24.11.3/bin/python -u video_collection/dataset_research/coinve_tripvvt_11k_20261108/scripts/build_scroll50.py all --skip-validation
