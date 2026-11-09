#!/usr/bin/env bash
#SBATCH --job-name=ltx25-bench60
#SBATCH --partition=testqueue,h200p
#SBATCH --account=prj0000000275
#SBATCH --qos=highpriority
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=1000G
#SBATCH --time=00:20:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
set -euo pipefail
ROOT="${LTX_BENCH_ROOT:?LTX_BENCH_ROOT must be provided}"
export LTX_BENCH_ROOT="$ROOT"
export OMP_NUM_THREADS=14
export OPENBLAS_NUM_THREADS=1
source "$ROOT/stage_job.sh"
trap 'echo "Slurm nearing time limit; completed samples already persisted"' USR1
srun --kill-on-bad-exit=0 --ntasks="${SLURM_NTASKS:-4}" --cpus-per-task=14 --gpus-per-task=1 --gpu-bind=single:1 \
  bash -c 'exec "$LTX_BENCH_ROOT/runtime/venv/bin/python" "$LTX_BENCH_ROOT/run_worker.py" --shard "$SLURM_PROCID"'
