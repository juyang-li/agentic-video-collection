#!/usr/bin/env bash
#SBATCH --partition=testqueue
#SBATCH --account=prj0000000275
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=18:00:00
#SBATCH --job-name=frozen300_a849
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == testqueue && -n ${SLURM_JOB_ID:-} ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/frozen300_20261113
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
mkdir -p "$TASK/logs" "$TASK/outputs/a849"
exec 9>"$TASK/logs/a849.lock"
flock -n 9 || exit 73
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$NATIVE/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/frozen300_triton_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"
"$NATIVE/runtime/envs/trainer/bin/python" - <<'PY'
import torch
assert torch.cuda.device_count()==4
for i in range(4):
 assert 'H200' in torch.cuda.get_device_name(i)
 free,total=torch.cuda.mem_get_info(i);assert total-free<2*1024**3
PY
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:?}"
pids=()
for shard in 0 1 2 3; do
 (
 export CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}"
 bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/native_cache.py" --shard "$shard"
 bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/native_worker.py" --model a849 --stage all --shard "$shard" --shards 4
 ) > "$TASK/logs/a849_${SLURM_JOB_ID}_${shard}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
