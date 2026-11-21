#!/usr/bin/env bash
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == testqueue && -n ${SLURM_JOB_ID:-} ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
exec 9>"$TASK/logs/driver.lock"
flock -n 9 || exit 73
exec 7>"$TASK/logs/allocation_${SLURM_JOB_ID}.lock"
flock 7
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$NATIVE/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/benchmark_full10s_triton"
mkdir -p "$TRITON_CACHE_DIR"
"$NATIVE/runtime/envs/trainer/bin/python" - <<'PY'
import torch
assert torch.cuda.device_count()==4
for i in range(4):
 assert 'H200' in torch.cuda.get_device_name(i)
 free,total=torch.cuda.mem_get_info(i);assert total-free<2*1024**3,f'GPU {i} busy'
PY
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:?}"
pids=()
for shard in 0 1 2 3; do
 CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}" bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/native_worker.py" --model a849 --stage all --shard "$shard" --shards 4 > "$TASK/logs/a849_four_gpu_${shard}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
