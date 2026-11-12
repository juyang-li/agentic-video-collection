#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue && "${SLURM_JOB_ID:-}" == 199052 ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
stage="${1:?prepare|cache|smoke|controls|second|all}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$NATIVE/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/a100_triton"
mkdir -p "$TASK/logs" "$TRITON_CACHE_DIR"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
"$NATIVE/runtime/envs/trainer/bin/python" - <<'PYGPU'
import torch
assert torch.cuda.device_count()==4, 'Require exactly four allocated GPUs'
assert all('H200' in torch.cuda.get_device_name(i) for i in range(4)), 'Expected H200 allocation'
for i in range(4):
    free,total=torch.cuda.mem_get_info(i)
    assert total-free < 2*1024**3, f'Allocated GPU {i} already busy; will not overlap another workload'
PYGPU
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:?Slurm must expose allocated devices}"
[[ "${#gpu_ids[@]}" == 4 ]] || exit 65
run_cache() {
  pids=()
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}" bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/native_cache.py" --shard "$shard" > "$TASK/logs/cache_$shard.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}
if [[ "$stage" == cache ]]; then run_cache; exit 0; fi
if [[ "$stage" == prepare ]]; then
  "$NATIVE/runtime/envs/trainer/bin/python" "$TASK/scripts/sample.py"
  "$NATIVE/runtime/envs/trainer/bin/python" "$TASK/scripts/native_cache.py" --index
  bash "$TASK/scripts/package_upload.sh"
  bash "$TASK/scripts/setup_vace.sh" > "$TASK/logs/vace_setup.log" 2>&1 &
  setup_pid=$!
  run_cache
  wait "$setup_pid"
  exit 0
fi
if [[ "$stage" == second || "$stage" == all ]]; then
  previous=smoke
  [[ "$stage" == all ]] && previous=second
  "$NATIVE/runtime/envs/trainer/bin/python" "$TASK/scripts/gate.py" --require "$previous"
fi
worker_stage="$stage"
extra=()
if [[ "$stage" == controls ]]; then worker_stage=smoke;extra=(--control);fi
for model in ltx_base a849 vace; do
  if [[ "${A100_NATIVE_ONLY:-0}" == 1 && "$model" == vace ]]; then continue; fi
  if [[ "${A100_VACE_ONLY:-0}" == 1 && "$model" != vace ]]; then continue; fi
  if [[ "$model" == vace && "$stage" != controls ]]; then
    pids=()
    for shard in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}" "$TASK/runtime/vace/bin/python" "$TASK/scripts/prepare_conditions.py" --stage "$worker_stage" --shard "$shard" > "$TASK/logs/${stage}_conditions_$shard.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
  fi
  pids=()
  for shard in 0 1 2 3; do
    if [[ "$model" == vace ]]; then
      CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}" "$TASK/runtime/vace/bin/python" "$TASK/scripts/vace_worker.py" --stage "$worker_stage" --shard "$shard" "${extra[@]}" > "$TASK/logs/${stage}_${model}_$shard.log" 2>&1 &
    else
      CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}" bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/native_worker.py" --model "$model" --stage "$worker_stage" --shard "$shard" "${extra[@]}" > "$TASK/logs/${stage}_${model}_$shard.log" 2>&1 &
    fi
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
