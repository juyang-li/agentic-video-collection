#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue && "${SLURM_JOB_ID:-}" == 199052 ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$NATIVE/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$NATIVE/runtime/envs/trainer/bin/python" "$TASK/scripts/gate.py" --require smoke
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:?}"
[[ "${#gpu_ids[@]}" == 4 ]] || exit 65
"$NATIVE/runtime/envs/trainer/bin/python" - <<'PYGPU'
import torch
assert torch.cuda.device_count()==4
for i in range(4):
    free,total=torch.cuda.mem_get_info(i)
    assert total-free < 2*1024**3
PYGPU
pids=()
for shard in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}"
    for model in ltx_base a849; do
      bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/native_worker.py" --model "$model" --stage second --shard "$shard" > "$TASK/logs/second_${model}_$shard.log" 2>&1
    done
    "$TASK/runtime/vace/bin/python" "$TASK/scripts/prepare_conditions.py" --stage second --shard "$shard" > "$TASK/logs/second_conditions_final_$shard.log" 2>&1
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
