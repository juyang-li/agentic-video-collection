#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue && "${SLURM_JOB_ID:-}" == 199052 ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$NATIVE/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:?}"
[[ "${#gpu_ids[@]}" == 4 ]] || exit 65
# All four allocated cards have work; no overlap with the preceding stage.
"$NATIVE/runtime/envs/trainer/bin/python" - <<'PYGPU'
import torch
assert torch.cuda.device_count()==4
for i in range(4):
    free,total=torch.cuda.mem_get_info(i)
    assert total-free < 2*1024**3, 'Prior stage still owns GPU memory'
PYGPU
pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}" "$TASK/runtime/vace/bin/python" "$TASK/scripts/prepare_conditions.py" --stage smoke --shard "$shard" > "$TASK/logs/final_smoke_condition_$shard.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
# Original whole-outfit attempt had an incomplete shoe/sleeve mask and is archived.
pids=()
# Identity/action repairs followed by the four native fixed-seed controls.
for pair in '0 0 0' '2 2 1'; do
  read -r gpu_index formal_shard control_shard <<< "$pair"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_ids[$gpu_index]}"
    "$TASK/runtime/vace/bin/python" "$TASK/scripts/vace_worker.py" --stage smoke --shard "$formal_shard" > "$TASK/logs/repaired_vace_$formal_shard.log" 2>&1
    for model in ltx_base a849; do
      bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/native_worker.py" --model "$model" --stage smoke --shard "$control_shard" --control > "$TASK/logs/control_${model}_$control_shard.log" 2>&1
    done
  ) &
  pids+=("$!")
done
(
  export CUDA_VISIBLE_DEVICES="${gpu_ids[1]}"
  "$TASK/runtime/vace/bin/python" "$TASK/scripts/vace_worker.py" --stage smoke --shard 1 > "$TASK/logs/repaired_vace_outfit.log" 2>&1
  "$TASK/runtime/vace/bin/python" "$TASK/scripts/vace_worker.py" --stage smoke --shard 1 --control > "$TASK/logs/control_vace_outfit.log" 2>&1
) &
pids+=("$!")
(
  export CUDA_VISIBLE_DEVICES="${gpu_ids[3]}"
  "$TASK/runtime/vace/bin/python" "$TASK/scripts/vace_worker.py" --stage smoke --shard 0 --control > "$TASK/logs/control_vace_identity.log" 2>&1
  # Prepare the next representative inputs; formal inference still requires its gate.
  for shard in 0 1 2 3; do
    "$TASK/runtime/vace/bin/python" "$TASK/scripts/prepare_conditions.py" --stage second --shard "$shard" > "$TASK/logs/second_condition_early_$shard.log" 2>&1
  done
) &
pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
