#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue && "${SLURM_JOB_ID:-}" == 199052 ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:?}"
[[ "${#gpu_ids[@]}" == 4 ]] || exit 65
"$TASK/runtime/vace/bin/python" - <<'PYGPU'
import torch
assert torch.cuda.device_count()==4
for i in range(4):
    free,total=torch.cuda.mem_get_info(i)
    assert total-free < 2*1024**3
PYGPU
pids=()
for i in 0 1 2 3; do
  shard=$((i%2))
  extra=()
  [[ "$i" -ge 2 ]] && extra=(--control)
  CUDA_VISIBLE_DEVICES="${gpu_ids[$i]}" "$TASK/runtime/vace/bin/python" "$TASK/scripts/vace_worker.py" --stage smoke --shard "$shard" "${extra[@]}" > "$TASK/logs/gray_repaint_${i}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
