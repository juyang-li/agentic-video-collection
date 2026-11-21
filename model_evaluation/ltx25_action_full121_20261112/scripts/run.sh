#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue && "${SLURM_JOB_ID:-}" == 199810 ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_action_full121_20261112
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$NATIVE/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/action_full121_triton"
mkdir -p "$TRITON_CACHE_DIR"
"$NATIVE/runtime/envs/trainer/bin/python" - <<'PY'
import torch
assert torch.cuda.device_count()==4
for i in range(4):
    assert 'H200' in torch.cuda.get_device_name(i)
    free,total=torch.cuda.mem_get_info(i)
    assert total-free<2*1024**3, f'GPU {i} busy'
PY
bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/prepare.py"
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:?}"
[[ ${#gpu_ids[@]} == 4 ]]
# Verify complete action_01 from both models before expanding.
for model in ltx_base a849; do
  CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/native_worker.py" --model "$model" --stage smoke --shard 0 > "$TASK/logs/smoke_${model}.log" 2>&1
  "$NATIVE/runtime/envs/trainer/bin/python" - "$TASK" "$model" <<'PY'
import json,sys
from pathlib import Path
s=json.loads((Path(sys.argv[1])/'outputs'/sys.argv[2]/'action_01.json').read_text())
assert s['status']=='complete' and s['full_decode_frames']==121,s
PY
done
for model in ltx_base a849; do
  pids=()
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}" bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/native_worker.py" --model "$model" --stage all --shard "$shard" > "$TASK/logs/all_${model}_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/package.py"
