#!/bin/bash
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == testqueue && -n ${SLURM_JOB_ID:-} ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/frozen300_vbench_20261115
source "$TASK/scripts/env.sh"
phase=${VB_PHASE_TAG:-full}
[[ $phase =~ ^[a-zA-Z0-9_-]+$ ]] || exit 64
exec 9>"$TASK/logs/gpu.lock"
flock -n 9
# Cooperate with existing VBench jobs on the same user's allocation.
exec 7>"$TASK/../a849_tripvvt_pilot32_20261115/runtime/job202896.lock"
flock -n 7
"$OLD/runtime/vbench/bin/python" - <<'PY'
import torch
assert torch.cuda.device_count()==4
for i in range(4):
 assert 'H200' in torch.cuda.get_device_name(i)
 free,total=torch.cuda.mem_get_info(i);assert total-free<2*1024**3,'GPU busy'
PY
IFS=',' read -r -a ids <<< "${CUDA_VISIBLE_DEVICES:?}"
pids=()
for shard in 0 1 2 3;do
 (
 export CUDA_VISIBLE_DEVICES="${ids[$shard]}" VB_SMOKE=1
 "$OLD/runtime/vbench/bin/python" "$TASK/scripts/score.py" quality --shard "$shard"
 "$OLD/runtime/vbench/bin/python" "$TASK/scripts/score.py" static --shard "$shard"
 ) > "$TASK/logs/smoke_${SLURM_JOB_ID}_${phase}_${shard}.log" 2>&1 &
 pids+=("$!")
done
for pid in "${pids[@]}";do wait "$pid";done
"$OLD/runtime/vbench/bin/python" "$TASK/scripts/check_smoke.py"
if [[ ${VB_SMOKE_ONLY:-0} == 1 ]];then exit 0;fi
pids=()
for shard in 0 1 2 3; do
 (
 export CUDA_VISIBLE_DEVICES="${ids[$shard]}"
 uuid=$(nvidia-smi --id="${CUDA_VISIBLE_DEVICES}" --query-gpu=uuid --format=csv,noheader)
 exec 8>"/tmp/frozen300_vbench_${uuid}.lock"
 flock -n 8
 "$OLD/runtime/vbench/bin/python" "$TASK/scripts/score.py" quality --shard "$shard"
 "$OLD/runtime/vbench/bin/python" "$TASK/scripts/score.py" static --shard "$shard"
 ) > "$TASK/logs/gpu_${SLURM_JOB_ID}_${phase}_${shard}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}";do wait "$pid" || failed=1;done
exit "$failed"
