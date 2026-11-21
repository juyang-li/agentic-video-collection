#!/usr/bin/env bash
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == testqueue && -n ${SLURM_JOB_ID:-} ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112
OLD=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111
exec 9>"$TASK/logs/vace_driver.lock"
flock -n 9 || exit 73
exec 7>"$TASK/logs/allocation_${SLURM_JOB_ID}.lock"
flock 7
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/full10s_vace_triton"
mkdir -p "$TRITON_CACHE_DIR"
PY="$OLD/runtime/vace/bin/python"
"$PY" - <<'PY'
import torch
assert torch.cuda.device_count()==4
for i in range(4):
 assert 'H200' in torch.cuda.get_device_name(i)
 free,total=torch.cuda.mem_get_info(i);assert total-free<2*1024**3,f'GPU {i} busy'
PY
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:?}"
for stage in smoke all; do
 if [[ "$stage" == smoke ]] && "$PY" - "$TASK" <<'PYGATE'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1])
try:
 for sid in ['identity_01','action_01']:
  p=r/'outputs/vace'/(sid+'.json');s=json.loads(p.read_text());contract=s['contract']
  assert s['status']=='complete' and s['full_decode_frames']==121 and contract['fps']==12
  for path,key in [(r/'scripts/vace_worker.py','worker_sha256'),(r/'metadata/inputs.jsonl','input_manifest_sha256'),(r/'conditions'/sid/'condition.json','condition_sha256')]:
   assert hashlib.sha256(path.read_bytes()).hexdigest()==contract[key]
  assert hashlib.sha256((r/'outputs/vace'/(sid+'.mp4')).read_bytes()).hexdigest()==s['output_sha256']
except (OSError,AssertionError,KeyError):sys.exit(1)
print('Reusing both verified complete 10s VACE smoke outputs')
PYGATE
 then continue; fi
 pids=()
 for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}" "$PY" "$TASK/scripts/prepare_conditions.py" --stage "$stage" --shard "$shard" > "$TASK/logs/${stage}_vace_conditions_${shard}.log" 2>&1 &
  pids+=("$!")
 done
 failed=0
 for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
 [[ $failed == 0 ]] || exit 1
 pids=()
 for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}" "$PY" "$TASK/scripts/vace_worker.py" --stage "$stage" --shard "$shard" > "$TASK/logs/${stage}_vace_${shard}.log" 2>&1 &
  pids+=("$!")
 done
 failed=0
 for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
 [[ $failed == 0 ]] || exit 1
done
