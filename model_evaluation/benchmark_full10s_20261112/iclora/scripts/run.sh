#!/usr/bin/env bash
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/iclora
[[ ! -f "$TASK/metadata/protocol_rejection.json" ]] || { echo "Rejected IC-LoRA input protocol; choose a replacement experiment before resuming." >&2; exit 65; }
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
exec 9>"$TASK/logs/driver.lock"
flock -n 9 || exit 73
exec 7>"$TASK/../logs/allocation_${SLURM_JOB_ID}.lock"
flock 7
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$NATIVE/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/full10s_iclora_triton"
mkdir -p "$TRITON_CACHE_DIR"
PY="$NATIVE/runtime/envs/trainer/bin/python"
"$PY" - "$TASK" <<'PY'
import torch,json,sys
from pathlib import Path
r=Path(sys.argv[1])
for name in ['preflight.json','pipeline_preflight.json','download_complete.json']:assert json.loads((r/'metadata'/name).read_text())['status']=='passed'
assert torch.cuda.device_count()==4
for i in range(4):
 assert 'H200' in torch.cuda.get_device_name(i)
 free,total=torch.cuda.mem_get_info(i);assert total-free<2*1024**3,f'GPU {i} busy'
PY
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:?}"
CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/worker.py" --stage pilot > "$TASK/logs/pilot.log" 2>&1
"$PY" - "$TASK" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1])
for sid in ['identity_01','hair_color_01','action_01']:
 s=json.loads((r/'outputs'/(sid+'.json')).read_text());assert s['status']=='complete' and s['full_decode_frames']==241 and s['source_and_motion_full10s'] and s['no_gt']
(r/'metadata/pilot_gate.json').write_text(json.dumps({'status':'passed','scope':'frozen source/reference protocol, actual official pipeline invocation and full decode only; not human quality'}))
PY
for stage in first all; do
 pids=()
 for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}" bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/worker.py" --stage "$stage" --shard "$shard" --shards 4 > "$TASK/logs/${stage}_${shard}.log" 2>&1 &
  pids+=("$!")
 done
 failed=0
 for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
 [[ $failed == 0 ]] || exit 1
done
