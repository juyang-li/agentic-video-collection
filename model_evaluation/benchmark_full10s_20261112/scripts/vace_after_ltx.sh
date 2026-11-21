#!/usr/bin/env bash
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112
OLD=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111
exec 8>"$TASK/logs/driver.lock"
while true; do
 flock 8
 if python - "$TASK" <<'PY'
from pathlib import Path
import json,sys
r=Path(sys.argv[1]);ids=[json.loads(l)['id'] for l in (r/'metadata/inputs.jsonl').read_text().splitlines()]
try:
 for model in ['a849','ltx_base']:
  for sid in ids:
   s=json.loads((r/'outputs'/model/(sid+'.json')).read_text());assert s['status']=='complete' and s['full_decode_frames']==121
except (OSError,AssertionError):sys.exit(1)
PY
 then break; fi
 flock -u 8
 sleep 60
done
# GPU inference and pending condition repair both wait until the fast baselines finish.
while ! "$OLD/runtime/vace/bin/python" - <<'PYGPU'
import torch
assert torch.cuda.device_count()==4
for i in range(4):
 assert 'H200' in torch.cuda.get_device_name(i)
 free,total=torch.cuda.mem_get_info(i);assert total-free<2*1024**3
PYGPU
 do sleep 900; done
IFS=',' read -r -a ids <<< "${CUDA_VISIBLE_DEVICES:?}"
if ! python - "$TASK" <<'PYMASK'
from pathlib import Path
import json,sys,hashlib
try:
 c=json.loads((Path(sys.argv[1])/'conditions/hair_color_10/condition.json').read_text())
 assert c['source_crop_repair']['method'].startswith('SOURCE-only per-frame CPU SAM2')
 assert len(c['mask_frames'])==len(c['mask_sha256'])==121
 for p,h in zip(c['mask_frames'],c['mask_sha256'],strict=True):
  assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
except (OSError,KeyError,AssertionError,ValueError):sys.exit(1)
PYMASK
then
 CUDA_VISIBLE_DEVICES="${ids[0]}" "$OLD/runtime/vace/bin/python" "$TASK/scripts/repair_head_conditions.py" --only hair_color_10 > "$TASK/logs/deferred_head_repair.log" 2>&1
fi
"$OLD/runtime/vace/bin/python" "$TASK/scripts/hair_color_last_overlay.py"
bash "$TASK/scripts/run_vace.sh"
