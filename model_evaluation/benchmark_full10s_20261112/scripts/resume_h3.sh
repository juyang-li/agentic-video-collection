#!/usr/bin/env bash
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112
exec 8>"$TASK/logs/driver.lock"
while true; do
 flock 8
 if python - "$TASK" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);items=[json.loads(l) for l in (r/'metadata/inputs.jsonl').read_text().splitlines()]
try:
 for model in ('ltx_base','a849'):
  for item in items:
   state=json.loads((r/'outputs'/model/(item['id']+'.json')).read_text())
   assert state['status']=='complete' and state['full_decode_frames']==121
except (OSError,AssertionError):sys.exit(1)
PY
 then break; fi
 flock -u 8
 sleep 60
done
while [[ ! -f "$TASK/h3/metadata/preflight.json" ]]; do sleep 60; done
python - "$TASK/h3" <<'PYCODE'
import json,os,sys,time
from pathlib import Path
r=Path(sys.argv[1]);archive=r/'metadata/interrupted_attempts'
for p in (r/'outputs').glob('*.json'):
 state=json.loads(p.read_text())
 if state['status']=='running' and state.get('slurm_job_id')!=os.environ['SLURM_JOB_ID']:
  archive.mkdir(exist_ok=True);state['interruption_reason']='previous allocation ended; request belongs to terminated resident server'
  (archive/(p.stem+'_'+str(time.time_ns())+'.json')).write_text(json.dumps(state,ensure_ascii=False,indent=2))
  p.unlink()
PYCODE
bash "$TASK/h3/scripts/run.sh"
