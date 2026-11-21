#!/usr/bin/env bash
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112
# Wait in the kernel for the prioritized LTX/A849 driver to release all GPUs.
exec 8>"$TASK/logs/driver.lock"
flock 8
python - "$TASK" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);items=[json.loads(l) for l in (r/'metadata/inputs.jsonl').read_text().splitlines()]
for model in ('ltx_base','a849'):
 for item in items:
  state=json.loads((r/'outputs'/model/(item['id']+'.json')).read_text())
  assert state['status']=='complete' and state['full_decode_frames']==121
PY
while [[ ! -f "$TASK/h3/metadata/preflight.json" ]]; do sleep 60; done
bash "$TASK/h3/scripts/run.sh"
