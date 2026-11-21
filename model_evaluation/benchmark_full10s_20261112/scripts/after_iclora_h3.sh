#!/usr/bin/env bash
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112
exec 8>"$TASK/iclora/logs/driver.lock"
flock 8
python - "$TASK" <<'PY'
from pathlib import Path
import json,sys
r=Path(sys.argv[1]);ids=[json.loads(l)['id'] for l in (r/'metadata/inputs.jsonl').read_text().splitlines()]
for sid in ids:
 s=json.loads((r/'iclora/outputs'/(sid+'.json')).read_text());assert s['status']=='complete' and s['full_decode_frames']==241
PY
bash "$TASK/h3/scripts/run.sh"
