#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue && "${SLURM_JOB_ID:-}" == 199810 ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
exec 9>"$TASK/metadata/driver.lock"
flock -n 9 || exit 75
while [[ ! -f "$TASK/metadata/gate_pilot.json" ]]; do
  "$TASK/runtime/env/bin/python" - "$TASK" <<'PYCHECK'
import sys,json
from pathlib import Path
root=Path(sys.argv[1])
for f in (root/'outputs').glob('*.json'):
 state=json.loads(f.read_text());assert state['status']!='failed',f'{f.name} failed; investigate before expansion'
PYCHECK
  sleep 30
done
"$TASK/runtime/env/bin/python" - "$TASK" <<'PY'
import sys,json
from pathlib import Path
r=Path(sys.argv[1]);assert json.loads((r/'metadata/gate_pilot.json').read_text())['status']=='passed'
PY
"$TASK/runtime/env/bin/python" "$TASK/scripts/protocol_gate.py" --stage pilot
bash "$TASK/scripts/run_client.sh" first > "$TASK/logs/first9.log" 2>&1
"$TASK/runtime/env/bin/python" "$TASK/scripts/protocol_gate.py" --stage first
"$TASK/runtime/env/bin/python" "$TASK/scripts/status.py"
bash "$TASK/scripts/run_client.sh" all > "$TASK/logs/all.log" 2>&1
"$TASK/runtime/env/bin/python" "$TASK/scripts/status.py"
"$TASK/runtime/env/bin/python" "$TASK/scripts/protocol_gate.py" --stage all
"$TASK/runtime/env/bin/python" "$TASK/scripts/package.py"
bash "$TASK/scripts/upload.sh"

"$TASK/runtime/env/bin/python" "$TASK/scripts/finalize.py"
