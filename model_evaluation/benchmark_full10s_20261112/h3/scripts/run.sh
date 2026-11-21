#!/usr/bin/env bash
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3
exec 9>"$TASK/logs/driver.lock"
flock -n 9 || exit 73
exec 7>"$TASK/../logs/allocation_${SLURM_JOB_ID}.lock"
flock 7
PY="$TASK/runtime/env/bin/python"
bash "$TASK/scripts/serve.sh" > "$TASK/logs/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
for stage in ${H3_STAGES:-pilot first all}; do
 "$PY" "$TASK/scripts/client.py" --stage "$stage" > "$TASK/logs/client_${stage}.log" 2>&1
 "$PY" "$TASK/scripts/protocol_gate.py" --stage "$stage" >> "$TASK/logs/driver.log" 2>&1
done
