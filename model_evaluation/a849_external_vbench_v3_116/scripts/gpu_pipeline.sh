#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_116
while [[ ! -f "$TASK/metadata/pipeline_ready.json" ]];do sleep 30;done
if flock "$TASK/runtime/gpu_generation.lock" bash "$TASK/scripts/generate116_locked.sh" >> "$TASK/logs/pipeline_${SLURM_JOB_ID:?}.log" 2>&1;then
 pipeline_rc=0
else
 pipeline_rc=$?
 printf 'Pipeline failed (%s); allocation retained for repair.\n' "$pipeline_rc" >&2
fi
while [[ -e "$TASK/../a849_external_vbench_v3_96/metadata/retain_allocation.request" ]];do sleep 30;done
exit "$pipeline_rc"
