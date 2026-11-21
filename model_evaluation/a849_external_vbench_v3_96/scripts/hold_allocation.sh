#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_96
[[ -e "$TASK/metadata/retain_allocation.request" ]] || exit 0
printf 'Evaluation stages ended; retaining allocation %s at user request until explicit release or scheduler time limit.\n' "${SLURM_JOB_ID:?}" > "$TASK/logs/allocation_hold.log"
if [[ -f "$TASK/metadata/extend104_authorized.json" ]]; then
 NEXT=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_104
 if ! bash "$NEXT/scripts/gpu_pipeline.sh" >> "$NEXT/logs/pipeline_job_${SLURM_JOB_ID}.log" 2>&1; then
  printf 'Extended pipeline stopped; allocation remains held. Inspect extended pipeline log.\n' >> "$TASK/logs/allocation_hold.log"
 fi
fi
while [[ -e "$TASK/metadata/retain_allocation.request" ]]; do sleep 30; done
