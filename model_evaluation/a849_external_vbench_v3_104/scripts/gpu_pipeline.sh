#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_104
if [[ -f "$TASK/metadata/final_suite_redirect.json" ]];then
 next_task=$(/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1]))["root"])' "$TASK/metadata/final_suite_redirect.json")
 exec bash "$next_task/scripts/gpu_pipeline.sh"
fi
# Kernel locks cover generation/scoring only and are released even on failure.
if flock -n "$TASK/runtime/gpu_generation.lock" bash "$TASK/scripts/generate104_locked.sh";then
 exit 0
else
 pipeline_rc=$?
 printf 'Extended pipeline stopped with exit %s in job %s; preserving this allocation at user request.\n' "$pipeline_rc" "${SLURM_JOB_ID:?}" >&2
 PY=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python
 "$PY" -c 'import sys,os,time;sys.path.insert(0,sys.argv[1]);from common import ROOT,atomic;atomic(ROOT/("metadata/gpu_pipeline_failure_"+os.environ["SLURM_JOB_ID"]+".json"),dict(status="failed",exit_code=int(sys.argv[2]),job_id=os.environ["SLURM_JOB_ID"],unix=time.time(),allocation_held_at_user_request=True))' "$TASK/scripts" "$pipeline_rc"
 while [[ -e /scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_96/metadata/retain_allocation.request ]];do sleep 30;done
 exit "$pipeline_rc"
fi
