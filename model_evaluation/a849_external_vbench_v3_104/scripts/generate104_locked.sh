#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_104
VACE=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python
H3PY=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/env/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$VACE" "$TASK/scripts/reuse_outputs.py"
# The dependency continuation also resumes unfinished original96 generation.
# Original manifests/workers own those outputs; the104 dispatcher never regenerates them.
if ! "$VACE" "$TASK/scripts/check_generation.py" --phase parents;then
 primary_job=$("$VACE" -c 'import json,sys;print(json.load(open(sys.argv[1]))["primary_job_id"])' "$TASK/metadata/execution_state.json")
 [[ "${SLURM_JOB_ID:?}" != "$primary_job" ]] || { printf 'Original parent generation is incomplete inside its own primary allocation; inspect the original stage before resuming.\n' >&2;exit 69; }
 # A successful scheduler query is required; an observation failure is not proof of termination.
 primary_state=$(squeue -h -u "$(id -un)" -o '%i %T' | awk -v jid="$primary_job" '$1==jid {print $2}')
 [[ -z "$primary_state" ]] || { printf 'Primary job is still live (%s); do not duplicate parent generation.\n' "$primary_state" >&2;exit 70; }
 bash "$TASK/../a849_external_vbench_v3_96/scripts/run.sh" generate
 "$VACE" "$TASK/scripts/reuse_outputs.py"
 "$VACE" "$TASK/scripts/check_generation.py" --phase parents
fi
if ! "$VACE" "$TASK/scripts/check_generation.py" --phase native;then
 bash "$TASK/scripts/native.sh" cache
 bash "$TASK/scripts/native.sh" native
fi
if ! "$VACE" "$TASK/scripts/check_generation.py" --phase vace;then bash "$TASK/scripts/native.sh" vace;fi
# The login CPU preparation must finish before any H3 request is issued.
while [[ ! -f "$TASK/metadata/h3_preflight.json" ]];do
 [[ ! -f "$TASK/metadata/h3_prepare_error.json" ]] || exit 68
 sleep 30
done
"$H3PY" "$TASK/scripts/h3_worker.py" --stage all
"$VACE" "$TASK/scripts/check_generation.py" --phase all
"$VACE" -c 'from pathlib import Path;import sys,os,time;sys.path.insert(0,sys.argv[1]);from common import ROOT,atomic;atomic(ROOT/"metadata/gpu_generation.json",dict(status="complete",job_id=os.environ["SLURM_JOB_ID"],completed_unix=time.time(),expected_native_outputs=416))' "$TASK/scripts"
# Lossless0–4s scoring clips are prepared asynchronously by the login CPU watcher.
while [[ ! -f "$TASK/metadata/h3_derivation.json" ]] || ! "$VACE" -c 'import json,sys;raise SystemExit(json.load(open(sys.argv[1]))["status"]!="complete")' "$TASK/metadata/h3_derivation.json";do sleep 30;done
bash "$TASK/scripts/score_gpu.sh"
