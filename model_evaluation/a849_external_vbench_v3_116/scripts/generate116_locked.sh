#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_116
VACE=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python
H3PY=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/env/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$VACE" "$TASK/scripts/reuse_outputs.py"
"$VACE" "$TASK/scripts/check_generation.py" --phase parents
"$VACE" "$TASK/scripts/reuse_h3.py"
if ! "$VACE" "$TASK/scripts/check_generation.py" --phase native;then
 bash "$TASK/scripts/native.sh" cache
 bash "$TASK/scripts/native.sh" native
fi
if ! "$VACE" "$TASK/scripts/check_generation.py" --phase vace;then bash "$TASK/scripts/native.sh" vace;fi
"$H3PY" "$TASK/scripts/h3_worker.py" --stage all
"$VACE" "$TASK/scripts/check_generation.py" --phase all
"$VACE" -c 'import sys,os,time;sys.path.insert(0,sys.argv[1]);from common import ROOT,atomic;atomic(ROOT/"metadata/gpu_generation.json",dict(status="complete",job_id=os.environ["SLURM_JOB_ID"],completed_unix=time.time(),expected_native_outputs=464))' "$TASK/scripts"
while [[ ! -f "$TASK/metadata/h3_derivation.json" ]] || ! "$VACE" -c 'import json,sys;raise SystemExit(json.load(open(sys.argv[1]))["status"]!="complete")' "$TASK/metadata/h3_derivation.json";do sleep 30;done
"$VACE" "$TASK/scripts/reuse_scores.py"
bash "$TASK/scripts/score_gpu.sh"
