#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
[[ "${SLURM_JOB_ID:-}" == 202896 ]] || exit 65
TASK=$(cd -- "$(dirname -- "$0")/.." && pwd)
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
VACE=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python
H3PY=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/env/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$NATIVE/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
IFS=',' read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:?}"
[[ "${#devices[@]}" == 4 ]] || exit 66
exec 9>"$TASK/runtime/job202896.lock"
flock -n 9 || exit 67
four() {
 local label="$1";shift;local pids=() failed=0
 for shard in 0 1 2 3; do CUDA_VISIBLE_DEVICES="${devices[$shard]}" "$@" --shard "$shard" > "$TASK/logs/${label}_${shard}.log" 2>&1 & pids+=("$!");done
 for pid in "${pids[@]}";do wait "$pid" || failed=1;done
 [[ "$failed" == 0 ]]
}
"$VACE" -c 'import os,json,sys,time;from pathlib import Path;p=Path(sys.argv[1]);p.write_text(json.dumps(dict(status="preparing",job_id=os.environ["SLURM_JOB_ID"],step_id=os.environ.get("SLURM_STEP_ID"),partition=os.environ["SLURM_JOB_PARTITION"],cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],pid=os.getpid(),started_unix=time.time()),indent=2))' "$TASK/metadata/gpu_execution.json"
four masks "$VACE" "$TASK/scripts/prepare_conditions.py"
touch "$TASK/runtime/masks_ready"
four cache bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/native_cache.py" --stage media
four text bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/native_cache.py" --stage text
touch "$TASK/runtime/cache_ready"
# The reviewer on login creates these receipts only after direct inspection.
while [[ ! -f "$TASK/metadata/generation_gate.json" || ! -f "$TASK/metadata/h3_preflight.json" ]];do sleep 15;done
four a849 bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/dispatch_worker.py" --model a849
"$H3PY" "$TASK/scripts/h3_worker.py" --stage all > "$TASK/logs/h3_worker.log" 2>&1
"$VACE" -c 'import sys,time;sys.path.insert(0,sys.argv[1]);from common import ROOT,read,rows,atomic;items=rows(ROOT/"metadata/inputs.jsonl");assert all(read(ROOT/"outputs"/m/(r["id"]+".json"))["status"]=="complete" for r in items for m in ["a849","h3_pdd8_native"]);atomic(ROOT/"metadata/gpu_generation.json",dict(status="complete",expected_outputs=64,completed_unix=time.time()))' "$TASK/scripts"
# Returning this step does not cancel/release the user's parent allocation.
