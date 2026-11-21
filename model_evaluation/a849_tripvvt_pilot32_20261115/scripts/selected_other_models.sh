#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue && "${SLURM_JOB_ID:-}" == 202896 ]] || exit 64
TASK=$(cd -- "$(dirname -- "$0")/.." && pwd)
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
VACE=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$NATIVE/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec 9>"$TASK/runtime/job202896.lock"
flock 9
"$VACE" -c 'import sys;sys.path.insert(0,sys.argv[1]);from common import ROOT,read;d=read(ROOT/"metadata/selection_decision.json");assert d["status"]=="complete" and len(d["selected"])==8;t=read(ROOT/"metadata/generation_targets.json");assert set(t["ltx_base"])==set(t["vace"])==set(d["selected"])' "$TASK/scripts"
IFS=',' read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:?}"
[[ "${#devices[@]}" == 4 ]] || exit 65
four() {
 local label="$1";shift;local pids=() failed=0
 for shard in 0 1 2 3;do CUDA_VISIBLE_DEVICES="${devices[$shard]}" "$@" --shard "$shard" > "$TASK/logs/${label}_${shard}.log" 2>&1 & pids+=("$!");done
 for pid in "${pids[@]}";do wait "$pid" || failed=1;done
 [[ "$failed" == 0 ]]
}
four selected_base bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/dispatch_worker.py" --model ltx_base
four selected_vace "$VACE" "$TASK/scripts/dispatch_worker.py" --model vace
