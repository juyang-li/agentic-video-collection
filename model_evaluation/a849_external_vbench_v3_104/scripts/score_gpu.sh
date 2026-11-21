#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_104
OLD=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v2_96
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME="$OLD/runtime/hf" TORCH_HOME="$OLD/runtime/torch" VBENCH_CACHE_DIR="$OLD/runtime/vbench_models"
IFS=',' read -r -a ids <<< "${CUDA_VISIBLE_DEVICES:?}"
[[ "${#ids[@]}" == 4 ]] || exit 65
four() {
 local label="$1";shift;local pids=() failed=0
 for shard in 0 1 2 3;do CUDA_VISIBLE_DEVICES="${ids[$shard]}" "$@" --shard "$shard" > "$TASK/logs/${label}_${shard}.log" 2>&1 & pids+=("$!");done
 for pid in "${pids[@]}";do wait "$pid" || failed=1;done
 [[ "$failed" == 0 ]]
}
four quality "$OLD/runtime/vbench/bin/python" "$TASK/scripts/score.py" quality
four static "$OLD/runtime/vbench/bin/python" "$TASK/scripts/score.py" static
"$OLD/runtime/vbench/bin/python" -c 'import sys,os,time;sys.path.insert(0,sys.argv[1]);from common import ROOT,atomic;atomic(ROOT/"metadata/gpu_scores_complete.json",dict(status="complete",job_id=os.environ["SLURM_JOB_ID"],completed_unix=time.time()))' "$TASK/scripts"
