#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_identity
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
VACE=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$NATIVE/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/a849_v3_identity_triton"
mkdir -p "$TRITON_CACHE_DIR"
"$VACE" -c 'import json; from pathlib import Path; assert json.loads(Path("'"$TASK"'/metadata/input_review.json").read_text())["status"]=="approved"'
run_four() {
 local phase="$1"; shift
 local pids=() failed=0
 for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" "$@" --shard "$shard" > "$TASK/logs/${phase}_${shard}.log" 2>&1 &
  pids+=("$!")
 done
 for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
 [[ "$failed" == 0 ]]
}
run_four cache bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/native_cache.py" --stage media
for model in ltx_base a849; do
 run_four "$model" bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/native_worker.py" --model "$model" --stage all
done
run_four vace "$VACE" "$TASK/scripts/vace_worker.py" --stage all
