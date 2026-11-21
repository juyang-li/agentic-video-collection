#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_96
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
VACE=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python
VBENCH=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v2_96
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$NATIVE/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/a849_v3_triton"
export HF_HOME="$VBENCH/runtime/hf" TORCH_HOME="$VBENCH/runtime/torch" VBENCH_CACHE_DIR="$VBENCH/runtime/vbench_models"
mkdir -p "$TRITON_CACHE_DIR"
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:?Slurm allocated devices required}"
[[ "${#gpu_ids[@]}" == 4 ]] || exit 65
run_four() {
 local label="$1"; shift
 local pids=() failed=0
 for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}" "$@" --shard "$shard" > "$TASK/logs/${label}_${shard}.log" 2>&1 &
  pids+=("$!")
 done
 for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
 [[ "$failed" == 0 ]]
}
case "${1:?conditions|cache|generate|score|static}" in
 conditions) run_four conditions "$VACE" "$TASK/scripts/prepare_conditions.py" ;;
 cache)
  for stage in media text; do
   run_four "cache_${stage}" bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/native_cache.py" --stage "$stage"
  done ;;
 generate)
  for model in ltx_base a849; do
   run_four "$model" bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/dispatch_worker.py" --model "$model"
  done
  run_four vace "$VACE" "$TASK/scripts/dispatch_worker.py" --model vace ;;
 score) run_four quality "$VBENCH/runtime/vbench/bin/python" "$TASK/scripts/score.py" quality ;;
 static) run_four static "$VBENCH/runtime/vbench/bin/python" "$TASK/scripts/score.py" static ;;
 *) exit 64 ;;
esac
