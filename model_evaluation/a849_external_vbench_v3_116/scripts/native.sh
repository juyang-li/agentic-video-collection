#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_116
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
VACE=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$NATIVE/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
IFS=',' read -r -a ids <<< "${CUDA_VISIBLE_DEVICES:?}"
[[ "${#ids[@]}" == 4 ]] || exit 65
four() {
 local label="$1";shift;local pids=() failed=0
 for shard in 0 1 2 3; do CUDA_VISIBLE_DEVICES="${ids[$shard]}" "$@" --shard "$shard" > "$TASK/logs/${label}_${shard}.log" 2>&1 & pids+=("$!");done
 for pid in "${pids[@]}";do wait "$pid" || failed=1;done
 [[ "$failed" == 0 ]]
}
case "${1:?cache|native|vace}" in
 cache) four cache bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/native_cache.py" --stage media
        four text bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/native_cache.py" --stage text ;;
 native) for model in ltx_base a849;do four "$model" bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/dispatch_worker.py" --model "$model";done ;;
 vace) four vace "$VACE" "$TASK/scripts/dispatch_worker.py" --model vace ;;
 *) exit 64;;
esac
