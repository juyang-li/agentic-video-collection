#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v1
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
VACE=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$NATIVE/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/external_vbench_triton"
mkdir -p "$TRITON_CACHE_DIR"
mode="${1:?cache|conditions|smoke|all}"
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:?Slurm must expose allocated devices}"
[[ "${#gpu_ids[@]}" == 4 ]] || exit 65
run_four() {
    local kind="$1"; shift
    pids=()
    for shard in 0 1 2 3; do
        if [[ "$kind" == native ]]; then
            CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}" bash "$NATIVE/scripts/common/run_trainer_python.sh" "$@" --shard "$shard" > "$TASK/logs/${mode}_${label}_$shard.log" 2>&1 &
        else
            CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}" "$VACE" "$@" --shard "$shard" > "$TASK/logs/${mode}_${label}_$shard.log" 2>&1 &
        fi
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]]
}
if [[ "$mode" == cache ]]; then
    for phase in media text; do label="$phase"; run_four native "$TASK/scripts/native_cache.py" --stage "$phase"; done
elif [[ "$mode" == conditions ]]; then
    label=conditions;run_four vace "$TASK/scripts/prepare_conditions.py" --stage all
else
    "$VACE" "$TASK/scripts/gate.py" --before "$mode"
    for model in ltx_base a849 vace; do
        label="$model"
        if [[ "$mode" == all ]]; then
            kind=native
            [[ "$model" == vace ]] && kind=vace
            run_four "$kind" "$TASK/scripts/dispatch_worker.py" --stage all --model "$model"
        elif [[ "$model" == vace ]]; then
            run_four vace "$TASK/scripts/vace_worker.py" --stage "$mode"
        else
            run_four native "$TASK/scripts/native_worker.py" --stage "$mode" --model "$model"
        fi
    done
    "$VACE" "$TASK/scripts/gate.py" --after "$mode"
fi
