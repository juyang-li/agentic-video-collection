#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY PYTHONPATH HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME="$TASK/runtime/hf" TORCH_HOME="$TASK/runtime/torch" VBENCH_CACHE_DIR="$TASK/runtime/vbench_models"
export LD_LIBRARY_PATH="/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
IFS=',' read -r -a ids <<< "${CUDA_VISIBLE_DEVICES:?}"
[[ "${#ids[@]}" == 4 ]] || exit 65
pids=()
for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${ids[$shard]}" "$TASK/runtime/vbench/bin/python" "$TASK/scripts/score.py" --stage "${1:?smoke|all}" --shard "$shard" > "$TASK/logs/score_${1}_$shard.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
