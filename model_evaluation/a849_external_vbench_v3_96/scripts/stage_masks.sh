#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v3_96
VACE=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
pids=()
for shard in 0 1 2 3; do
 CUDA_VISIBLE_DEVICES="$shard" "$VACE" "$TASK/scripts/prepare_conditions.py" --candidates --shard "$shard" > "$TASK/logs/stage_masks_${shard}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
