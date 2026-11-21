#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue && "${SLURM_JOB_ID:-}" == 202896 ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_tripvvt_pilot32_20261115
PY=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python
[[ -f "$TASK/runtime/cache_ready" && ! -f "$TASK/metadata/generation_gate.json" ]] || exit 65
IFS=',' read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:?}"
[[ "${#devices[@]}" == 4 ]] || exit 66
pids=()
for shard in 0 1 2 3;do CUDA_VISIBLE_DEVICES="${devices[$shard]}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" "$TASK/scripts/repair_masks.py" --shard "$shard" > "$TASK/logs/repair_masks_${shard}.log" 2>&1 & pids+=("$!");done
failed=0
for pid in "${pids[@]}";do wait "$pid" || failed=1;done
[[ "$failed" == 0 ]]
