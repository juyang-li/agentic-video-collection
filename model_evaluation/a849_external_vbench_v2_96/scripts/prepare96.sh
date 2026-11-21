#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/a849_external_vbench_v2_96
PY=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
pids=()
for shard in 0 1 2 3; do
    "$PY" "$TASK/scripts/media.py" --shard "$shard" --shards 4 > "$TASK/logs/media96_shard$shard.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]]
bash "$TASK/scripts/run.sh" cache
bash "$TASK/scripts/run.sh" conditions
"$PY" "$TASK/scripts/mask_sheets.py"
echo '96-task preparation complete; direct mask review required before inference'
