#!/usr/bin/env bash
set -euo pipefail

root=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/omni_baselines_20260926
runner="$root/vace/run_vace_omni.py"
output="$root/vace/outputs_tier_a"
model=/tmp/models/Wan2.1-VACE-14B
python=/scratch/prj0000000275/jcy/sx/vace_cv_control_pilot_h200_v1/venv/bin/python
wan=/scratch/prj0000000275/jcy/sx/vace_cv_control_pilot_h200_v1/tools/Wan2.1

while [[ ! -f "$model/.download_complete" ]]; do
  printf '%s waiting for VACE checkpoint\n' "$(date -Is)"
  sleep 30
done

mkdir -p "$root/vace/logs"
pids=()
for shard in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="$shard"
    export PYTHONPATH="$wan"
    "$python" "$runner" \
      --manifest "$root/aligned_subset.csv" \
      --mask-root "$root/vace/masks_dino" \
      --output-root "$output" \
      --ckpt-dir "$model" \
      --wan-repo "$wan" \
      --tier A --num-shards 4 --shard-index "$shard"
  ) > "$root/vace/logs/tier_a_shard${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo 'one or more VACE shards failed' >&2
  exit 1
fi
touch "$root/vace/.tier_a_4gpu_done"
printf '%s VACE Tier A complete\n' "$(date -Is)"
