#!/usr/bin/env bash
set -euo pipefail

root=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/omni_baselines_20260926
mask_root="$root/vace/masks_dino"
output_root="$root/vace/outputs_tier_a"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/scratch/prj0000000275/jcy/sam2
cd /scratch/prj0000000275/jcy/sam2
/tmp/vace_mask_env/bin/python "$root/vace/generate_vace_masks.py" \
  --manifest "$root/aligned_subset.csv" \
  --output-root "$mask_root" \
  --sam2-checkpoint /scratch/prj0000000275/jcy/sam2_ckpt/sam2.1_hiera_base_plus.pt \
  --gdino /tmp/models/grounding-dino-base \
  --tier A

export PYTHONPATH=/scratch/prj0000000275/jcy/sx/vace_cv_control_pilot_h200_v1/tools/Wan2.1
/scratch/prj0000000275/jcy/sx/vace_cv_control_pilot_h200_v1/venv/bin/python \
  "$root/vace/run_vace_omni.py" \
  --manifest "$root/aligned_subset.csv" \
  --mask-root "$mask_root" \
  --output-root "$output_root" \
  --ckpt-dir /tmp/models/Wan2.1-VACE-14B \
  --wan-repo /scratch/prj0000000275/jcy/sx/vace_cv_control_pilot_h200_v1/tools/Wan2.1 \
  --tier A
