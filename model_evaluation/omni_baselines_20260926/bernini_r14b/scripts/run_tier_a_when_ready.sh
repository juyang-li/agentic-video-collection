#!/usr/bin/env bash
set -euo pipefail

root=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/omni_baselines_20260926
bench="$root/bernini_r14b"
runtime=/tmp/omni_bernini_20260926
model="$runtime/Bernini-R-Diffusers"
env="$runtime/env"
gpu_count=${BERNINI_GPU_COUNT:-4}
# A 144 GB H200 can hold one 81-frame Bernini inference replica.  With four
# H200s, Ulysses=1 gives four data-parallel samples at once.  Set
# BERNINI_ULYSSES=2 to fall back to the verified 2x2 topology if needed.
ulysses=${BERNINI_ULYSSES:-1}

# The watcher may start before the downloader has created component folders.
mkdir -p "$model/transformer" "$model/transformer_2" "$model/text_encoder"

while true; do
  transformer_count=$(find "$model/transformer" -maxdepth 1 -name 'diffusion_pytorch_model-*-of-*.safetensors' 2>/dev/null | wc -l)
  transformer2_count=$(find "$model/transformer_2" -maxdepth 1 -name 'diffusion_pytorch_model-*-of-*.safetensors' 2>/dev/null | wc -l)
  text_count=$(find "$model/text_encoder" -maxdepth 1 -name 'model-*-of-*.safetensors' 2>/dev/null | wc -l)
  incomplete_count=$(find "$model" -name '*.incomplete' 2>/dev/null | wc -l)
  printf '%s waiting model: transformer=%s transformer_2=%s text=%s incomplete=%s\n' \
    "$(date -Is)" "$transformer_count" "$transformer2_count" "$text_count" "$incomplete_count"
  if [[ "$transformer_count" -eq 12 && "$transformer2_count" -eq 12 && \
        "$text_count" -eq 3 && "$incomplete_count" -eq 0 ]]; then
    break
  fi
  sleep 30
done

"$env/bin/python" - <<'PY'
import flash_attn, torch, transformers, diffusers, veomni
print("environment ready", torch.__version__, transformers.__version__, diffusers.__version__, flush=True)
PY

"$env/bin/python" "$bench/scripts/split_orientation.py" \
  --inputs "$bench/manifests/tasks_tier_a.json" \
  --output-prefix "$bench/manifests/tasks_tier_a"

# VACE uses four independent replicas first. Avoid overlapping GPU memory with
# Bernini's four-way Ulysses group.
while [[ ! -f "$root/vace/.tier_a_4gpu_done" ]]; do
  printf '%s waiting for VACE four-GPU completion\n' "$(date -Is)"
  sleep 30
done

# Keep the smoke gate resumable too: outputs created before a preemption must
# not be regenerated.
"$env/bin/python" "$bench/scripts/filter_pending.py" \
  --inputs "$bench/manifests/tasks_smoke.json" \
  --output "$bench/manifests/tasks_smoke_pending.json"
if [[ $("$env/bin/python" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
       "$bench/manifests/tasks_smoke_pending.json") -gt 0 ]]; then
  bash "$bench/scripts/run_inference.sh" "$bench/manifests/tasks_smoke_pending.json" \
    "$gpu_count" "$ulysses"
fi

"$env/bin/python" "$bench/scripts/filter_pending.py" \
  --inputs "$bench/manifests/tasks_tier_a_landscape.json" \
  --output "$bench/manifests/tasks_tier_a_landscape_pending.json"
if [[ $("$env/bin/python" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
       "$bench/manifests/tasks_tier_a_landscape_pending.json") -gt 0 ]]; then
  BERNINI_WIDTH=848 BERNINI_HEIGHT=480 \
    bash "$bench/scripts/run_inference.sh" "$bench/manifests/tasks_tier_a_landscape_pending.json" \
      "$gpu_count" "$ulysses"
fi

"$env/bin/python" "$bench/scripts/filter_pending.py" \
  --inputs "$bench/manifests/tasks_tier_a_portrait.json" \
  --output "$bench/manifests/tasks_tier_a_portrait_pending.json"
if [[ $("$env/bin/python" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
       "$bench/manifests/tasks_tier_a_portrait_pending.json") -gt 0 ]]; then
  BERNINI_WIDTH=480 BERNINI_HEIGHT=848 \
    bash "$bench/scripts/run_inference.sh" "$bench/manifests/tasks_tier_a_portrait_pending.json" \
      "$gpu_count" "$ulysses"
fi

touch "$bench/inference_tier_a.done"
printf '%s Bernini Tier A complete\n' "$(date -Is)"
