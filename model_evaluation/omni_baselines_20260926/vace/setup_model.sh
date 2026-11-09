#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/models /tmp/hf-vace-cache
export HF_HOME=/tmp/hf-vace-cache
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT=1800

/scratch/prj0000000275/jcy/sx/vace_cv_control_pilot_h200_v1/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    "Wan-AI/Wan2.1-VACE-14B",
    local_dir="/tmp/models/Wan2.1-VACE-14B",
    max_workers=8,
)
print(path, flush=True)
PY

test -f /tmp/models/Wan2.1-VACE-14B/config.json
if find /tmp/models/Wan2.1-VACE-14B -name '*.incomplete' -print -quit | grep -q .; then
  echo 'VACE download still contains incomplete files' >&2
  exit 1
fi
touch /tmp/models/Wan2.1-VACE-14B/.download_complete
