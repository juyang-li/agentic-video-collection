#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/runtime" "$ROOT/logs" "$ROOT/metadata"
if [ ! -d "$ROOT/runtime/ComfyUI/.git" ]; then
  git clone --depth 1 https://github.com/Comfy-Org/ComfyUI.git "$ROOT/runtime/ComfyUI"
fi
if [ ! -d "$ROOT/runtime/ComfyUI/custom_nodes/ComfyUI-LTXVideo/.git" ]; then
  git -c filter.lfs.process= -c filter.lfs.smudge=cat -c filter.lfs.required=false clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git "$ROOT/runtime/ComfyUI/custom_nodes/ComfyUI-LTXVideo"
fi
if [ ! -x "$ROOT/runtime/venv/bin/python" ]; then
  uv venv --python /apps/miniforge/24.11.3/bin/python "$ROOT/runtime/venv"
fi
uv pip install --python "$ROOT/runtime/venv/bin/python" -r "$ROOT/runtime/ComfyUI/requirements.txt" -r "$ROOT/runtime/ComfyUI/custom_nodes/ComfyUI-LTXVideo/requirements.txt" imageio-ffmpeg websocket-client
"$ROOT/runtime/venv/bin/python" -m pip freeze > "$ROOT/metadata/environment_freeze.txt" 2>/dev/null || uv pip freeze --python "$ROOT/runtime/venv/bin/python" > "$ROOT/metadata/environment_freeze.txt"
git -C "$ROOT/runtime/ComfyUI" rev-parse HEAD > "$ROOT/metadata/comfyui_revision.txt"
git -C "$ROOT/runtime/ComfyUI/custom_nodes/ComfyUI-LTXVideo" rev-parse HEAD > "$ROOT/metadata/ltx_nodes_revision.txt"
echo 'Environment ready'
