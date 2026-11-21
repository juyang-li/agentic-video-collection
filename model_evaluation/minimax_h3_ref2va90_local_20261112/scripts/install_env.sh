#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue && "${SLURM_JOB_ID:-}" == 199810 ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export UV_CACHE_DIR="${TMPDIR:-/tmp}/h3_uv_cache" SGLANG_BUILD_RUST_EXTS=none PYTHONNOUSERSITE=1
/home/users/astar/cfar/stuchengyou/.local/bin/uv pip install --python "$TASK/runtime/env/bin/python" -e "$TASK/runtime/sglang/python[diffusion]" --prerelease=allow
/home/users/astar/cfar/stuchengyou/.local/bin/uv pip freeze --python "$TASK/runtime/env/bin/python" > "$TASK/metadata/requirements.lock.txt"
printf 'ENVIRONMENT_READY\n'
