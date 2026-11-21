#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue && "${SLURM_JOB_ID:-}" == 199810 ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export UV_CACHE_DIR="${TMPDIR:-/tmp}/h3_uv_cache" HF_XET_CACHE="${TMPDIR:-/tmp}/h3_xet_cache" HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 SGLANG_BUILD_RUST_EXTS=none UV_LINK_MODE=copy
"$NATIVE/runtime/envs/trainer/bin/python" "$TASK/scripts/inspect_remote.py"
"$NATIVE/runtime/envs/trainer/bin/python" "$TASK/scripts/download_model.py" > "$TASK/logs/download_model.log" 2>&1 &
download_pid=$!
if [[ ! -d "$TASK/runtime/sglang/.git" ]]; then git clone --depth 1 https://github.com/sgl-project/sglang.git "$TASK/runtime/sglang"; fi
if [[ -f "$TASK/metadata/sglang_commit.txt" ]]; then
  pinned_commit=$(cat "$TASK/metadata/sglang_commit.txt")
  if [[ "$(git -C "$TASK/runtime/sglang" rev-parse HEAD)" != "$pinned_commit" ]]; then
    git -C "$TASK/runtime/sglang" fetch --depth 1 origin "$pinned_commit"
    git -C "$TASK/runtime/sglang" checkout --detach "$pinned_commit"
  fi
else
  git -C "$TASK/runtime/sglang" rev-parse HEAD > "$TASK/metadata/sglang_commit.txt"
fi
if [[ -s "$TASK/metadata/sglang_build_portability.patch" ]]; then
  if ! git -C "$TASK/runtime/sglang" apply --reverse --check "$TASK/metadata/sglang_build_portability.patch"; then
    git -C "$TASK/runtime/sglang" apply --check "$TASK/metadata/sglang_build_portability.patch"
    git -C "$TASK/runtime/sglang" apply "$TASK/metadata/sglang_build_portability.patch"
  fi
fi
if [[ ! -x "$TASK/runtime/env/bin/python" ]]; then /home/users/astar/cfar/stuchengyou/.local/bin/uv venv --python "$NATIVE/runtime/envs/trainer/bin/python" "$TASK/runtime/env"; fi
/home/users/astar/cfar/stuchengyou/.local/bin/uv pip install --python "$TASK/runtime/env/bin/python" -e "$TASK/runtime/sglang/python[diffusion]" --prerelease=allow
/home/users/astar/cfar/stuchengyou/.local/bin/uv pip freeze --python "$TASK/runtime/env/bin/python" > "$TASK/metadata/requirements.lock.txt"
printf 'ENVIRONMENT_READY\n'
wait "$download_pid"
"$TASK/runtime/env/bin/python" "$TASK/scripts/install_ffmpeg.py"
printf 'BOOTSTRAP_COMPLETE\n'
