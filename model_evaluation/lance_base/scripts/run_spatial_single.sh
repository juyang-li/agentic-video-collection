#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/scratch/prj0000000275/jcy/ljy/video_collection
EVAL_ROOT="$REPO_ROOT/model_evaluation/lance_base"
RUNTIME_ROOT="$EVAL_ROOT/runtime/upstream"
MODEL_PATH="$RUNTIME_ROOT/downloads/Lance_3B_Video"
CONFIG_PATH="$EVAL_ROOT/configs/spatial_full_seed42/shard_00.json"
RUN_NAME=omni_edit_spatial_base_seed42_shard00
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-6760}
export TMPDIR=${TMPDIR:-/tmp/lance_omni_edit_spatial_123169}
mkdir -p "$TMPDIR"

exec bash "$REPO_ROOT/model_evaluation/lance_tools/run_video_edit.sh" \
    "$EVAL_ROOT" "$RUNTIME_ROOT" "$MODEL_PATH" "$CONFIG_PATH" "$RUN_NAME"
