#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PILOT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python}
STATUS_PATH="$PILOT_ROOT/runs/ablation_queue_status.json"
TOTAL_STEPS=${TOTAL_STEPS:-500}
ACTIVE_RUN="queue_initializing"

update_status() {
    "$PYTHON_BIN" "$SCRIPT_DIR/write_queue_status.py" "$STATUS_PATH" "$@"
}

on_error() {
    code=$?
    update_status --state failed --active-run "$ACTIVE_RUN" --message "queue exited with code $code"
    exit "$code"
}
trap on_error ERR

run_experiment() {
    local run_name=$1
    local config_name=$2
    ACTIVE_RUN=$run_name
    update_status --state running --active-run "$run_name" --message "training $run_name"
    RUN_NAME="$run_name" \
    DATASET_CONFIG="$PILOT_ROOT/configs/$config_name" \
    TOTAL_STEPS="$TOTAL_STEPS" \
    SAVE_EVERY=499 \
    CKPT_DEBUG_STEPS=249 \
    AUTO_RESUME=False \
      bash "$SCRIPT_DIR/run_ablation_train.sh"
    update_status --state running --completed-run "$run_name" --message "finished $run_name"
}

update_status --state queued --active-run "$ACTIVE_RUN" --message "two-run ablation queue created"
run_experiment mixed_300_200_steps500 mixed_300_200.yaml
run_experiment insave_500_steps500 insave_500.yaml
ACTIVE_RUN=""
update_status \
    --state complete \
    --completed-run mixed_300_200_steps500 \
    --completed-run insave_500_steps500 \
    --message "both ablation experiments completed"
