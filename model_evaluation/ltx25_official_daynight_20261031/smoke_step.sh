#!/usr/bin/env bash
set -euo pipefail
ROOT="${LTX_BENCH_ROOT:?}"
source "$ROOT/stage_job.sh"
exec "$ROOT/runtime/venv/bin/python" "$ROOT/run_worker.py" --smoke
