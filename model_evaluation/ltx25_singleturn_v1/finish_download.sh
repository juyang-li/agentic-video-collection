#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INITIAL_DOWNLOAD_PID="${1:?initial downloader PID required}"
while kill -0 "$INITIAL_DOWNLOAD_PID" 2>/dev/null; do sleep 10; done
exec /scratch/prj0000000275/jcy/.conda/envs/xks_lance/bin/python "$ROOT/download_models.py"
