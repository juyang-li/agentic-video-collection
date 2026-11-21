#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/h3
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/h3_triton" FLASHINFER_WORKSPACE_BASE="${TMPDIR:-/tmp}/h3_flashinfer"
export H3_MEMORY_LOG_DIR="$TASK/logs/memory" PYTHONPATH="$TASK/runtime/instrumentation"
export PATH="$TASK/runtime/bin:$TASK/runtime/env/bin:$PATH"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
mkdir -p "$TRITON_CACHE_DIR" "$H3_MEMORY_LOG_DIR" "$TASK/outputs/server"
"$TASK/runtime/env/bin/python" - "$TASK" <<'PY'
import torch,json,sys
from pathlib import Path
r=Path(sys.argv[1]);assert (r/'metadata/download_complete.json').exists()
assert torch.cuda.device_count()==4
for i in range(4):
 assert 'H200' in torch.cuda.get_device_name(i)
 free,total=torch.cuda.mem_get_info(i);assert total-free<2*1024**3, f'GPU {i} busy'
print('FOUR_FREE_H200_CONFIRMED',flush=True)
PY
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu --format=csv -l 1 > "$TASK/logs/memory/nvml.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
sglang serve --model-path "$TASK/runtime/models/H3/Ref2VA" --served-model-name MiniMaxAI/MiniMax-H3 \
 --num-gpus 4 --tp-size 2 --ulysses-degree 2 --encoder-parallel auto \
 --performance-mode speed --enable-torch-compile false \
 --warmup-mode off --batching-max-size 1 --host 127.0.0.1 --port 30011 \
 --output-path "$TASK/outputs/server" &
service_pid=$!
trap 'kill "$service_pid" "$telemetry_pid" 2>/dev/null || true' EXIT
wait "$service_pid"
