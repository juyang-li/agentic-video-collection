#!/usr/bin/env bash
set -euo pipefail
ROOT="${LTX_BENCH_ROOT:?}"
LTX_JOB_CACHE="${TMPDIR:-/tmp}/ltx25_${SLURM_JOB_ID}"
mkdir -p "$LTX_JOB_CACHE/models"
echo "Staging model files to $LTX_JOB_CACHE/models"
"$ROOT/runtime/venv/bin/python" - "$ROOT" "$LTX_JOB_CACHE" <<'PY'
from pathlib import Path
import json,shutil,sys
root,cache=map(Path,sys.argv[1:])
rows=json.loads((root/'metadata/model_lock.json').read_text())
for row in rows:
    src=Path(row['local']);rel=src.relative_to(root/'models');dst=cache/'models'/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists() or dst.stat().st_size!=row['bytes']:
        tmp=dst.with_suffix('.copying');shutil.copyfile(src,tmp);tmp.replace(dst)
    print('staged',rel,flush=True)
(cache/'model_paths.yaml').write_text('ltx25_local:\n  base_path: '+str(cache/'models')+'\n  diffusion_models: diffusion_models\n  text_encoders: text_encoders\n  vae: vae\n  loras: loras\n')
PY
export LTX_MODEL_PATHS="$LTX_JOB_CACHE/model_paths.yaml"
export TORCHINDUCTOR_CACHE_DIR="$LTX_JOB_CACHE/torchinductor"
