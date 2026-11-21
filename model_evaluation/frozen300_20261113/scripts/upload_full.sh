#!/usr/bin/env bash
set -euo pipefail
[[ -z ${SLURM_JOB_ID:-} ]] || { echo 'CPU upload runs on login' >&2; exit 64; }
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/frozen300_20261113
PY=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/env/bin/python
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY PYTHONPATH
export CUDA_VISIBLE_DEVICES=""
"$PY" "$TASK/scripts/upload_parallel.py" \
 --local "$TASK/frozen300_20261113.zip" --remote /apps/bypy/lance_review/frozen300_20261113.zip \
 --state "$TASK/metadata/baidu_upload.json" \
 --bypy-config /scratch/prj0000000275/jcy/ljy/video_collection/dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json
"$PY" - "$TASK" <<'PY'
import json,sys,datetime
from pathlib import Path
r=Path(sys.argv[1]);state=json.loads((r/'metadata/baidu_upload.json').read_text());assert state['status']=='complete'
p={'status':'complete','remote':'/apps/bypy/lance_review/frozen300_20261113.zip','only_zip_uploaded':True,'extra_remote_verification':False,'uploader_receipt':'metadata/baidu_upload.json','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'metadata/delivery.json').write_text(json.dumps(p,indent=2)+'\n')
print('DELIVERY_COMPLETE',p['remote'],flush=True)
PY
