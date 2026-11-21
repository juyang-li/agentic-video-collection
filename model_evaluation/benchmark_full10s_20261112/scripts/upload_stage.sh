#!/usr/bin/env bash
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == testqueue ]] || exit 64
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112
REVIEW_UPLOAD_ENV=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/env
NAME=full10s_base_a849_80_scroll_20261113
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY PYTHONPATH
export CUDA_VISIBLE_DEVICES=""
"$REVIEW_UPLOAD_ENV/bin/python" "$TASK/scripts/upload_baidu_xpan.py" \
 --local "$TASK/review/$NAME.zip" --remote "/apps/bypy/lance_review/$NAME.zip" \
 --state "$TASK/review/$NAME.baidu_upload.json" \
 --bypy-config /scratch/prj0000000275/jcy/ljy/video_collection/dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json
"$REVIEW_UPLOAD_ENV/bin/python" - "$TASK" "$NAME" <<'PY'
from pathlib import Path
import json,sys,time
root=Path(sys.argv[1]);name=sys.argv[2];state=json.loads((root/'review'/(name+'.baidu_upload.json')).read_text());assert state['status']=='complete'
receipt={'status':'complete','scope':'stage delivery: base and A849 only, 80 samples','remote':'/apps/bypy/lance_review/'+name+'.zip','uploader_receipt':str(root/'review'/(name+'.baidu_upload.json')),'completed_unix':time.time(),'only_zip_uploaded':True,'extra_remote_verification':False}
(root/'metadata/stage_delivery_base_a849.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
with (root/'README.md').open('a') as f:f.write('\n阶段审核包已由上传器返回complete：'+receipt['remote']+'。仅基座/A849共160条输出、80条样本；其余模型仍在运行或待决定。仅上传ZIP，无额外远端复核。\n')
print('STAGE_DELIVERY_COMPLETE',receipt['remote'],flush=True)
PY
