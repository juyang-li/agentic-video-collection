#!/usr/bin/env bash
set -euo pipefail
if [[ -n ${SLURM_JOB_ID:-} && ${SLURM_JOB_PARTITION:-} != testqueue ]]; then exit 64; fi
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112
REVIEW_PY=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111/runtime/vace/bin/python
H3_PY=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/minimax_h3_ref2va90_local_20261112/runtime/env/bin/python
NAME=full10s_completed_241outputs_20261113
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY PYTHONPATH
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$REVIEW_PY" "$TASK/scripts/audit_vace.py" --completed-only
"$H3_PY" "$TASK/h3/scripts/protocol_gate.py" --stage completed
"$REVIEW_PY" "$TASK/scripts/package_stage.py" --models ltx_base a849 vace h3 --completed-only --name "$NAME" --workers 2
"$H3_PY" "$TASK/scripts/upload_baidu_xpan.py" \
 --local "$TASK/review/$NAME.zip" --remote "/apps/bypy/lance_review/$NAME.zip" \
 --state "$TASK/review/$NAME.baidu_upload.json" \
 --bypy-config /scratch/prj0000000275/jcy/ljy/video_collection/dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json
"$H3_PY" - "$TASK" "$NAME" <<'PY'
from pathlib import Path
import json,sys,time
root=Path(sys.argv[1]);name=sys.argv[2]
state=json.loads((root/'review'/(name+'.baidu_upload.json')).read_text());assert state['status']=='complete'
package=json.loads((root/'metadata'/('package_'+name+'.json')).read_text())
receipt={'status':'complete','remote':'/apps/bypy/lance_review/'+name+'.zip','model_counts':package['model_counts'],'outputs':package['outputs'],'samples':80,'only_zip_uploaded':True,'extra_remote_verification':False,'completed_unix':time.time()}
(root/'metadata'/('delivery_'+name+'.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
with (root/'README.md').open('a') as f:f.write('\n已完成结果快照上传成功：'+receipt['remote']+'；数量'+str(receipt['model_counts'])+'。仅ZIP，以上传器complete为准，未额外远端复核。\n')
print('COMPLETED_DELIVERY_SUCCESS',json.dumps(receipt,ensure_ascii=False),flush=True)
PY
