#!/usr/bin/env bash
set -euo pipefail
extra=()
if [[ "${A100_ALLOW_LOGIN_PACKAGE:-0}" == 1 ]]; then
  extra=(--allow-login)
else
  [[ "${SLURM_JOB_PARTITION:-}" == testqueue ]] || exit 64
fi
TASK=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/ltx25_vace_a100_20261111
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
# Serialize login/allocation delivery and reuse the already delivered ZIP.
exec 9>"$TASK/metadata/input_delivery.lock"
flock 9
if "$NATIVE/runtime/envs/trainer/bin/python" - "$TASK" <<'PYSTATE'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);p=root/'review/a100_inputs_scroll_20261111.baidu_upload.json'
sys.exit(0 if p.exists() and json.loads(p.read_text()).get('status')=='complete' else 1)
PYSTATE
then exit 0; fi
if [[ ! -f "$TASK/metadata/package_inputs.json" || ! -f "$TASK/review/a100_inputs_scroll_20261111.zip" ]]; then
  bash "$NATIVE/scripts/common/run_trainer_python.sh" "$TASK/scripts/package.py" --kind inputs "${extra[@]}"
fi
bash "$NATIVE/scripts/common/run_trainer_python.sh" /scratch/prj0000000275/jcy/ljy/video_collection/lance_general_edit_finetune/1_specialist_filtering/scripts/upload_baidu_xpan.py \
  --local "$TASK/review/a100_inputs_scroll_20261111.zip" \
  --remote /apps/bypy/lance_review/a100_inputs_scroll_20261111.zip \
  --state "$TASK/review/a100_inputs_scroll_20261111.baidu_upload.json" \
  --bypy-config "${A100_BAIDU_CONFIG:-/scratch/prj0000000275/jcy/ljy/video_collection/dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json}"
