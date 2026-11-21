#!/usr/bin/env bash
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == testqueue && ${SLURM_JOB_ID:-} == 202896 ]] || exit 64
BASE=/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation
EXT=$BASE/a849_v4_extensions16_20261115
PILOT=$BASE/a849_tripvvt_pilot32_20261115
FINAL=$BASE/a849_external_vbench_v4_116
VACE=$BASE/ltx25_vace_a100_20261111/runtime/vace/bin/python
H3=$BASE/minimax_h3_ref2va90_local_20261112/runtime/env/bin/python
NATIVE=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$NATIVE/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec 9>"$PILOT/runtime/job202896.lock"
flock 9
IFS=',' read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:?}"
[[ ${#devices[@]} == 4 ]] || exit 65
four() {
 local root=$1 label=$2;shift 2;local pids=() failed=0
 for shard in 0 1 2 3;do CUDA_VISIBLE_DEVICES="${devices[$shard]}" "$@" --shard "$shard" > "$root/logs/${label}_${shard}.log" 2>&1 & pids+=("$!");done
 for pid in "${pids[@]}";do wait "$pid" || failed=1;done
 [[ $failed == 0 ]]
}
CUDA_VISIBLE_DEVICES="${devices[0]}" VACE_SMOKE=1 "$VACE" "$EXT/scripts/dispatch_worker.py" --model vace --shard 0 > "$EXT/logs/distilled_smoke.log" 2>&1
"$VACE" - "$EXT" <<'PY'
import sys,json
from pathlib import Path
r=Path(sys.argv[1]);plan=json.loads((r/'metadata/vace_inference_plan.json').read_text());ids=[s for s,p in plan['profiles'].items()if p=='lightx2v4']
if ids:
 x=json.loads((r/'outputs/vace'/f'{ids[0]}.json').read_text());assert x['status']=='complete'and x['contract']['steps']==4 and x['lora_load']['all_loaded_values_checked']
print('Distilled smoke passed')
PY
four "$EXT" distilled_vace "$VACE" "$EXT/scripts/dispatch_worker.py" --model vace
"$VACE" - "$EXT" <<'PY'
import sys,json
from pathlib import Path
r=Path(sys.argv[1]);items=[json.loads(l)for l in(r/'metadata/inputs.jsonl').read_text().splitlines()]
assert all(json.loads((r/'outputs/vace'/f"{x['id']}.json").read_text())['status']=='complete'for x in items)
PY
"$H3" "$EXT/scripts/h3_worker.py" --stage all > "$EXT/logs/h3_worker.log" 2>&1
four "$PILOT" selected_base bash "$NATIVE/scripts/common/run_trainer_python.sh" "$PILOT/scripts/dispatch_worker.py" --model ltx_base
four "$PILOT" selected_vace "$VACE" "$PILOT/scripts/dispatch_worker.py" --model vace
"$VACE" - "$EXT" "$PILOT" <<'PY'
import sys,json,time
from pathlib import Path
ext,pilot=map(Path,sys.argv[1:]);items=[json.loads(l)for l in(ext/'metadata/inputs.jsonl').read_text().splitlines()];selected=json.loads((pilot/'metadata/selection_decision.json').read_text())['selected']
assert all(json.loads((ext/'outputs'/m/f"{x['id']}.json").read_text())['status']=='complete'for x in items for m in ['ltx_base','a849','vace','h3_pdd8_native'])
assert all(json.loads((pilot/'outputs'/m/f'{sid}.json').read_text())['status']=='complete'for sid in selected for m in ['ltx_base','vace'])
(ext/'metadata/gpu_generation.json').write_text(json.dumps(dict(status='complete',expected_outputs=64,completed_unix=time.time())))
PY
