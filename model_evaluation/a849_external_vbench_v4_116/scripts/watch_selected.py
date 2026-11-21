"""CPU contact sheets for the selected eight's remaining model outputs."""
import subprocess,time
from common import *
PILOT=ROOT.parent/'a849_tripvvt_pilot32_20261115'
PY=ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace/bin/python'
ids=read(PILOT/'metadata/selection_decision.json')['selected']
while True:
    for model in ['ltx_base','vace']:subprocess.run([str(PY),str(PILOT/'scripts/visual.py'),'contacts','--model',model],check=True)
    if all((PILOT/'outputs'/m/f'{sid}.json').exists()and read(PILOT/'outputs'/m/f'{sid}.json').get('status')=='complete'for sid in ids for m in ['ltx_base','vace']):break
    time.sleep(60)
