"""CPU contacts as outputs arrive; visual.py serializes observation-file access."""
import subprocess,time
from common import *
from score import MODELS

def main():
    assert not os.environ.get('SLURM_JOB_ID')
    py=ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace/bin/python'
    while True:
        complete=True
        for model in MODELS:
            subprocess.run([str(py),str(ROOT/'scripts/visual.py'),'contacts','--model',model],check=True)
            complete &= all((ROOT/'outputs'/model/f"{r['id']}.json").exists()and read(ROOT/'outputs'/model/f"{r['id']}.json")['status']=='complete'for r in rows(ROOT/'metadata/inputs.jsonl'))
        if complete:break
        time.sleep(60)
    atomic(ROOT/'metadata/output_contacts.json',dict(status='complete',models=list(MODELS),tasks=len(rows(ROOT/'metadata/inputs.jsonl'))))

if __name__=='__main__':main()
