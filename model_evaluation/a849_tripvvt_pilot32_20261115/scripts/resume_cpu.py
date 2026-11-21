"""Resume CPU coordination of an existing GPU step without launching another."""
import subprocess,time
from common import *
PY=ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace/bin/python'

def main():
    assert not os.environ.get('SLURM_JOB_ID')
    items=rows(ROOT/'metadata/inputs.jsonl')
    while not (ROOT/'metadata/gpu_generation.json').exists():
        paths=[ROOT/'conditions'/r['id']/'condition.json'for r in items]
        if all(p.exists()and read(p)['status']=='complete'for p in paths)and not (ROOT/'runtime/mask_contacts_ready').exists():
            subprocess.run([str(PY),str(ROOT/'scripts/contacts.py'),'--mode','masks'],check=True);(ROOT/'runtime/mask_contacts_ready').touch()
        if (ROOT/'metadata/generation_gate.json').exists()and not (ROOT/'metadata/h3_preflight.json').exists():
            with (ROOT/'logs/h3_prepare.log').open('a')as log:subprocess.run([str(PY),str(ROOT/'scripts/h3_prepare.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
        atomic(ROOT/'metadata/cpu_coordination.json',dict(status='generating'if (ROOT/'metadata/h3_preflight.json').exists()else'waiting_direct_input_review',pid=os.getpid(),updated_unix=time.time()))
        time.sleep(15)
    atomic(ROOT/'metadata/cpu_coordination.json',dict(status='waiting_direct_output_review',pid=os.getpid(),updated_unix=time.time()))

if __name__=='__main__':main()
