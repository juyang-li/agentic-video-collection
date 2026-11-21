"""Persistent login-side orchestration; GPU phases attach to existing202896 only."""
import subprocess,time,traceback
from common import *

PY=ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace/bin/python'

def main():
    assert not os.environ.get('SLURM_JOB_ID'),'CPU supervisor must run on login'
    state=dict(status='waiting_cpu_media',pid=os.getpid(),started_unix=time.time(),job_id=202896)
    def save(**kwargs):state.update(kwargs);atomic(ROOT/'metadata/supervisor.json',state)
    save()
    items=rows(ROOT/'metadata/inputs.jsonl');deadline=time.monotonic()+7200
    while not all((ROOT/'inputs/prepared'/r['id']/'pixels.json').is_file()for r in items):
        assert time.monotonic()<deadline,'CPU media preparation timed out; inspect shard logs'
        time.sleep(10)
    cmd=['srun','--jobid=202896','--partition=testqueue','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=56','--gres=gpu:4','bash',str(ROOT/'scripts/gpu_pipeline.sh')]
    with (ROOT/'logs/gpu_pipeline.log').open('a',buffering=1)as log:
        gpu=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(status='gpu_preparation',gpu_launcher_pid=gpu.pid,command=cmd)
        with (ROOT/'logs/h3_derive.log').open('a',buffering=1)as dlog:
            derive=subprocess.Popen([str(PY),str(ROOT/'scripts/h3_derive.py'),'--loop','--interval','30'],stdout=dlog,stderr=subprocess.STDOUT,start_new_session=True)
        save(derive_pid=derive.pid)
        while gpu.poll()is None:
            if (ROOT/'runtime/masks_ready').exists()and not (ROOT/'runtime/mask_contacts_ready').exists():
                subprocess.run([str(PY),str(ROOT/'scripts/contacts.py'),'--mode','masks'],check=True)
                (ROOT/'runtime/mask_contacts_ready').touch();save(status='waiting_direct_input_review')
            if (ROOT/'metadata/generation_gate.json').exists()and not (ROOT/'metadata/h3_preflight.json').exists():
                save(status='h3_cpu_preparation')
                with (ROOT/'logs/h3_prepare.log').open('a',buffering=1)as hlog:
                    subprocess.run([str(PY),str(ROOT/'scripts/h3_prepare.py')],stdout=hlog,stderr=subprocess.STDOUT,check=True)
                save(status='generating')
            time.sleep(15)
        assert gpu.returncode==0, f'GPU step exited {gpu.returncode}; inspect logs/gpu_pipeline.log'
    save(status='waiting_direct_output_review',gpu_returncode=0,gpu_completed_unix=time.time())
    print('All GPU generation complete; all32 outcomes must be directly reviewed before selecting8.',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'metadata/supervisor_failure.json',dict(status='failed',error=traceback.format_exc(),time=time.time()));raise
