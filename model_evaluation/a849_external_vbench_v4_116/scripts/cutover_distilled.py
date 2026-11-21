"""Drain the frozen in-flight cases, replace only our steps, retain allocation202896."""
import subprocess,time,traceback,shutil
from common import *
EXT=ROOT.parent/'a849_v4_extensions16_20261115'
PILOT=ROOT.parent/'a849_tripvvt_pilot32_20261115'
PY=ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace/bin/python'
def main():
    assert not os.environ.get('SLURM_JOB_ID')
    state=read(ROOT/'metadata/vace_cutover.json');state.update(coordinator_pid=os.getpid())
    def update(**kw):state.update(kw,updated_unix=time.time());atomic(ROOT/'metadata/vace_cutover.json',state)
    update(status='draining')
    while not all((EXT/'outputs/vace'/f'{sid}.json').exists()and read(EXT/'outputs/vace'/f'{sid}.json')['status']=='complete'for sid in state['inflight']):time.sleep(5)
    # Specific step only, never the parent allocation.
    assert state['old_step']=='202896.6'
    subprocess.run(['scancel',state['old_step']],check=True)
    while state['old_step'] in subprocess.check_output(['squeue','--steps','-j','202896','-h','-o','%i'],text=True).split():time.sleep(2)
    update(status='freezing_profiles')
    for root,ids in [(EXT,[r['id']for r in rows(EXT/'metadata/inputs.jsonl')]),(PILOT,sorted(read(PILOT/'metadata/selection_decision.json')['selected']))]:
        profiles={}
        for sid in ids:
            path=root/'outputs/vace'/f'{sid}.json';old=read(path)if path.exists()else{}
            if old.get('status')=='complete':
                assert old['contract']['steps']==40 and sha(path.with_suffix('.mp4'))==old['output_sha256'];profiles[sid]='unipc40'
            else:
                profiles[sid]='lightx2v4'
                for candidate in [path,path.with_suffix('.mp4'),path.with_suffix('.part.mp4')]:
                    if candidate.exists():
                        dst=root/'metadata/cutover_attempts'/candidate.name;dst.parent.mkdir(exist_ok=True);assert not dst.exists();shutil.move(str(candidate),dst)
        atomic(root/'metadata/vace_inference_plan.json',dict(status='frozen',profiles=profiles,input_manifest_sha256=sha(root/'metadata/inputs.jsonl'),created_unix=time.time()))
    update(status='running_distilled')
    cmd=['srun','--jobid=202896','--partition=testqueue','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=56','--gres=gpu:4','bash',str(ROOT/'scripts/resume_distilled.sh')]
    with (ROOT/'logs/resumed_generation.log').open('a',buffering=1)as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    update(gpu_launcher_pid=p.pid)
    assert p.wait()==0,'Resumed GPU pipeline failed; preserve outputs, inspect logs'
    update(status='waiting_h3_derivation')
    while read(EXT/'metadata/h3_derivation.json').get('status')!='complete':time.sleep(15)
    atomic(ROOT/'metadata/resumed_generation.json',dict(status='complete',completed_unix=time.time()))
    update(status='generation_complete')
    with (ROOT/'logs/finalize_distilled.log').open('a',buffering=1)as log:p=subprocess.Popen([str(PY),str(ROOT/'scripts/finalize_distilled.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    update(finalizer_pid=p.pid)
if __name__=='__main__':
    try:main()
    except BaseException:atomic(ROOT/'metadata/cutover_failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
