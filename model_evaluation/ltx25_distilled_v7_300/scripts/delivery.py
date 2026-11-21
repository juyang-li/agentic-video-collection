"""Detached login controller: wait for GPU job exit, package, then authorized upload."""
from common import *
import subprocess,time,fcntl,traceback

def main():
    assert not os.environ.get('SLURM_JOB_ID')
    lock=(ROOT/'logs/delivery.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=read(ROOT/'metadata/submission.json')['job_id'];status=ROOT/'metadata/delivery_status.json'
    atomic(status,{'status':'waiting_for_gpu_job','job_id':job,'pid':os.getpid(),'poll_interval_seconds':60})
    while True:
        q=subprocess.run(['squeue','-h','-j',job,'-o','%P|%T'],capture_output=True,text=True,timeout=30)
        if q.returncode and 'invalid job id' not in q.stderr.lower():
            print('Scheduler temporarily unavailable; no resubmission',flush=True);time.sleep(60);continue
        if not q.stdout.strip():break
        for row in q.stdout.splitlines():assert row.split('|')[0]=='testqueue'
        time.sleep(60)
    # Slurm no longer owns GPUs; handle hard timeout/node failure even when the driver could not finalize.
    items=rows(ROOT/'metadata/inputs.jsonl');complete=0
    for item in items:
        p=ROOT/f'outputs/{item["id"]}.json';old=read(p) if p.exists() else {}
        if old.get('status')=='complete':complete+=1
        elif old.get('status')!='failed':atomic(p,{**old,'id':item['id'],'status':'failed' if old else 'not_run','failure_reason':'GPU job exited before completion; inspect Slurm/worker logs'})
    if complete!=300:atomic(ROOT/'metadata/run_status.json',{'status':'failed','completed':complete,'total':300,'job_id':job})
    atomic(status,{'status':'packaging','job_id':job,'completed':complete})
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    with (ROOT/'logs/package.log').open('ab',buffering=0) as log:
        subprocess.run(['bash',str(NATIVE/'scripts/common/run_trainer_python.sh'),str(ROOT/'scripts/package.py')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    if complete!=300:
        atomic(status,{'status':'partial_local_only','completed':complete,'total':300,'uploaded':False});return
    atomic(status,{'status':'uploading','remote':REMOTE,'completed':300})
    for key in ['http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','PYTHONPATH']:env.pop(key,None)
    env['CUDA_VISIBLE_DEVICES']='';env['PYTHONUNBUFFERED']='1'
    py=ROOT.parent/'minimax_h3_ref2va90_local_20261112/runtime/env/bin/python'
    with (ROOT/'logs/upload.log').open('ab',buffering=0) as log:
        subprocess.run([str(py),str(ROOT/'scripts/upload_review.py')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    atomic(status,{'status':'complete','completed':300,'remote':REMOTE,'only_zip_uploaded':True,'completed_unix':time.time()})
if __name__=='__main__':
    try:main()
    except Exception:
        atomic(ROOT/'metadata/delivery_status.json',{'status':'failed','error':traceback.format_exc(),'failed_unix':time.time()});raise
