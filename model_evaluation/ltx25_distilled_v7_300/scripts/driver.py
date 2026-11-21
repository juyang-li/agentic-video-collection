"""Own all four workers; terminate peers immediately if any worker fails."""
from common import *
import subprocess,time,signal,fcntl

def main():
    assert os.environ.get('SLURM_JOB_PARTITION')=='testqueue' and os.environ.get('SLURM_JOB_ID')
    verify_implementation()
    lock=(ROOT/'logs/driver.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'metadata/stop.json').exists(),'Inspect and authorize continuation before clearing stop marker'
    assert not list((ROOT/'outputs').glob('*.json')),'Initial batch requires fresh output directory'
    gpuids=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(gpuids)==len(set(gpuids))==4
    atomic(ROOT/'metadata/run_status.json',{'status':'running','job_id':os.environ['SLURM_JOB_ID'],'started_unix':time.time(),'total':300,'pilot_reused':0})
    children=[];logs=[]
    def stop(signum,frame):
        atomic(ROOT/'metadata/stop.json',{'reason':'Driver interrupted','signal':signum})
        for p in children:
            if p.poll() is None:p.terminate()
        raise RuntimeError('Driver interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for shard,gpu in enumerate(gpuids):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            log=(ROOT/f'logs/worker_{os.environ["SLURM_JOB_ID"]}_{shard}.log').open('ab',buffering=0);logs.append(log)
            children.append(subprocess.Popen(['bash',str(ROOT/'scripts/run_worker.sh'),str(shard)],env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Worker failure; stopping peers')
            time.sleep(1)
        assert all(p.returncode==0 for p in children)
        items=rows(ROOT/'metadata/inputs.jsonl');assert all(read(ROOT/f'outputs/{r["id"]}.json')['status']=='complete' for r in items)
        atomic(ROOT/'metadata/run_status.json',{'status':'complete','job_id':os.environ['SLURM_JOB_ID'],'completed_unix':time.time(),'completed':300,'pilot_reused':0})
    except BaseException as exc:
        if not (ROOT/'metadata/stop.json').exists():atomic(ROOT/'metadata/stop.json',{'reason':str(exc)})
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        count=0
        for r in rows(ROOT/'metadata/inputs.jsonl'):
            path=ROOT/f'outputs/{r["id"]}.json';old=read(path) if path.exists() else {}
            if old.get('status')=='complete':count+=1
            elif old.get('status')!='failed':atomic(path,{**old,'id':r['id'],'status':'failed' if old else 'not_run','failure_reason':'Batch stopped after technical failure; see metadata/stop.json'})
        atomic(ROOT/'metadata/run_status.json',{'status':'failed','completed':count,'total':300,'reason':str(exc),'completed_unix':time.time()})
        raise
    finally:
        for log in logs:log.close()
if __name__=='__main__':main()
