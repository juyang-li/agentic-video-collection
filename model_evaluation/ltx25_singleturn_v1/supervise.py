"""CPU-only supervisor: wait for allocation, smoke, sbatch and bounded resume."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parent

def save(path,value):
    tmp=path.with_suffix(path.suffix+'.part');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def cmd(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=30)
    return p.returncode,p.stdout.strip(),p.stderr.strip()
def job_state(job):
    code,out,err=cmd(['squeue','-h','-j',str(job),'-o','%T'])
    if code==0 and out:return out.splitlines()[0].strip()
    code,out,err=cmd(['sacct','-X','-n','-P','-j',str(job),'--format=State'])
    if code==0 and out:return out.splitlines()[0].split('|')[0].split('+')[0].strip()
    return 'UNKNOWN'
def completion():
    originals=[json.loads(s) for s in (ROOT/'inputs.jsonl').read_text().splitlines()]
    ids=[r['sample_id'] for r in originals]+[r['sample_id'] for r in json.loads((ROOT/'reference_controls.json').read_text())]
    good=[];bad=[];exhausted=[]
    for sid in ids:
        p=ROOT/f'outputs/records/{sid}.json';r=json.loads(p.read_text()) if p.exists() else {}
        if r.get('status') in ['generated','recovered_tmp'] and Path(r.get('output','')).is_file():good.append(sid)
        else:
            bad.append(sid)
            if r.get('status')=='failed' and r.get('attempts',0)>=2:exhausted.append(sid)
    return good,bad,exhausted

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allocation',type=int,required=True);a=parser.parse_args()
    (ROOT/'metadata').mkdir(exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
    lock=(ROOT/'metadata/supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy();env['LTX_BENCH_ROOT']=str(ROOT)
    state=dict(supervisor_pid=os.getpid(),allocation=a.allocation,phase='preparing',created_at=time.time(),formal_jobs=[])
    status=ROOT/'metadata/supervisor_status.json'
    def tick(**values):
        state.update(values,heartbeat=time.time());save(status,state)
    tick()
    # These files are emitted only after successful dependency installation / model download.
    while not ((ROOT/'metadata/ltx_nodes_revision.txt').exists() and (ROOT/'metadata/model_lock.json').exists()):
        tick(phase='preparing',allocation_state=job_state(a.allocation))
        if time.time()-state['created_at']>7200:
            tick(phase='preparation_needs_attention');return
        time.sleep(60)
    model_rows=json.loads((ROOT/'metadata/model_lock.json').read_text())
    if not all(Path(r['local']).is_file() and Path(r['local']).stat().st_size==r['bytes'] for r in model_rows):
        tick(phase='model_files_need_attention');return
    expected=json.loads((ROOT/'inventory_summary.json').read_text())['input_manifest_sha256']
    if hashlib.sha256((ROOT/'inputs.jsonl').read_bytes()).hexdigest()!=expected:
        tick(phase='frozen_input_changed');return
    while True:
        js=job_state(a.allocation);tick(phase='waiting_smoke_allocation',allocation_state=js)
        if js=='RUNNING':break
        if js in ['CANCELLED','TIMEOUT','FAILED','COMPLETED','NODE_FAIL','OUT_OF_MEMORY']:
            tick(phase='allocation_ended_before_smoke');return
        time.sleep(60)
    logfile=(ROOT/f'logs/srun_smoke_{a.allocation}.log').open('a')
    child=subprocess.Popen(['srun','--jobid='+str(a.allocation),'--exact','--ntasks=1','--cpus-per-task=14',
                            '--gres=gpu:h200:1','bash',str(ROOT/'smoke_step.sh')],env=env,stdout=logfile,stderr=subprocess.STDOUT)
    while child.poll() is None:
        tick(phase='srun_smoke',smoke_launcher_pid=child.pid);time.sleep(30)
    logfile.close()
    smoke=ROOT/'metadata/smoke_status.json'
    evidence=json.loads(smoke.read_text()) if smoke.exists() else {}
    if child.returncode!=0 or evidence.get('phase')!='complete':
        tick(phase='smoke_needs_attention',smoke_exit=child.returncode);return
    tick(phase='smoke_passed')
    stall_seconds=max(900,3*evidence.get('max_sample_seconds',0))
    for attempt in range(4):
        good,bad,exhausted=completion()
        if not bad:tick(phase='complete',completed=64,total=64);return
        if exhausted:tick(phase='failed_items_need_attention',completed=len(good),failed=exhausted);return
        result=subprocess.run(['sbatch','--parsable','--export=ALL,LTX_BENCH_ROOT='+str(ROOT),
             '--output='+str(ROOT/'logs/infer_%j.log'),'--error='+str(ROOT/'logs/infer_%j.log'),str(ROOT/'infer.sbatch')],
             env=env,capture_output=True,text=True)
        if result.returncode:tick(phase='sbatch_submission_failed',error=result.stderr);return
        job=int(result.stdout.strip().split(';')[0]);state['formal_jobs'].append(job)
        tick(phase='formal_submitted',active_job=job,completed=len(good),total=64)
        # This allocation was created solely for this smoke; release it after submission.
        if attempt==0:cmd(['scancel',str(a.allocation)])
        while True:
            js=job_state(job);good,bad,exhausted=completion()
            tick(phase='formal_running' if js=='RUNNING' else 'formal_waiting',job_state=js,completed=len(good),remaining=bad)
            if not bad:
                # Let the batch close its servers normally before marking the run finished.
                if js not in ['RUNNING','COMPLETING']:tick(phase='complete',completed=64,total=64);return
            if js in ['COMPLETED','CANCELLED','FAILED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL']:break
            if js=='RUNNING':
                progress=[]
                for shard in range(4):
                    p=ROOT/f'metadata/shard{shard}_status.json'
                    if p.exists():
                        r=json.loads(p.read_text())
                        if str(r.get('job_id'))==str(job) and r.get('phase') not in ['complete','failed_items']:
                            progress.append(r.get('last_progress',r.get('heartbeat',time.time())))
                if progress and min(time.time()-t for t in progress)>stall_seconds:
                    tick(phase='stalled_restarting',active_job=job);cmd(['scancel',str(job)]);break
            time.sleep(60)
        tick(phase='resuming_missing',active_job=job,completed=len(good),remaining=bad)
        time.sleep(30)
    tick(phase='resume_limit_reached')

if __name__=='__main__':main()
