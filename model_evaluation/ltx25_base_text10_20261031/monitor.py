from pathlib import Path
import json,time,subprocess,os
ROOT=Path(__file__).resolve().parent
JOB='183476'
def run(cmd):return subprocess.run(cmd,text=True,capture_output=True,timeout=40)
def save(d):
 p=ROOT/'metadata/monitor_status.json';tmp=p.with_suffix('.part');tmp.write_text(json.dumps({**d,'updated_at':time.time()},indent=2)+'\n');tmp.replace(p)
def state():
 p=run(['squeue','-h','-j',JOB,'-o','%T']);return p.stdout.strip() if p.returncode==0 else 'UNKNOWN'
def count():
 n=0
 for p in (ROOT/'outputs/records').glob('*.json'):
  d=json.loads(p.read_text())
  if d.get('status') in ['generated','recovered_tmp'] and Path(d['output']).is_file():n+=1
 return n
save(dict(phase='waiting_smoke',job=JOB))
while True:
 p=ROOT/'metadata/smoke_status.json';s=json.loads(p.read_text()) if p.exists() else {}
 if s.get('phase')=='complete':break
 if s.get('phase') in ['crashed','failed_items']:
  save(dict(phase='smoke_failed',job=JOB,details=s));raise SystemExit(1)
 js=state()
 if js not in ['RUNNING','COMPLETING','UNKNOWN']:
  save(dict(phase='allocation_not_running',job=JOB,job_state=js));raise SystemExit(1)
 time.sleep(15)
# Run remaining examples on the allocated cards after the smoke Comfy process closes.
time.sleep(10)
env=os.environ.copy();env['LTX_BENCH_ROOT']=str(ROOT)
# Each worker stages once on-node; use same job-local path as smoke. No login staging.
command=['srun','--jobid='+JOB,'--exact','--ntasks=2','--cpus-per-task=14','--gpus-per-task=1','--gpu-bind=single:1',
 'bash','-c','source "$LTX_BENCH_ROOT/stage_job.sh"; exec "$LTX_BENCH_ROOT/runtime/venv/bin/python" "$LTX_BENCH_ROOT/run_worker.py" --shard "$SLURM_PROCID"']
# Smoke already staged all immutable model files, so parallel workers just check sizes.
with (ROOT/'logs/formal.log').open('a') as f:
 child=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT)
 while child.poll() is None:
  save(dict(phase='formal_running',job=JOB,completed=count(),total=10,launcher_pid=child.pid));time.sleep(15)
code=child.returncode;n=count();save(dict(phase='complete' if n==10 else 'needs_recovery',job=JOB,completed=n,total=10,exit_code=code))
if n==10:run(['scancel',JOB])
print(json.dumps(dict(completed=n,total=10,exit_code=code)),flush=True)
