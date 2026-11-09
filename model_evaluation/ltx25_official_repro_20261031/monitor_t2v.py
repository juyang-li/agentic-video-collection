from pathlib import Path
import os,time,subprocess,json
ROOT=Path(__file__).resolve().parent;JOB='183496'
def save(**values):
 p=ROOT/'metadata/monitor_status.json';tmp=p.with_suffix('.part');tmp.write_text(json.dumps(dict(job=JOB,pid=os.getpid(),updated_at=time.time(),**values),indent=2)+'\n');tmp.replace(p)
def call(args):return subprocess.run(args,capture_output=True,text=True,timeout=40)
def jobstate():
 p=call(['squeue','-h','-j',JOB,'-o','%T'])
 if p.returncode==0 and p.stdout.strip():return p.stdout.strip()
 p=call(['sacct','-X','-n','-P','-j',JOB,'-o','State'])
 return p.stdout.strip().split('|')[0] if p.returncode==0 and p.stdout.strip() else 'UNKNOWN'
while True:
 js=jobstate();save(phase='waiting_allocation',job_state=js)
 if js=='RUNNING':break
 if js.split()[0] in ['CANCELLED','TIMEOUT','FAILED','COMPLETED','NODE_FAIL']:
  save(phase='allocation_terminal',job_state=js);raise SystemExit(1)
 time.sleep(30)
env=os.environ.copy();env['LTX_BENCH_ROOT']=str(ROOT)
for phase,extra in [('smoke',['--smoke']),('formal',['--shard','0'])]:
 cmd=['srun','--jobid='+JOB,'--exact','--ntasks=1','--cpus-per-task=14','--gres=gpu:h200:1','bash','-c',
      'source "$LTX_BENCH_ROOT/stage_job.sh"; exec "$LTX_BENCH_ROOT/runtime/venv/bin/python" "$LTX_BENCH_ROOT/run_worker.py" '+ ' '.join(extra)]
 with (ROOT/f'logs/{phase}.log').open('a') as f:
  child=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  while child.poll() is None:save(phase=phase,launcher_pid=child.pid);time.sleep(15)
 if child.returncode:
  save(phase=phase+'_failed',exit_code=child.returncode);raise SystemExit(1)
records=[json.loads(p.read_text()) for p in (ROOT/'outputs/records').glob('*.json')]
good=[d for d in records if d.get('status') in ['generated','recovered_tmp'] and Path(d['output']).is_file()]
if len(good)==2:
 call(['scancel',JOB]);save(phase='complete',completed=2,total=2,exit_code=0,allocation_released=True)
else:save(phase='incomplete',completed=len(good),total=2)
print(json.dumps(dict(completed=len(good),total=2)),flush=True)
