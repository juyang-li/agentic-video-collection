"""Wait on the user-provided allocation; sleep thirty minutes between checks."""
from pathlib import Path
import subprocess,time,json,fcntl
root=Path(__file__).resolve().parents[1];lock=(root/'logs/queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (root/'metadata/protocol_rejection.json').exists():raise SystemExit('Rejected IC-LoRA input protocol; allocation watcher disabled')
job='199975';submitted=False
while True:
 states=list((root/'outputs').glob('*.json'))
 if len(states)==80 and all(json.loads(p.read_text()).get('status')=='complete' for p in states):
  print('All 80 IC-LoRA outputs complete; stopping allocation watcher',flush=True);break
 p=subprocess.run(['squeue','-h','-j',job,'-o','%T'],capture_output=True,text=True)
 if p.returncode and 'Invalid job id' not in p.stderr:
  print('Transient scheduler failure; retain handle',job,flush=True);time.sleep(1800);continue
 state=p.stdout.strip()
 print(time.strftime('%F %T'),job,state or 'terminal/missing',flush=True)
 if state=='RUNNING':
  if not all((root/'metadata'/n).exists() for n in ['preflight.json','pipeline_preflight.json']):time.sleep(60);continue
  if submitted:
   time.sleep(1800);continue
  with (root/'logs/driver.log').open('a') as f:
   code=subprocess.run(['srun','--jobid='+job,'--partition=testqueue','--overlap','--ntasks=1','--cpus-per-task=1','bash',str(root/'scripts/run.sh')],stdout=f,stderr=subprocess.STDOUT).returncode
  if code==0:
   print('IC-LoRA generation complete',flush=True);break
  check=subprocess.run(['squeue','-h','-j',job,'-o','%T'],capture_output=True,text=True)
  if check.returncode:print('Scheduler observation failed after exit; inspect before retry',flush=True);break
  if check.stdout.strip()=='RUNNING':raise RuntimeError('Live allocation inference failed; inspect driver log before resuming')
 elif state in ['PENDING','CONFIGURING','COMPLETING']:
  time.sleep(1800);continue
 else:
  # User authorized testqueue-only sbatch if the provided allocation is unavailable.
  p=subprocess.run(['sbatch','--parsable','--partition=testqueue','--gres=gpu:h200:4','--ntasks=1','--cpus-per-task=16','--mem=900G','--time=18:00:00','--job-name=full10s_iclora','--output='+str(root/'logs/slurm_%j.log'),str(root/'scripts/run.sh')],capture_output=True,text=True)
  if p.returncode:print('sbatch failed',p.stderr,flush=True);time.sleep(1800);continue
  job=p.stdout.strip().split(';')[0];submitted=True
  (root/'metadata/allocation.json').write_text(json.dumps({'job_id':job,'partition':'testqueue','submitted_unix':time.time()}));time.sleep(1800)
