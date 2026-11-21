"""Sleep between checks; resume only after confirmed allocation termination."""
import fcntl,json,subprocess,time,os,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['ltx','vace','h3'],default='ltx');args=parser.parse_args()
role=args.role
log=ROOT/'logs'/f'allocation_watch_{role}.log'
def note(s):
 with log.open('a') as f:f.write(time.strftime('%F %T')+' '+s+'\n')
def state(job):
 p=subprocess.run(['squeue','-h','-j',str(job),'-o','%T'],capture_output=True,text=True)
 if p.returncode:
  if 'Invalid job id' in p.stderr:return ''
  return 'UNKNOWN'
 return p.stdout.strip()
def allocation_available(job):
 with (ROOT/'logs'/f'allocation_{job}.lock').open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return True
  except BlockingIOError:return False
def locked():
 if role=='h3':
  paths=[ROOT/'h3/logs/driver.lock',ROOT/'iclora/logs/driver.lock']
  for p in paths:
   with p.open('a') as h:
    try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return True
  return False
 with (ROOT/'logs'/('driver.lock' if role=='ltx' else 'driver.lock')).open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
  except BlockingIOError:return True
def done():
 inputs=ROOT/'metadata/inputs.jsonl'
 if not inputs.exists():return False
 ids=[json.loads(l)['id'] for l in inputs.read_text().splitlines()]
 for model in (('ltx_base','a849') if role=='ltx' else (role,)):
  for sid in ids:
   p=(ROOT/'h3/outputs' if role=='h3' else ROOT/'outputs'/model)/(sid+'.json')
   if not p.exists() or json.loads(p.read_text()).get('status')!='complete':return False
 return bool(ids)
watch=(ROOT/'logs'/f'allocation_watch_{role}.lock').open('a')
fcntl.flock(watch,fcntl.LOCK_EX|fcntl.LOCK_NB)
current='199975' if role=='h3' else '199810'
entry=ROOT/'scripts'/{'ltx':'run.sh','vace':'vace_after_ltx.sh','h3':'h3_direct_resume.sh'}[role]
note('Watching current allocation; sleep=1800 seconds; fallback=199975 then sbatch testqueue gpu:h200:4')
while True:
 if role=='vace':
  prior_complete=True
  for model in ['a849','ltx_base']:
   for item in [json.loads(l) for l in (ROOT/'metadata/inputs.jsonl').read_text().splitlines()]:
    p=ROOT/'outputs'/model/(item['id']+'.json')
    if not p.exists() or json.loads(p.read_text()).get('status')!='complete':prior_complete=False;break
   if not prior_complete:break
  if not prior_complete:time.sleep(1800);continue
 if done():note('LTX2.5/A849 complete');break
 status=state(current)
 if status=='UNKNOWN':note('Scheduler observation failed; retain current handle');time.sleep(1800);continue
 if locked() or status in ('RUNNING','PENDING','CONFIGURING','COMPLETING'):
  if not locked() and status=='RUNNING':
   note('Driver exited while allocation remains live; inspect inference logs rather than restart blindly');break
  time.sleep(1800);continue
 note('Allocation '+current+' terminal/missing; trying 199975')
 fallback=state('199975')
 if fallback=='UNKNOWN':time.sleep(1800);continue
 if fallback=='RUNNING' and allocation_available('199975'):
  current='199975'
  with (ROOT/'logs/resume_199975.log').open('a') as f:
   p=subprocess.run(['srun','--jobid=199975','--partition=testqueue','--overlap','--ntasks=1','--cpus-per-task=16' if role=='h3' else '--cpus-per-task=8','bash',str(entry)],stdout=f,stderr=subprocess.STDOUT)
  if p.returncode==0:continue
  note('199975 attach/run returned '+str(p.returncode)+'; inspecting terminal state')
  if state(current)=='RUNNING':note('Live allocation run failed; inspect logs');break
 else:
  p=subprocess.run(['sbatch','--parsable','--account=prj0000000275','--partition=testqueue','--gres=gpu:h200:4','--ntasks=1','--cpus-per-task=16','--mem=900G','--time=18:00:00','--job-name=full10s_'+role,'--output='+str(ROOT/'logs/slurm_%j.log'),str(entry)],capture_output=True,text=True)
  if p.returncode:
   note('sbatch failed: '+p.stderr.strip());time.sleep(1800);continue
  current=p.stdout.strip().split(';')[0]
  (ROOT/'metadata'/f'fallback_allocation_{role}.json').write_text(json.dumps({'jobid':current,'partition':'testqueue','submitted_unix':time.time()}))
  note('Submitted '+current+'; waiting in 1800s intervals');time.sleep(1800)
