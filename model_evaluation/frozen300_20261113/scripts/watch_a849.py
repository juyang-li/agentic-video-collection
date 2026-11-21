"""Independent 15-minute A849 watcher. Existing benchmark monitors are untouched."""
from build import *
import fcntl,subprocess,time,json

def note(message):
 with (ROOT/'logs/a849_watch.log').open('a') as f:f.write(datetime.now(timezone.utc).isoformat()+' '+message+'\n')
def main():
 lock=(ROOT/'logs/a849_watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 submission=ROOT/'metadata/a849_submission.json'
 while True:
  try:
   state=read(submission);job=state['job_id']
   q=subprocess.run(['squeue','-h','-j',job,'-o','%P|%T'],capture_output=True,text=True,timeout=30)
   if q.returncode and 'invalid job id' not in q.stderr.lower():
    note('squeue unavailable; retain job '+job+'; '+q.stderr.strip()[:200]);time.sleep(900);continue
   if q.stdout.strip():
    for row in q.stdout.splitlines():assert row.split('|')[0]=='testqueue'
    note(job+' '+q.stdout.strip());time.sleep(900);continue
   items=rows(ROOT/'package/inputs.jsonl');complete=[];failed=[]
   for x in items:
    p=ROOT/'outputs/a849'/(x['id']+'.json')
    if p.exists():
     r=read(p)
     if r.get('status')=='failed':failed.append(x['id'])
     if r.get('status')=='complete':
      assert r['contract']['input_manifest_sha256']==state['input_manifest_sha256']
      assert (ROOT/'outputs/a849'/(x['id']+'.mp4')).is_file()
      complete.append(x['id'])
   if len(complete)==300:
    state.update(status='complete',completed=300);write(submission,state);note('300/300 inference complete; human quality not reviewed');return
   a=subprocess.run(['sacct','-n','-X','-j',job,'--format=State','--parsable2'],capture_output=True,text=True,timeout=30)
   if a.returncode or not a.stdout.strip():note('accounting not ready');time.sleep(900);continue
   terminal=a.stdout.splitlines()[0].strip().split('|')[0].split()[0]
   if not failed and terminal in {'TIMEOUT','PREEMPTED','NODE_FAIL','OUT_OF_MEMORY'} and terminal!='OUT_OF_MEMORY':
    cmd=['sbatch','--parsable','--partition=testqueue','--account=prj0000000275','--output='+str(ROOT/'logs/a849_%j.slurm.log'),str(ROOT/'scripts/run_a849.sh')]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=30);result.check_returncode()
    new=result.stdout.strip().split(';')[0];assert new.isdigit()
    state.update(previous_jobs=state.get('previous_jobs',[])+[job],job_id=new,status='resubmitted',completed=len(complete));write(submission,state);note('resume '+job+' -> '+new)
   else:
    state.update(status='needs_inspection',scheduler_state=terminal,completed=len(complete),failed=failed);write(submission,state);note('stopped for inspection '+json.dumps(state));return
  except Exception as e:note(type(e).__name__+': '+str(e));time.sleep(900)
if __name__=='__main__':main()
