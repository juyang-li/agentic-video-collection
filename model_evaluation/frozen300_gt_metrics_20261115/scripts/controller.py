from common import *
import subprocess,time,fcntl,traceback

def main():
 lock=(ROOT/'logs/controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=os.environ.copy();env['PYTHONNOUSERSITE']='1';env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
 # Wait for explicit successful download marker; no GPU reserved during download.
 while not (ROOT/'metadata/download.json').exists():
  pid=int((ROOT/'metadata/download.pid').read_text())
  try:os.kill(pid,0)
  except ProcessLookupError:raise RuntimeError('Downloader exited without completion receipt')
  time.sleep(30)
 assert read(ROOT/'metadata/download.json')['status']=='complete'
 while True:
  failures=list((ROOT/'metadata').glob('cpu_failure_*.json'))
  if failures:raise RuntimeError('CPU evaluation failed: '+str(failures))
  expected=[ROOT/f'cpu_scores/{model}/{item["id"]}.json' for model in MODELS for item in rows(PKG/'inputs.jsonl')]
  if all(p.exists() for p in expected):
   digest=sha(ROOT/'metadata/protocol.json')
   assert all(read(p)['protocol_sha256']==digest for p in expected)
   atomic(ROOT/'metadata/cpu_progress.json',{'status':'complete','completed':1200,'scopes':['local','remote']})
   break
  time.sleep(30)
 with (ROOT/'logs/cpu_tests.log').open('w') as log:subprocess.run([str(ROOT/'runtime/bin/python'),str(ROOT/'scripts/test_cpu.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert read(ROOT/'metadata/cpu_tests.json')['status']=='passed'
 if (ROOT/'metadata/submission.json').exists():return
 result=subprocess.run(['sbatch','--parsable',str(ROOT/'scripts/run.sbatch')],capture_output=True,text=True,check=True)
 job=result.stdout.strip().split(';')[0];assert job.isdigit();atomic(ROOT/'metadata/submission.json',{'status':'submitted','job_id':job,'partition':'testqueue','gpu':'H200:1','cpus':16,'memory':'128G','time':'08:00:00'})
 print('SUBMITTED',job,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:atomic(ROOT/'metadata/controller_failure.json',{'error':traceback.format_exc()});raise
