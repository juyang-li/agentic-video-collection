from common import *
import subprocess,fcntl
def main():
 assert not os.environ.get('SLURM_JOB_ID')
 lock=(ROOT/'logs/login_cpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for phase in ['smoke','full']:
  if phase=='full':
   while not (ROOT/'metadata/smoke.json').exists():
    if (ROOT/'metadata/node_failure.json').exists():raise RuntimeError('Node smoke failed')
    time.sleep(5)
   assert read(ROOT/'metadata/smoke.json')['status']=='passed'
  with (ROOT/f'logs/cpu_login_{phase}.log').open('ab',buffering=0) as f:subprocess.run([str(GTPY),str(ROOT/'scripts/cpu.py'),'--scope','login','--phase',phase],stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main()
