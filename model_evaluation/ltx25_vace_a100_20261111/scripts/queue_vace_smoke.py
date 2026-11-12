"""Chain the independent VACE first pass after resident native workers exit."""
import os
import subprocess
import time
from common import *
from native_worker import SMOKE
import fcntl
lock=(ROOT/'metadata/vace_smoke_launcher.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
    files=[ROOT/'outputs'/model/f'{sid}.json' for model in ('ltx_base','a849') for sid in SMOKE]
    if all(p.exists() and read(p).get('status') in ('complete','failed') for p in files):
        # Worker teardown releases the last CUDA contexts before the next stage.
        time.sleep(20)
        env=os.environ.copy();env['A100_VACE_ONLY']='1'
        cmd=['srun','--jobid=199052','--partition=testqueue','--account=prj0000000275','--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task=56','--gres=gpu:4','bash',str(ROOT/'scripts/launch_stage.sh'),'smoke']
        with (ROOT/'logs/vace_smoke_start.log').open('a') as log:
            result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        atomic(ROOT/'metadata/vace_smoke_step.json',{'exit_code':result.returncode,'completed_unix':time.time(),'command':cmd})
        break
    time.sleep(60)
