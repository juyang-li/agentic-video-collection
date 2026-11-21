from common import *
import time
require_job()
for p in (ROOT/'outputs').glob('*.json'):
 s=read(p)
 if s['status']=='running' and s.get('slurm_job_id')!=os.environ['SLURM_JOB_ID']:
  atomic(ROOT/'metadata/interrupted_attempts'/f'{p.stem}_{time.time_ns()}.json',s);p.unlink()
