"""Read-only status; never submit jobs, rebuild media or aggregate scores."""
from common import *
from collections import Counter
print('Prepared tasks:',len(list((ROOT/'metadata/media').glob('*.json'))),'/300')
for kind in ['quality','raw','static']:
 paths=list((ROOT/f'metadata/vbench/{kind}').glob('*.json'));counts=Counter();dims=Counter()
 for p in paths:
  r=read(p);counts[r.get('status','unknown')]+=1
  if r.get('status')=='complete':dims[r.get('dimension',kind)]+=1
 print(kind,dict(counts),dict(dims))
for name in ['run_status','submission','gpu_smoke','completion','supervisor_failure']:
 p=ROOT/f'metadata/{name}.json'
 if p.exists():
  r=read(p)
  if name=='submission':r={k:r[k] for k in ['phase','prepared_tasks','allocation','partition','time_minutes','allocation_remaining'] if k in r}
  print(name,r)
