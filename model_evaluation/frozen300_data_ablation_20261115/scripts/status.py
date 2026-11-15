"""Read-only state and score counts. Does not submit or restart anything."""
from common import *
from collections import Counter
print('Verified input IDs',len(list((ROOT/'metadata/pairs').glob('*.json'))),'/300')
for kind in ['cpu','lpips','vbench']:
 fs=list((ROOT/f'scores/{kind}').glob('*/*.json'));counts=Counter();dims=Counter()
 for p in fs:
  r=read(p);counts[r['status']]+=1
  if r['status']=='complete':dims[r.get('dimension',kind)]+=1
 print(kind,dict(counts),dict(dims))
for name in ['run_status','smoke','preparation','node_done','prepare_failure','node_failure','supervisor_failure','completion','final_validation','release']:
 p=ROOT/f'metadata/{name}.json'
 if p.exists():print(name,read(p))
