"""Read-only local progress; generation receipts, not inferred from log percentages."""
from collections import Counter
from common import *
items=rows(ROOT/'metadata/inputs.jsonl');print('candidate116, frozen manifest',sha(ROOT/'metadata/inputs.jsonl'))
for model in ['ltx_base','a849','vace','h3_pdd8_native','h3_pdd8']:
 counts=Counter();new=Counter()
 for item in items:
  p=ROOT/'outputs'/model/f"{item['id']}.json";status=read(p)['status']if p.exists()else'missing';counts[status]+=1
  if item['cohort']=='candidate28':new[status]+=1
 print(model,'all116',dict(counts),'candidates28',dict(new))
for name in ['visual_observations','gpu_generation','gpu_scores_complete','completion_watch','inputs_baidu_delivery','baidu_delivery']:
 p=ROOT/'metadata'/f'{name}.json'
 if p.exists():
  r=read(p);print(name,{k:r[k]for k in ['status','reviewed_outputs','expected_outputs','remote','checked_unix']if k in r})
