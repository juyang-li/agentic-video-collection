import argparse
from common import *
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['native','vace','parents','h3','all'],required=True);a=parser.parse_args()
items=rows(ROOT/'metadata/inputs.jsonl')
if a.phase in ['native','vace']:
 ids=[r['id']for r in items if r['cohort']=='candidate28'];models=['ltx_base','a849']if a.phase=='native'else['vace']
elif a.phase=='parents':ids=[r['id']for r in items if r['cohort']=='retained88'];models=['ltx_base','a849','vace']
else:ids=[r['id']for r in items];models=['h3_pdd8_native']if a.phase=='h3'else['ltx_base','a849','vace','h3_pdd8_native']
missing=[]
for sid in ids:
 for model in models:
  p=ROOT/'outputs'/model/f'{sid}.json'
  if not p.exists()or read(p)['status']!='complete':missing.append([sid,model]);continue
  r=read(p);assert sha(p.with_suffix('.mp4'))==r['output_sha256']
print(a.phase,'missing',len(missing),flush=True)
raise SystemExit(bool(missing))
