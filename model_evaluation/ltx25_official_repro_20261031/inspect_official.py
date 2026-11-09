from pathlib import Path
import requests,json,re
ROOT=Path(__file__).resolve().parent
p=Path('/scratch/prj0000000275/jcy/.cache/huggingface/token');token=p.read_text().strip() if p.exists() else None
s=requests.Session()
for name in ['LTX-2.5-22b-IC-LoRA-Day-To-Night','LTX-2.5-22b-IC-LoRA-Ingredients','LTX-2.3-22b-IC-LoRA-Instant-Shave']:
 repo='Lightricks/'+name
 api=s.get('https://huggingface.co/api/models/'+repo,headers={'Authorization':'Bearer '+token} if token else {},timeout=60)
 if api.ok:
  d=api.json();(ROOT/'metadata'/f'{name}_repo.json').write_text(json.dumps({k:d.get(k) for k in ['id','sha','gated','siblings']},indent=2)+'\n')
  print(name,'revision',d.get('sha'),'files',[v['rfilename'] for v in d.get('siblings',[])],flush=True)
 card=s.get('https://huggingface.co/'+repo+'/resolve/main/README.md',headers={'Authorization':'Bearer '+token} if token else {},timeout=60)
 print('card_status',name,card.status_code,flush=True)
 if card.ok:
  (ROOT/'metadata'/f'{name}_README.md').write_text(card.text)
  print('media',re.findall(r'(?:src=|https://)[^\s<>\)]+',card.text)[:30],flush=True)
