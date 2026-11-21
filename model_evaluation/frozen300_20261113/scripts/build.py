"""CPU-only deterministic freeze; never reads model outputs or training/validation pools."""
import concurrent.futures as cf
import hashlib
import json
import math
import random
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
NATIVE = Path('/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5')
BUNDLE = NATIVE/'data/mixed/outfit_camera_720p/sources.json'
OLD = ROOT.with_name('ltx25_vace_a100_20261111')/'metadata/selection.jsonl'
GROUPS = ['identity','single_piece','whole_outfit','action','site','hair','hair_color','camera']
SEED = 20261113

def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
 return h.hexdigest()
def write(p,o):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
 tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def jsonl(p,rs):
 p=Path(p);content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs)
 if p.exists():assert p.read_text()==content,f'Frozen JSONL changed: {p}'
 else:p.write_text(content)
def attributes(r):
 a=r['source']['attributes'];t=r['target'].get('attributes',{})
 key={'identity':'identity_id','action':'action_id','site':'scene_id','hair':'hair_id','hair_color':'hair_color_id','camera':'camera_id'}.get(r['axis'],'clothing')
 return (a.get('sex_id'),a.get('aspect_id'),json.dumps(t.get(key),sort_keys=True))
def resolve(m):
 p=Path(m['absolute_path'])
 if p.is_file(): return p
 # Retired Lance files were relocated into the canonical 11K dataset. Require byte identity.
 q=PROJECT/'dataset_research/coinve_tripvvt_11k_20261108/media/coinve'/p.name
 if q.is_file() and sha(q)==m['sha256']: return q
 raise FileNotFoundError(p)
def select():
 selected=rows(OLD);rng=random.Random(SEED)
 pools={};provenance={}
 for s in read(BUNDLE)['sources']:
  pool=rows(s['test_manifest']);assert all(r['split']=='test' for r in pool)
  pools[s['name']]=pool
  provenance[s['name']]={'path':s['test_manifest'],'sha256':sha(s['test_manifest']),'pairs':len(pool)}
 # Verify the old identity still belongs to the formal test pool.
 for x in selected:
  r=next(r for r in pools[x['dataset']] if str(r['sample_id'])==x['uid'].split(':')[1])
  old=x['record'];assert old['prompt']==r['prompt']
  assert old['source']['sha256']==r['source']['sha256'] and old['target']['sha256']==r['target']['sha256']
  assert [(a['sha256'],a['type'],a.get('view')) for a in old['references']]==[(a['sha256'],a['type'],a.get('view')) for a in r['references']]
 used_uid={x['uid'] for x in selected};used_src={x['record']['source']['sha256'] for x in selected}
 def take(pool,n,group,subgroup=None):
  dataset='ue' if group in GROUPS else group
  candidates=[r for r in sorted(pool,key=lambda r:r['sample_id']) if f'{dataset}:{r["sample_id"]}' not in used_uid]
  rng.shuffle(candidates)
  counts=[Counter() for _ in range(3)]
  for x in selected:
   if x['group']==group:
    for c,v in zip(counts,attributes(x['record'])):c[v]+=1
  for _ in range(n):
   assert candidates,(group,n)
   def score(r):
    v=attributes(r)
    return (r['source']['sha256'] in used_src,counts[0][v[0]]+counts[1][v[1]],counts[2][v[2]])
   r=min(candidates,key=score);candidates.remove(r)
   for c,v in zip(counts,attributes(r)):c[v]+=1
   uid=f'{dataset}:{r["sample_id"]}';used_uid.add(uid);used_src.add(r['source']['sha256'])
   selected.append({'id':f'{group}_{1+sum(x["group"]==group for x in selected):02d}','uid':uid,'dataset':dataset,'group':group,'subgroup':subgroup,'record':r})
 for g in GROUPS:
  pool=[r for r in pools['ue'] if (r['mode']==g if g in ['single_piece','whole_outfit'] else r['axis']==g)]
  if g=='single_piece':
   for slot,n in [('shirts',4),('full_body',4),('skirts',4),('pants',3)]:take([r for r in pool if r['sample_metadata'].get('slot')==slot],n,g,slot)
  else:take(pool,15,g)
 # 53 two-edit / 22 three-edit pairs in total; all original sub-instructions preserved.
 for edits,n in [(2,46),(3,19)]:take([r for r in pools['coinve'] if len(r['provenance']['instructions'])==edits],n,'coinve',f'{edits}_edits')
 for category,n in [('upper_body',5),('lower_body',5),('dresses',5)]:take([r for r in pools['tripvvt'] if r['references'][0]['attributes']['category']==category],n,'tripvvt',category)
 assert len(selected)==len(used_uid)==300
 return selected,provenance

def inspect_media(entry):
 import av
 from PIL import Image
 digest,m=entry;p=resolve(m);assert sha(p)==digest,(p,'sha mismatch')
 info={'sha256':digest,'bytes':p.stat().st_size,'original_path':str(p),'manifest_original_path':m['absolute_path']}
 if p.suffix.lower() in ['.png','.jpg','.jpeg','.webp']:
  with Image.open(p) as im:info.update(kind='image',width=im.width,height=im.height);im.load()
  info['full_decode']=True
 else:
  with av.open(str(p)) as c:
   s=c.streams.video[0];s.thread_type='NONE';s.codec_context.thread_count=1
   pts=[];wh=set()
   for f in c.decode(s):
    assert f.pts is not None
    pts.append(float(f.pts*f.time_base));wh.add((f.width,f.height))
   assert len(pts)>1 and len(wh)==1 and all(b>a for a,b in zip(pts,pts[1:]))
   w,h=next(iter(wh));info.update(kind='video',width=w,height=h,frames=len(pts),fps=str(s.average_rate),first_frame_seconds=pts[0],last_frame_seconds=pts[-1],pts_seconds=pts,full_decode=True)
 return digest,info

def main():
 assert not __import__('os').environ.get('SLURM_JOB_ID'),'Run pure CPU preparation on login'
 package=ROOT/'package';(package/'media').mkdir(parents=True,exist_ok=True)
 selection=ROOT/'metadata/selection.jsonl'
 selected,provenance=select()
 if selection.exists():assert rows(selection)==selected,'Frozen selection changed'
 else:jsonl(selection,selected)
 unique={m['sha256']:m for x in selected for m in [x['record']['source'],x['record']['target'],*x['record']['references']]}
 cachepath=ROOT/'metadata/media_probes.json';probes=read(cachepath) if cachepath.exists() else {}
 pending=[e for e in unique.items() if e[0] not in probes]
 with cf.ThreadPoolExecutor(max_workers=4) as pool:
  for i,(k,v) in enumerate(pool.map(inspect_media,pending)):
   probes[k]=v
   if i%20==0:write(cachepath,probes);print('full decoded',len(probes),'/',len(unique),flush=True)
 write(cachepath,probes)
 inventory=[]
 for k,m in unique.items():
  p=resolve(m);dst=package/'media'/(k+p.suffix.lower())
  if not dst.exists():shutil.copyfile(p,dst)
  assert sha(dst)==k
  inventory.append({**{a:b for a,b in probes[k].items() if a not in ['original_path','manifest_original_path','pts_seconds']},'path':dst.relative_to(package).as_posix(),'origin':{'dataset_path':m['path'],'manifest_sha256':k}})
 inventory_by_sha={m['sha256']:m for m in inventory}
 def media(m):
  # Allowlist: never propagate reference provenance containing a GT video path.
  v={k:m[k] for k in ['sha256','media_key','reference_id','type','view','attributes'] if k in m}
  inv=inventory_by_sha[m['sha256']];v.update(path=inv['path'],original_wh=[inv['width'],inv['height']],bytes=inv['bytes'])
  return v
 inputs=[];gt=[];oldids={x['id'] for x in rows(OLD)}
 frozen_at=datetime.now(timezone.utc).isoformat()
 previous=package/'sampling.json'
 if previous.exists():frozen_at=read(previous)['frozen_at_utc']
 for x in selected:
  r=x['record'];videos=[r['source'],r['target'],*[m for m in r['references'] if m['type']=='motion']]
  start=max(0.,*[probes[m['sha256']]['first_frame_seconds'] for m in videos]);end=min(start+10,*[probes[m['sha256']]['last_frame_seconds'] for m in videos])
  if x['dataset']=='ue':
   coverage=min(probes[m['sha256']]['pts_seconds'][-1]+(probes[m['sha256']]['pts_seconds'][-1]-probes[m['sha256']]['pts_seconds'][-2]) for m in videos)
   assert start==0 and coverage>=10-1e-6
   end=10.;n=121
  else:n=8*math.floor(((end-start)*12+1e-7)/8)+1
  assert n>=9 and (n-1)%8==0
  times=[start+i/12 for i in range(n)];assert times[-1]<=end+1e-6
  indices={}
  for m in videos:
   from bisect import bisect_left
   pts=probes[m['sha256']]['pts_seconds'];ix=[]
   for t in times:
    j=bisect_left(pts,t);j=min(j,len(pts)-1)
    if j and abs(pts[j-1]-t)<abs(pts[j]-t)-1e-9:j-=1
    ix.append(j)
   indices[m['sha256']]=ix
  window={'synthetic_endpoint_policy':('t=10 maps to final native display frame when original clip covers 10 seconds; at most one native frame interval, no long tail' if x['dataset']=='ue' else None),'start_seconds':start,'end_seconds':times[-1],'common_valid_end_seconds':end,'fps':12,'num_frames':n,'timestamps_seconds':times,'sampling':'nearest decoded presentation timestamp; ties choose later (1e-9 second numerical tolerance); absolute clock; no speed change or tail padding'}
  record={k:x[k] for k in ['id','uid','dataset','group','subgroup']}
  record.update(schema='frozen300_inputs_v1',prompt=r['prompt'],source=media(r['source']),references=[dict(media(m),index=i) for i,m in enumerate(r['references'])],canvas_wh=[1280,736] if r['source']['attributes']['aspect_id']==1 else [736,1280],fps=12,num_frames=n,timestamps_seconds=times,temporal_window=window,frozen_at_utc=frozen_at,retained_from_old100=x['id'] in oldids,split='test')
  record['source']['sample_frame_indices']=indices[r['source']['sha256']]
  for a,m in zip(record['references'],r['references']):
   if m['type']=='motion':a['sample_frame_indices']=indices[m['sha256']]
  if x['dataset']=='coinve':record['ordered_instructions']=r['provenance']['instructions']
  inputs.append(record)
  gt.append({'id':x['id'],'uid':x['uid'],'target':media(r['target']),'sample_frame_indices':indices[r['target']['sha256']],'caption':r.get('caption'),'caption_key':r.get('caption_key'),'temporal_window':window,'review_only':True})
 jsonl(package/'inputs.jsonl',inputs);jsonl(package/'ground_truth.jsonl',gt);jsonl(package/'media_inventory.jsonl',inventory)
 summary={'seed':SEED,'frozen_at_utc':frozen_at,'pair_count':300,'counts':dict(Counter(x['group'] for x in selected)),'dataset_pairs':dict(Counter(x['dataset'] for x in selected)),'unique_sources':len({x['source']['sha256'] for x in inputs}),'unique_sources_by_dataset':{d:len({x['source']['sha256'] for x in inputs if x['dataset']==d}) for d in ['ue','coinve','tripvvt']},'retained_old100':100,'external96_included':False,'authority_bundle_sha256':sha(BUNDLE),'old_selection_sha256':sha(OLD),'test_pools':{k:{a:b for a,b in v.items() if a!='path'} for k,v in provenance.items()},'sampling':'old 100 fixed; seeded greedy source uniqueness, then sex/aspect balance and target attribute diversity; no generated outputs read','ue_balance':{g:{'sex':dict(Counter(str(attributes(x['record'])[0]) for x in selected if x['group']==g)),'aspect':dict(Counter(str(attributes(x['record'])[1]) for x in selected if x['group']==g))} for g in GROUPS},'input_manifest_sha256':sha(package/'inputs.jsonl')}
 write(package/'sampling.json',summary);write(ROOT/'metadata/provenance.json',provenance)
 print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
