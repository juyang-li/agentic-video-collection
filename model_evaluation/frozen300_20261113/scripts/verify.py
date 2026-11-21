"""Freeze gate: identities, temporal legality, portable allowlisted inputs, full media decode evidence."""
from build import *
from bisect import bisect_left

def verify():
 p=ROOT/'package';inputs=rows(p/'inputs.jsonl');gt=rows(p/'ground_truth.jsonl');inventory=rows(p/'media_inventory.jsonl');probes=read(ROOT/'metadata/media_probes.json')
 byid={x['id']:x for x in inputs};gts={x['id']:x for x in gt}
 assert len(inputs)==len(byid)==len(gts)==300 and len({x['uid'] for x in inputs})==300
 assert Counter(x['dataset'] for x in inputs)=={'ue':200,'coinve':75,'tripvvt':25}
 assert all(sum(x['group']==g for x in inputs)==25 for g in GROUPS)
 old=rows(OLD)
 for x in old:
  a=byid[x['id']];r=x['record'];assert a['uid']==x['uid'] and a['prompt']==r['prompt'] and a['retained_from_old100']
  assert a['source']['sha256']==r['source']['sha256'] and gts[a['id']]['target']['sha256']==r['target']['sha256']
  assert [(m['sha256'],m.get('type'),m.get('view')) for m in a['references']]==[(m['sha256'],m.get('type'),m.get('view')) for m in r['references']]
 assert sum(x['retained_from_old100'] for x in inputs)==100
 forbidden={'target','caption','target_caption','caption_key','ground_truth','video_latents','target_node_id','absolute_path','original_absolute_path','media_root'}
 def walk(o):
  if isinstance(o,dict):
   assert not (set(o)&forbidden),set(o)&forbidden
   for v in o.values():walk(v)
  elif isinstance(o,list):
   for v in o:walk(v)
 inv={m['path']:m for m in inventory};assert len(inv)==len(inventory)
 referenced=set()
 for a in inputs:
  walk(a);assert a['split']=='test'
  b=gts[a['id']];assert b['uid']==a['uid'] and b['temporal_window']==a['temporal_window']
  ts=a['timestamps_seconds'];assert len(ts)==a['num_frames'] and a['fps']==12 and (len(ts)-1)%8==0
  w=a['temporal_window'];assert ts[0]==w['start_seconds'] and ts[-1]==w['end_seconds'] and ts[-1]<=w['common_valid_end_seconds']+1e-6
  assert 0<ts[-1]-ts[0]<=10+1e-6
  if a['dataset']=='ue':assert len(ts)==121 and ts[0]==0 and ts[-1]==10
  else:assert ts[-1]+8/12>w['common_valid_end_seconds']-1e-6
  assert all(abs(t-(ts[0]+i/12))<1e-8 for i,t in enumerate(ts))
  assert [m['index'] for m in a['references']]==list(range(len(a['references'])))
  if a['dataset']=='coinve':
   assert not a['references'] and len(a['ordered_instructions'])>=2
   assert a['prompt']['en']=='\n'.join(f'{i+1}. {v}' for i,v in enumerate(a['ordered_instructions']))
  if a['dataset']=='tripvvt':assert len(a['references'])==1 and a['references'][0]['view']=='official' and b['caption']
  for m,ix in [(a['source'],a['source']['sample_frame_indices']),*[(m,m.get('sample_frame_indices')) for m in a['references']],(b['target'],b['sample_frame_indices'])]:
   rel=Path(m['path']);assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0]=='media'
   assert m['path'] in inv and inv[m['path']]['sha256']==m['sha256'];referenced.add(m['path'])
   if ix is not None:
    pts=probes[m['sha256']]['pts_seconds'];assert len(ix)==len(ts)
    end_bound=pts[-1]+(pts[-1]-pts[-2] if a['dataset']=='ue' else 0)
    assert ts[0]>=pts[0]-1e-7 and ts[-1]<=end_bound+1e-7
    for j,t in zip(ix,ts):
     k=min(bisect_left(pts,t),len(pts)-1)
     if k and abs(pts[k-1]-t)<abs(pts[k]-t)-1e-9:k-=1
     assert j==k
 assert referenced==set(inv)
 for i,m in enumerate(inventory):
  f=p/m['path'];assert f.is_file() and not f.is_symlink() and sha(f)==m['sha256'] and f.stat().st_size==m['bytes']
  assert m['full_decode'] and probes[m['sha256']]['full_decode']
  if i%100==0:print('checked packaged media',i,'/',len(inventory),flush=True)
 # Validate relocated path resolution under a separate package root without depending on the CWD.
 import tempfile,os
 with tempfile.TemporaryDirectory(prefix='frozen300_relocate_') as tmp:
  relocated=Path(tmp)/'dataset';os.symlink(p.resolve(),relocated)
  for rel in referenced:assert (relocated/rel).is_file()
 summary=read(p/'sampling.json');assert summary['input_manifest_sha256']==sha(p/'inputs.jsonl')
 result={'status':'passed','pair_count':300,'retained_old100':100,'media_files':len(inventory),'unique_sources':summary['unique_sources'],'input_manifest_sha256':sha(p/'inputs.jsonl'),'checks':['exact quotas','old100 identities/full prompts/ordered references','all media fully decoded from original bytes','packaged media SHA256/bytes','no missing references','absolute timestamps and maximal legal 8n+1 length','relative media paths and relocated root resolution','GT/caption excluded from inference','ordered complete CoinVE sub-instructions','official TripVVT garment'],'human_quality_review':'not_performed','automatic_quality_scoring':'not_performed','checked_at_utc':datetime.now(timezone.utc).isoformat()}
 write(p/'validation.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':verify()
