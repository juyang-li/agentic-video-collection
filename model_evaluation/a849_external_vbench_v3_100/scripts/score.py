"""Official pinned VBench functions. CPU raw flicker; GPU quality/static filtering."""
import argparse,sys,traceback,subprocess
from functools import lru_cache
from common import *
from contracts import DIMS,finite_score
MODELS=('ltx_base','a849','vace','h3_pdd8')
OLD=ROOT.parent/'a849_external_vbench_v2_96'
def inventory():
 result=[]
 for item in rows(ROOT/'metadata/inputs.jsonl'):
  for model in MODELS:
   path=ROOT/'outputs'/model/f"{item['id']}.mp4";receipt=path.with_suffix('.json')
   r=read(receipt)if receipt.exists()else {'status':'missing'};ok=r.get('status')=='complete'
   if ok:assert path.is_file()and sha(path)==r['output_sha256']
   result.append(dict(cohort=item['cohort'],id=item['id'],group=item['group'],subgroup=item.get('subgroup'),model=model,path=str(path),output_sha256=r.get('output_sha256')if ok else None,generation_status=r['status']))
 return result

@lru_cache(maxsize=1)
def provenance():
 vendor=OLD/'runtime/VBench'
 return dict(vbench_revision=subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip(),script_sha256=sha(__file__),manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'))

def raw():
 # Official cal_score is CPU-only; do not require a Slurm allocation.
 from vbench.temporal_flickering import cal_score
 import vbench.temporal_flickering as official
 dest=ROOT/'metadata/vbench/raw';dest.mkdir(parents=True,exist_ok=True)
 for r in inventory():
  h=r['output_sha256']
  if not h:continue
  p=dest/f'{h}.json'
  if p.exists() and read(p)['status']=='complete':continue
  rec=dict(output_sha256=h,path=r['path'],dimension='temporal_flickering',official_function='vbench.temporal_flickering.cal_score',official_source_sha256=sha(official.__file__),**provenance())
  try:
   value=float(cal_score(r['path']));assert finite_score(value)
   rec.update(status='complete',raw_score=value,score=value)
  except Exception:rec.update(status='failed',error=traceback.format_exc())
  atomic(p,rec)
 print('Raw temporal flickering complete; normal motion affects this value.',flush=True)

def reuse_metrics():
 import csv
 known={(r['output_sha256'],r['dimension']):r for r in csv.DictReader((OLD/'metadata/per_video_metrics.csv').open()) if r['dimension'] in DIMS and r['dimension']!='temporal_flickering' and r['status']=='complete' and r['score']!=''}
 dest=ROOT/'metadata/vbench/quality';dest.mkdir(parents=True,exist_ok=True);n=0
 for r in inventory():
  h=r['output_sha256']
  if not h:continue
  for dim in DIMS:
   if (h,dim) not in known:continue
   old=known[h,dim];p=dest/f'{dim}--{h}.json'
   if p.exists():continue
   atomic(p,dict(status='complete',dimension=dim,output_sha256=h,raw_score=float(old['raw_score']),score=float(old['score']),reused_from=str(OLD/'metadata/per_video_metrics.csv'),reused_metrics_sha256=sha(OLD/'metadata/per_video_metrics.csv'),reused_by='exact output SHA256',**provenance()));n+=1
 print('Reused official quality scores',n,flush=True)

def quality(shard):
 require_job()
 import torch,csv
 from vbench import VBench
 dims=[d for d in DIMS if d!='temporal_flickering'][shard::4]
 known={}
 for r in csv.DictReader((OLD/'metadata/per_video_metrics.csv').open()):
  if r['dimension'] in DIMS and r['status']=='complete' and r['score']!='':
   known[(r['output_sha256'],r['dimension'])]=r
 videos={r['output_sha256']:r['path'] for r in inventory() if r['output_sha256']}
 dest=ROOT/'metadata/vbench/quality';dest.mkdir(parents=True,exist_ok=True)
 for dim in dims:
  pending={}
  for h,path in videos.items():
   p=dest/f'{dim}--{h}.json'
   if p.exists() and read(p)['status']=='complete':continue
   if (h,dim) in known:
    old=known[h,dim];atomic(p,dict(status='complete',dimension=dim,output_sha256=h,raw_score=float(old['raw_score']),score=float(old['score']),reused_from=str(OLD/'metadata/per_video_metrics.csv'),reused_by='exact output SHA256',**provenance()))
   else:pending[h]=path
  if not pending:continue
  links=ROOT/'runtime/score_inputs'/dim;links.mkdir(parents=True,exist_ok=True)
  for stale in links.glob('*.mp4'):
   if stale.stem not in pending:
    assert stale.is_symlink();stale.unlink()
  for h,path in pending.items():
   target=links/f'{h}.mp4'
   if not target.exists():target.symlink_to(Path(path).resolve())
   assert target.resolve()==Path(path).resolve()
  result_dir=ROOT/'runtime/vbench_results'/dim;result_dir.mkdir(parents=True,exist_ok=True)
  bench=VBench(torch.device('cuda'),str(OLD/'runtime/VBench/VBench_full_info.json'),str(result_dir))
  def persist(data):
   for entry in data[dim][1]:
    h=Path(entry['video_path']).stem;assert h in pending
    value=float(entry['video_results']);assert finite_score(value)
    atomic(dest/f'{dim}--{h}.json',dict(status='complete',dimension=dim,output_sha256=h,raw_score=value,score=value/100 if dim=='imaging_quality' else value,**provenance()))
  try:
   bench.evaluate(str(links),dim,dimension_list=[dim],local=True,mode='custom_input',imaging_quality_preprocessing_mode='longer')
   persist(read(result_dir/f'{dim}_eval_results.json'))
  except Exception:
   atomic(result_dir/'batch_error.json',dict(error=traceback.format_exc()))
   for h,path in pending.items():
    if (dest/f'{dim}--{h}.json').exists():continue
    try:
     name=f'{dim}--{h}';bench.evaluate(str(links/f'{h}.mp4'),name,dimension_list=[dim],local=True,mode='custom_input',imaging_quality_preprocessing_mode='longer');persist(read(result_dir/f'{name}_eval_results.json'))
    except Exception:atomic(dest/f'{dim}--{h}.json',dict(status='failed',output_sha256=h,dimension=dim,error=traceback.format_exc(),**provenance()))
  print(dim,'complete',flush=True)

def static(shard):
 require_job()
 import importlib.util,types
 vendor=OLD/'runtime/VBench';spec=importlib.util.spec_from_file_location('a849_official_static_filter',vendor/'static_filter.py')
 official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
 args=argparse.Namespace(model=str(OLD/'runtime/vbench_models/raft_model/models/raft-things.pth'),small=False,mixed_precision=False,alternate_corr=False)
 filt=official.StaticFilter(args,device='cuda')
 videos={}
 for r in inventory():
  if r['output_sha256']:videos[r['output_sha256']]=r['path']
 for root in [ROOT]:
  for item in rows(root/'metadata/inputs.jsonl'):
   path=root/'inputs/prepared'/item['id']/'source.mp4';videos[sha(path)]=str(path)
 dest=ROOT/'metadata/vbench/static';dest.mkdir(parents=True,exist_ok=True)
 for h in sorted(videos)[shard::4]:
  p=dest/f'{h}.json'
  if p.exists() and read(p)['status']=='complete':continue
  rec=dict(media_sha256=h,path=videos[h],official_function='static_filter.StaticFilter.infer',official_source_sha256=sha(vendor/'static_filter.py'),raft_weights_sha256=sha(args.model),**provenance())
  try:rec.update(status='complete',is_static=bool(filt.infer(videos[h])),parameters=dict(filt.params))
  except Exception:rec.update(status='failed',error=traceback.format_exc())
  atomic(p,rec);print(h[:12],rec['status'],rec.get('is_static'),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['raw','quality','static','reuse']);p.add_argument('--shard',type=int,choices=range(4),default=0);a=p.parse_args()
 if a.mode=='reuse':reuse_metrics()
 elif a.mode=='raw':raw()
 elif a.mode=='quality':quality(a.shard)
 else:static(a.shard)
