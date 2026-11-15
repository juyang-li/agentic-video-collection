"""Official pinned VBench functions. CPU raw flicker; GPU quality/static filtering."""
import argparse,sys,traceback,subprocess
from functools import lru_cache
from common import *
import math
def finite_score(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def inventory():
 result=smoke_media()[0] if os.environ.get('VB_SMOKE')=='1' else rows(Path(os.environ.get('VB_SCORING_MANIFEST',ROOT/'metadata/scoring_media.jsonl')))
 for r in result:assert sha(r['path'])==r['output_sha256']
 return result

@lru_cache(maxsize=1)
def provenance():
 vendor=OLD/'runtime/VBench'
 return dict(vbench_revision=subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip(),script_sha256=sha(__file__),manifest_sha256=sha(PKG/'inputs.jsonl'),implementation_sha256=sha(ROOT/'metadata/implementation.json'))

def compatible(path,h):
 if not path.exists():return False
 r=read(path)
 return r.get('status')=='complete' and r.get('implementation_sha256')==sha(ROOT/'metadata/implementation.json') and r.get('output_sha256',r.get('media_sha256'))==h

def raw():
 # Official cal_score is CPU-only; do not require a Slurm allocation.
 from vbench.temporal_flickering import cal_score
 import vbench.temporal_flickering as official
 dest=ROOT/'metadata/vbench/raw';dest.mkdir(parents=True,exist_ok=True)
 for r in inventory():
  h=r['output_sha256']
  if not h:continue
  p=dest/f'{h}.json'
  if compatible(p,h):continue
  rec=dict(output_sha256=h,path=r['path'],dimension='temporal_flickering',official_function='vbench.temporal_flickering.cal_score',official_source_sha256=sha(official.__file__),**provenance())
  try:
   value=float(cal_score(r['path']));assert finite_score(value)
   rec.update(status='complete',raw_score=value,score=value)
  except Exception:rec.update(status='failed',error=traceback.format_exc())
  atomic(p,rec)
 atomic(ROOT/'metadata/raw_completion.json',{'status':'finished','expected_task_scores':len(inventory())})
 print('Raw temporal flickering complete; normal motion affects this value.',flush=True)


def quality(shard):
 require_job()
 import torch,gc
 from vbench import VBench
 dims=[d for d in DIMS if d!='temporal_flickering'][shard::4]
 videos={r['output_sha256']:r['path'] for r in inventory()}
 dest=ROOT/'metadata/vbench/quality';dest.mkdir(parents=True,exist_ok=True)
 for dim in dims:
  pending={h:path for h,path in videos.items() if not compatible(dest/f'{dim}--{h}.json',h)}
  entries=list(pending.items())
  for offset in range(0,len(entries),64):
   chunk=dict(entries[offset:offset+64]);links=ROOT/'runtime/score_inputs'/dim;links.mkdir(parents=True,exist_ok=True)
   for stale in links.glob('*.mp4'):
    assert stale.is_symlink();stale.unlink()
   for h,path in chunk.items():(links/f'{h}.mp4').symlink_to(Path(path).resolve())
   result_dir=ROOT/'runtime/vbench_results'/dim;result_dir.mkdir(parents=True,exist_ok=True)
   bench=VBench(torch.device('cuda'),str(OLD/'runtime/VBench/vbench/VBench_full_info.json'),str(result_dir))
   def persist(data):
    for entry in data[dim][1]:
     h=Path(entry['video_path']).stem;assert h in chunk
     value=float(entry['video_results']);assert finite_score(value)
     atomic(dest/f'{dim}--{h}.json',dict(status='complete',dimension=dim,output_sha256=h,raw_score=value,score=value/100 if dim=='imaging_quality' else value,**provenance()))
   name=f'{dim}_{offset}'
   try:
    bench.evaluate(str(links),name,dimension_list=[dim],local=True,mode='custom_input',imaging_quality_preprocessing_mode='longer')
    persist(read(result_dir/f'{name}_eval_results.json'))
   except Exception:atomic(result_dir/f'{name}_batch_error.json',dict(error=traceback.format_exc()))
   # A returned aggregate can omit an unreadable clip: retry every missing individual explicitly.
   for h,path in chunk.items():
    p=dest/f'{dim}--{h}.json'
    if compatible(p,h):continue
    try:
     name=f'{dim}--{h}'
     bench.evaluate(str(links/f'{h}.mp4'),name,dimension_list=[dim],local=True,mode='custom_input',imaging_quality_preprocessing_mode='longer')
     persist(read(result_dir/f'{name}_eval_results.json'))
     assert p.exists() and read(p)['status']=='complete','Official evaluator omitted video'
    except Exception:atomic(p,dict(status='failed',output_sha256=h,dimension=dim,error=traceback.format_exc(),**provenance()))
   del bench;gc.collect();torch.cuda.empty_cache()
   print(dim,offset+len(chunk),'/',len(entries),flush=True)

def static(shard):
 if not static_required():
  print('Static filtering omitted by explicit user request',flush=True);return
 require_job()
 import importlib.util,types
 vendor=OLD/'runtime/VBench';spec=importlib.util.spec_from_file_location('a849_official_static_filter',vendor/'static_filter.py')
 official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
 args=argparse.Namespace(model=str(OLD/'runtime/vbench_models/raft_model/models/raft-things.pth'),small=False,mixed_precision=False,alternate_corr=False)
 filt=official.StaticFilter(args,device='cuda')
 videos={}
 for r in inventory():
  if r['output_sha256']:videos[r['output_sha256']]=r['path']
 for source in (smoke_media()[1] if os.environ.get('VB_SMOKE')=='1' else rows(Path(os.environ.get('VB_SOURCE_MANIFEST',ROOT/'metadata/source_media.jsonl')))):
  if os.environ.get('VB_SMOKE')=='1' and source['id'] not in SMOKE:continue
  videos[source['output_sha256']]=source['path']
 dest=ROOT/'metadata/vbench/static';dest.mkdir(parents=True,exist_ok=True)
 for h in sorted(videos)[shard::4]:
  p=dest/f'{h}.json'
  if compatible(p,h):continue
  rec=dict(media_sha256=h,path=videos[h],official_function='static_filter.StaticFilter.infer',official_source_sha256=sha(vendor/'static_filter.py'),raft_weights_sha256=sha(args.model),**provenance())
  try:rec.update(status='complete',is_static=bool(filt.infer(videos[h])),parameters=dict(filt.params))
  except Exception:rec.update(status='failed',error=traceback.format_exc())
  atomic(p,rec);print(h[:12],rec['status'],rec.get('is_static'),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['raw','quality','static']);p.add_argument('--shard',type=int,choices=range(4),default=0);a=p.parse_args()
 if a.mode=='raw':raw()
 elif a.mode=='quality':quality(a.shard)
 else:static(a.shard)
