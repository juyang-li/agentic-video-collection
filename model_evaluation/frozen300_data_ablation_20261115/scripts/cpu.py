from common import *
from gt_kernels import gt_frames,output_frames,frame_metrics
import argparse,concurrent.futures as cf,traceback

def score_id(sid,threads):
 items={i['id']:i for i in rows(PKG/'inputs.jsonl')};gts={i['id']:i for i in rows(PKG/'ground_truth.jsonl')}
 item=items[sid];gt=gts[sid];selected=[r for r in load_cases() if r['id']==sid]
 pending=[r for r in selected if not compatible(path_score('cpu',r),r)]
 if not pending:return
 size=(1280,720) if item['canvas_wh'][0]>item['canvas_wh'][1] else (720,1280)
 t=time.perf_counter();truth,gtpts=gt_frames(gt,size)
 for case in pending:
  try:
   assert sha(case['path'])==case['output_sha256']
   start=time.perf_counter();meta=read(Path(case['path']).with_suffix('.json'))
   frames,alignment=output_frames(case['path'],item,meta,case['model'],size)
   with cf.ThreadPoolExecutor(max_workers=threads) as pool:values=list(pool.map(frame_metrics,zip(truth,frames)))
   assert len(values)==case['num_frames'] and all(math.isfinite(v[0]) and (v[2] or math.isfinite(v[1])) for v in values)
   infinite=any(v[2] for v in values)
   mean={'ssim':math.fsum(v[0] for v in values)/len(values),'psnr_db':None if infinite else math.fsum(v[1] for v in values)/len(values),'psnr_infinite':infinite}
   atomic(path_score('cpu',case),receipt(case,status='complete',mean=mean,values=values,alignment=alignment,gt_pts=gtpts,num_frames=len(values),seconds=time.perf_counter()-start,process_pid=os.getpid(),threads=threads))
   print('CPU',case['model'],sid,flush=True)
  except Exception:failures('cpu',case,traceback.format_exc())
 return time.perf_counter()-t

def main():
 p=argparse.ArgumentParser();p.add_argument('--scope',choices=['login','node'],required=True);p.add_argument('--phase',choices=['smoke','full'],required=True);a=p.parse_args()
 if a.scope=='node':require_node()
 else:assert not os.environ.get('SLURM_JOB_ID')
 inputs=rows(PKG/'inputs.jsonl');ids=[i['id'] for n,i in enumerate(inputs) if (n%9==0)==(a.scope=='login') and (a.phase=='full' or i['id'] in SMOKE)]
 ids.sort(key=lambda sid:(sid not in SMOKE, next(n for n,x in enumerate(inputs) if x['id']==sid)))
 workers,threads=(2,2) if a.scope=='login' else (8,4)
 start=time.time()
 with cf.ProcessPoolExecutor(max_workers=workers) as pool:
  futures=[pool.submit(score_id,sid,threads) for sid in ids]
  for f in cf.as_completed(futures):f.result()
 selected=[r for r in load_cases() if r['id'] in ids]
 ok=all(compatible(path_score('cpu',r),r) for r in selected)
 atomic(ROOT/f'metadata/cpu_{a.scope}_{a.phase}.json',{'status':'complete' if ok else 'failed','sample_ids':ids,'outputs':len(selected),'workers':workers,'threads_each':threads,'seconds':time.time()-start,'pid':os.getpid()})
 if not ok:raise RuntimeError('CPU scores failed; inspect per-video records')
if __name__=='__main__':main()
