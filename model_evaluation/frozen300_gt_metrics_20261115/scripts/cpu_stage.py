"""Login-only CPU preprocessing and SSIM/PSNR, local models first while remote downloads."""
from common import *
from evaluate import location,validate,gt_frames,output_frames,frame_metrics
import time,concurrent.futures as cf,fcntl

def main():
 if os.environ.get('SLURM_JOB_ID'):assert os.environ.get('SLURM_JOB_PARTITION')=='testqueue' and os.environ['SLURM_JOB_ID']=='202896'
 scope=os.environ.get('METRIC_SCOPE','all');assert scope in ['local','remote','all']
 lock=(ROOT/f'logs/cpu_stage_{scope}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 items=rows(PKG/'inputs.jsonl');gts={r['id']:r for r in rows(PKG/'ground_truth.jsonl')};byid={r['id']:r for r in items}
 order=SMOKE+[r['id'] for r in items if r['id'] not in SMOKE];digest=sha(ROOT/'metadata/protocol.json');count=0
 with cf.ThreadPoolExecutor(max_workers=int(os.environ.get("METRIC_CPU_WORKERS","4"))) as pool:
  for models in ([MODELS[:2]] if scope=='local' else [MODELS[2:]] if scope=='remote' else [MODELS[:2],MODELS[2:]]):
   if models==MODELS[2:]:
    while not (ROOT/'metadata/download.json').exists():time.sleep(30)
   for sid in order:
    item=byid[sid];gt=gts[sid];size=(1280,720) if item['canvas_wh'][0]>item['canvas_wh'][1] else (720,1280)
    assert sha(PKG/gt['target']['path'])==gt['target']['sha256'];truth,pts=gt_frames(gt,size)
    for model in models:
     p=ROOT/f'cpu_scores/{model}/{sid}.json';path,meta=location(model,sid);validate(item,model,path,meta)
     if p.exists():
      old=read(p);assert old['protocol_sha256']==digest and old['output_sha256']==meta['output_sha256'];count+=1;continue
     frames,alignment=output_frames(path,item,meta,model,size);values=list(pool.map(frame_metrics,zip(truth,frames)))
     atomic(p,{'id':sid,'model':model,'protocol_sha256':digest,'output_sha256':meta['output_sha256'],'gt_sha256':gt['target']['sha256'],'values':values,'alignment':alignment,'gt_pts':pts})
     count+=1;atomic(ROOT/f'metadata/cpu_progress_{scope}.json',{'status':'running','completed':count,'total':1200 if scope=='all' else 600,'model':model,'id':sid});print('CPU_COMPLETE',count,model,sid,flush=True)
 atomic(ROOT/f'metadata/cpu_progress_{scope}.json',{'status':'complete','completed':count})
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  atomic(ROOT/f'metadata/cpu_failure_{os.environ.get("METRIC_SCOPE","all")}.json',{'error':traceback.format_exc()});raise
