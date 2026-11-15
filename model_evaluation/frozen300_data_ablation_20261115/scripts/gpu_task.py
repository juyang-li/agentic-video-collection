from common import *
import argparse,traceback

def lpips_task(cases,tag):
 import torch,lpips,numpy as np
 from gt_kernels import gt_frames,output_frames
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.reset_peak_memory_stats()
 start=time.perf_counter();network=lpips.LPIPS(net='alex',version='0.1').cuda().float().eval();torch.cuda.synchronize();load=time.perf_counter()-start
 x=torch.zeros((1,3,128,128),device='cuda');y=x+.1
 with torch.inference_mode():
  zero=network(x,x).item();forward=network(x,y).item();backward=network(y,x).item();batch=network(torch.cat([x,x]),torch.cat([y,y])).flatten()
 assert abs(zero)<1e-6 and forward>zero and abs(forward-backward)<1e-5 and bool((abs(batch-forward)<1e-5).all())
 atomic(ROOT/f'metadata/kernel_gpu/{tag}.json',{'status':'passed','identity':zero,'symmetric':True,'batch_consistent':True,'gpu':torch.cuda.get_device_name(0),'model_load_seconds':load,'torch':torch.__version__,'lpips_version':'0.1.4','dtype':'FP32','tf32':False})
 items={i['id']:i for i in rows(PKG/'inputs.jsonl')};gts={i['id']:i for i in rows(PKG/'ground_truth.jsonl')};last_sid=None;truth=None
 for case in cases:
  p=path_score('lpips',case)
  if compatible(p,case):continue
  try:
   t=time.perf_counter();item=items[case['id']];gt=gts[case['id']];size=(1280,720) if item['canvas_wh'][0]>item['canvas_wh'][1] else (720,1280)
   if last_sid!=case['id']:
    truth,gtpts=gt_frames(gt,size);last_sid=case['id']
   assert sha(case['path'])==case['output_sha256'] and sha(PKG/gt['target']['path'])==case['gt_sha256']
   frames,alignment=output_frames(case['path'],item,read(Path(case['path']).with_suffix('.json')),case['model'],size)
   values=[];pos=0;bs=4
   while pos<len(frames):
    x=y=None
    try:
     x=torch.from_numpy(np.stack(frames[pos:pos+bs])).permute(0,3,1,2).cuda().float()/127.5-1
     y=torch.from_numpy(np.stack(truth[pos:pos+bs])).permute(0,3,1,2).cuda().float()/127.5-1
     with torch.inference_mode():out=network(x,y).flatten().cpu().tolist()
     assert all(math.isfinite(v) for v in out);values.extend(out);pos+=len(out);del x,y
    except torch.cuda.OutOfMemoryError:
     if bs==1:raise
     del x,y;bs//=2;torch.cuda.empty_cache()
   torch.cuda.synchronize();assert len(values)==case['num_frames']
   atomic(p,receipt(case,status='complete',values=values,score=float(np.mean(values)),num_frames=len(values),alignment=alignment,gt_pts=gtpts,seconds=time.perf_counter()-t,batch_size=bs,gpu=torch.cuda.get_device_name(0),peak_allocated_bytes=torch.cuda.max_memory_allocated(),model_load_seconds=load))
   print('LPIPS',case['model'],case['id'],flush=True)
  except Exception:failures('lpips',case,traceback.format_exc())

def vbench_task(cases,dim,tag):
 import torch,gc
 from vbench import VBench
 torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
 pending=[r for r in cases if not compatible(path_score('vbench',r,dim),r)]
 if not pending:return
 inputs=ROOT/f'runtime/inputs/{tag}';results=ROOT/f'runtime/official/{tag}';inputs.mkdir(parents=True,exist_ok=True);results.mkdir(parents=True,exist_ok=True)
 mapping={}
 for case in pending:
  assert sha(case['path'])==case['output_sha256']
  h=case['output_sha256'];mapping.setdefault(h,[]).append(case)
  p=inputs/f'{h}.mp4'
  if not p.exists():p.symlink_to(Path(case['path']).resolve())
 bench=VBench(torch.device('cuda'),str(VBENV/'runtime/VBench/vbench/VBench_full_info.json'),str(results))
 def persist(data,elapsed):
  for v in data[dim][1]:
   h=Path(v['video_path']).stem;assert h in mapping
   raw=float(v['video_results']);assert math.isfinite(raw)
   for case in mapping[h]:atomic(path_score('vbench',case,dim),receipt(case,status='complete',dimension=dim,raw_score=raw,score=raw/100 if dim=='imaging_quality' else raw,batch_wall_seconds=elapsed,gpu=torch.cuda.get_device_name(0),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
 t=time.perf_counter()
 try:
  bench.evaluate(str(inputs),'batch',dimension_list=[dim],local=True,mode='custom_input',imaging_quality_preprocessing_mode='longer')
  torch.cuda.synchronize();persist(read(results/'batch_eval_results.json'),time.perf_counter()-t)
 except Exception:atomic(results/'batch_error.json',{'error':traceback.format_exc()})
 for h,group in mapping.items():
  if all(compatible(path_score('vbench',case,dim),case) for case in group):continue
  try:
   t=time.perf_counter();bench.evaluate(str(inputs/f'{h}.mp4'),h,dimension_list=[dim],local=True,mode='custom_input',imaging_quality_preprocessing_mode='longer')
   torch.cuda.synchronize();persist(read(results/f'{h}_eval_results.json'),time.perf_counter()-t)
   assert all(compatible(path_score('vbench',case,dim),case) for case in group),'Official evaluator omitted video'
  except Exception:
   error=traceback.format_exc()
   for case in group:failures('vbench',case,error,dim)
 del bench;gc.collect();torch.cuda.empty_cache()

def main():
 require_node();p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);a=p.parse_args();task=read(a.task)
 import torch
 assert torch.cuda.device_count()==1 and 'H200' in torch.cuda.get_device_name(0)
 cases=task['cases']
 if task['kind']=='lpips':lpips_task(cases,task['tag']);ok=all(compatible(path_score('lpips',r),r) for r in cases)
 else:vbench_task(cases,task['dimension'],task['tag']);ok=all(compatible(path_score('vbench',r,task['dimension']),r) for r in cases)
 if not ok:raise RuntimeError('Task has explicit failed/missing scores; inspect records before retry')
if __name__=='__main__':main()
