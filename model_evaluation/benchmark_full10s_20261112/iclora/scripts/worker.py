"""Official IC-LoRA pipeline, strict frozen-input resume and actual full decode."""
from common import *
import argparse,sys,time,traceback
sys.path.insert(0,str(NATIVE/'scripts/common'))
from project_layout import add_script_paths
add_script_paths()
PILOT=['identity_01','hair_color_01','action_01']

def pipeline(device):
 import torch
 from ltx_core.loader import LoraPathStrengthAndSDOps,LTXV_LORA_COMFY_RENAMING_MAP
 from ltx_pipelines.ic_lora import ICLoraPipeline,ICLoraStageConfig
 from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
 from ltx_pipelines.utils.model_paths import ModelPaths
 old=BENCH.with_name('ltx25_official_repro_20261031')
 weights=read(old/'metadata/run_config.json')['models'];paths={r['file'].split('/')[0]:r['local'] for r in weights}
 # Both VAE entries share the directory; select the component explicitly.
 def component(prefix):return next(r['local'] for r in weights if r['file'].startswith(prefix))
 for r in weights:assert Path(r['local']).is_file() and Path(r['local']).stat().st_size==r['bytes']
 ready=read(ROOT/'metadata/download_complete.json');assert ready['status']=='passed' and sha(ready['path'])==ready['sha256']
 stages=(ICLoraStageConfig(resolution_divisor=1,apply_ic_lora=True,sigmas=tuple(DISTILLED_SIGMA_VALUES)),)
 model_paths=ModelPaths.from_split(transformer_path=component('diffusion_models/'),text_encoder_path=component('text_encoders/'),video_vae_path=component('vae/ltx-2.5-video'),audio_vae_path=component('vae/ltx-2.5-audio'))
 pipe=ICLoraPipeline(model_paths=model_paths,spatial_upsampler_path=None,loras=[LoraPathStrengthAndSDOps(ready['path'],1.,LTXV_LORA_COMFY_RENAMING_MAP)],device=device,quantization=None,stages=stages)
 return pipe,weights,list(DISTILLED_SIGMA_VALUES)

def main(args):
 require_job();import torch,av
 if args.init_only:
  p,w,s=pipeline(torch.device('cpu'))
  for r in w:assert sha(r['local'])==r['sha256']
  atomic(ROOT/'metadata/pipeline_preflight.json',{'status':'passed','official_pipeline':'ltx_pipelines.ic_lora.ICLoraPipeline','weights':w,'sigmas':s,'actual_updates':len(s)-1,'native_pipeline_sha256':sha(NATIVE/'runtime/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py'),'unused_spatial_upsampler':None})
  print('OFFICIAL_PIPELINE_CPU_PREFLIGHT_PASSED',flush=True);return
 assert torch.cuda.device_count()==1 and 'H200' in torch.cuda.get_device_name(0)
 items=rows(ROOT/'metadata/inputs.jsonl');byid={i['id']:i for i in items}
 ids=PILOT if args.stage=='pilot' else [g+'_01' for g in GROUPS if g not in ['coinve','tripvvt']] if args.stage=='first' else list(byid)
 selected=[byid[sid] for sid in ids][args.shard::args.shards]
 if not selected:return
 audit=read(ROOT/'metadata/pipeline_preflight.json');assert audit['status']=='passed'
 pipe,weights,sigmas=pipeline(torch.device('cuda:0'))
 from ltx_pipelines.utils.media_io import encode_video
 for item in selected:
  sid=item['id'];out=ROOT/'outputs'/f'{sid}.mp4';status=out.with_suffix('.json')
  for guide in item['guides']:assert sha(guide['path'])==guide['sha256']
  contract={'inputs_sha256':sha(ROOT/'metadata/inputs.jsonl'),'worker_sha256':sha(__file__),'pipeline_preflight_sha256':sha(ROOT/'metadata/pipeline_preflight.json'),'adapter_sha256':read(ROOT/'metadata/download_complete.json')['sha256'],'seed':42,'frames':241,'fps':24,'canvas_wh':item['iclora_canvas_wh'],'sigmas':sigmas,'lora_strength':1.,'official_reference_positioning':True,'guide_sha256s':[g['sha256'] for g in item['guides']]}
  if status.exists() and read(status)['status']=='complete':
   prior=read(status);assert prior['contract']==contract and sha(out)==prior['output_sha256'];continue
  record={'id':sid,'status':'running','contract':contract,'started_unix':time.time(),'job_id':os.environ['SLURM_JOB_ID'],'source_and_motion_full10s':True,'no_gt':True}
  atomic(status,record)
  try:
   torch.cuda.reset_peak_memory_stats();start=time.monotonic();w,h=item['iclora_canvas_wh']
   with torch.inference_mode():
    result=pipe(prompt=item['prompt_adapter'],seed=42,height=h,width=w,num_frames=241,frame_rate=24,images=[],video_conditioning=[(g['path'],g['strength']) for g in item['guides']],enhance_prompt=False,enhance_static_cache=False,tiling_config=None)
    assert result.num_frames==241
    tmp=out.with_suffix('.part.mp4')
    encode_video(result.video,fps=24,audio=None,output_path=str(tmp),video_chunks_number=1,thread_count=2)
   torch.cuda.synchronize()
   with av.open(str(tmp)) as c:
    assert c.streams.video[0].average_rate==24;n=0
    for f in c.decode(video=0):assert (f.width,f.height)==(w,h);n+=1
    assert n==241
   tmp.replace(out);record.update(status='complete',seconds=time.monotonic()-start,full_decode_frames=n,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),output_sha256=sha(out),completed_unix=time.time());atomic(status,record)
   print(sid,'complete',record['seconds'],flush=True)
  except Exception:
   record.update(status='failed',error=traceback.format_exc(),completed_unix=time.time());atomic(status,record);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--init-only',action='store_true');p.add_argument('--stage',choices=['pilot','first','all'],default='pilot');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);main(p.parse_args())
