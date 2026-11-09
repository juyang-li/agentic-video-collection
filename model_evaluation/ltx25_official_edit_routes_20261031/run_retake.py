from pathlib import Path
import argparse,json,time,hashlib,os,traceback
R=Path(__file__).resolve().parent

def save(p,d):
 tmp=p.with_suffix('.part');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['distilled','dev'],default='distilled');a=p.parse_args()
 sid='retake_'+a.model;out=R/f'outputs/generated/{sid}.mp4';record=R/f'outputs/records/{sid}.json';source=R/'assets/appearance_002_source49.mp4'
 prompt='A cinematic close-up of the same short-haired person wearing large opaque black sunglasses with black frames. The black sunglasses clearly cover both eyes. The person wears the same light-colored coat and looks around in the same dim outdoor setting, with natural expression and movement.'
 d=dict(sample_id=sid,route='official_native_RetakePipeline',status='running',model=a.model,source_video=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),prompt=prompt,reference_images=[],adapters=[],start_time=1/3,end_time=5/3,seed=42,regenerate_video=True,regenerate_audio=False,enhance_prompt=False,source_frames=49,fps=24,output=str(out),started_at=time.time(),job_id=os.environ.get('SLURM_JOB_ID'))
 save(record,d)
 try:
  import torch
  from ltx_pipelines.retake import RetakePipeline
  from ltx_pipelines.utils.model_paths import ModelPaths
  from ltx_pipelines.utils.constants import detect_params
  from ltx_pipelines.utils.types import OffloadMode
  from ltx_pipelines.utils.media_io import encode_video
  from ltx_core.model.video_vae import get_video_chunks_number
  cache=Path(os.environ['LTX_LOCAL_MODELS']);transformer=cache/f'diffusion_models/ltx-2.5-22b-{a.model}-transformer-bf16.safetensors'
  paths=ModelPaths.from_split(transformer_path=str(transformer),text_encoder_path=str(cache/'text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors'),video_vae_path=str(cache/'vae/ltx-2.5-video-vae-conv-bf16.safetensors'),audio_vae_path=str(cache/'vae/ltx-2.5-audio-vae-bf16.safetensors'))
  parameters=detect_params(str(transformer));d.update(num_inference_steps=8 if a.model=='distilled' else parameters.num_inference_steps,video_guidance=str(parameters.video_guider_params));save(record,d)
  with torch.inference_mode():
   pipe=RetakePipeline(model_paths=paths,loras=[],distilled=a.model=='distilled',device=torch.device('cuda'),offload_mode=OffloadMode.NONE)
   result=pipe(video_path=str(source),prompt=prompt,start_time=d['start_time'],end_time=d['end_time'],seed=42,negative_prompt='low quality, blurry, distorted face, deformed, flickering',num_inference_steps=parameters.num_inference_steps,video_guider_params=parameters.video_guider_params,audio_guider_params=parameters.audio_guider_params,regenerate_video=True,regenerate_audio=False,enhance_prompt=False)
   # Official encode sink, including lazy video decoding; silent export only.
   encode_video(video=result.video,fps=24,audio=None,output_path=str(out),video_chunks_number=get_video_chunks_number(result.num_frames,result.tiling_config),crf=18,thread_count=2)
  import av
  with av.open(str(out)) as c:
   st=c.streams.video[0];count=sum(1 for _ in c.decode(st));d['media']=dict(frames=count,width=st.width,height=st.height,fps=float(st.average_rate))
  d.update(status='generated',seconds=time.time()-d['started_at'],output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),finished_at=time.time());save(record,d);print(json.dumps(d),flush=True)
 except Exception as e:
  d.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),finished_at=time.time());save(record,d);raise
if __name__=='__main__':main()
