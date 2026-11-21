"""CPU Ref2VA projection: fixed17n+5 native alignment, same frozen0–4s content."""
import concurrent.futures,importlib.util,time
from common import *
from contracts import digest
H3=ROOT.parent/'minimax_h3_ref2va90_local_20261112'
VENDOR=H3/'runtime/sglang'
TIME_CODE=VENDOR/'python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/time_request.py'
QWEN=Path('/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/qwen3.8')
PDD=QWEN/'runtime/acceleration_loras/h3_pdd_native'
GUIDE=QWEN/'reports/accelerated_editors_portable_20261114'

def prompt(item):
 definitions=['<Video 1> is the source video to edit.']
 for k,r in enumerate(item['references'],1):definitions.append(f'<Picture {k}> is the supplied {r["type"]} reference, view: {r.get("view","front")}. Use only the attributes requested in the original edit specification.')
 target=item.get('edit_target')
 if target:definitions.append('Target in source: '+str(target))
 return ('subject_definitions:\n'+'\n'.join(definitions)+'\n\nsummary:\nCreate the edited version of <Video 1> according to the original edit specification.\n\nretention_analysis:\nPreserve all source content, other people, facial identity, movement, camera, timing, lighting and appearance except the explicitly requested changes. A reference supplies only the requested edit attributes.\n\ndetailed_description:\nOriginal edit specification (complete):\n'+item['prompt']['en']+'\nProduce one continuous full-frame video, following the source timeline from0 to4seconds without time compression. Any source frames after4seconds are endpoint padding required by native temporal alignment, not a new action or shot.\n\noverall_soundscape:\nNo dialogue requested. Audio is outside this visual evaluation.\n\nnon_diegetic_music:\nNone.')

def prepare_source(spec):
 import av,numpy as np
 from PIL import Image,ImageOps
 key,item,canvas,nframes=spec
 dest=ROOT/'inputs/h3'/f'{key}.mp4';receipt=dest.with_suffix('.json');p=read(ROOT/'inputs/prepared'/item['id']/'pixels.json')
 identity=dict(source_sha256=item['source']['sha256'],source_frame_sha256=p['frame_sha256'],canvas_wh=canvas,native_fps=24,native_frames=nframes,index_mapping=[min(48,(j+1)//2)for j in range(nframes)],time_code_sha256=sha(TIME_CODE),mapping='README12fps canonical content ->24fps nearest endpoint mapping; only0–4s used in scoring')
 if receipt.exists():
  r=read(receipt);assert r['identity']==identity and sha(dest)==r['sha256'];return key,r
 frames=[];pixel_hashes=[]
 for n in range(49):
  path=ROOT/'inputs/prepared'/item['id']/f'{n:03d}.png';assert sha(path)==p['frame_sha256'][n]
  frame=Image.open(path).convert('RGB');out=Image.new('RGB',tuple(canvas));im=ImageOps.contain(frame,tuple(canvas),method=Image.Resampling.LANCZOS);out.paste(im,((out.width-im.width)//2,(out.height-im.height)//2));frames.append(out);pixel_hashes.append(hashlib.sha256(np.asarray(out).tobytes()).hexdigest())
 dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.part.mp4')
 with av.open(str(tmp),'w')as c:
  stream=c.add_stream('libx264rgb',rate=24);stream.width,stream.height=canvas;stream.pix_fmt='rgb24';stream.thread_count=2;stream.options={'crf':'0','preset':'fast'}
  for n in identity['index_mapping']:
   for packet in stream.encode(av.VideoFrame.from_image(frames[n])):c.mux(packet)
  for packet in stream.encode():c.mux(packet)
 with av.open(str(tmp))as c:
  assert c.streams.video[0].average_rate==24 and not c.streams.audio;decoded=0
  for n,frame in enumerate(c.decode(video=0)):
   assert [frame.width,frame.height]==canvas and hashlib.sha256(frame.to_ndarray(format='rgb24').tobytes()).hexdigest()==pixel_hashes[identity['index_mapping'][n]];decoded+=1
 assert decoded==nframes;tmp.replace(dest)
 r=dict(status='complete',identity=identity,path=str(dest),sha256=sha(dest),full_decoded_frames=decoded,source_resized_canonical_pixel_sha256=pixel_hashes)
 atomic(receipt,r);print('H3 source prepared',item['id'],nframes,flush=True);return key,r

def main():
 gate=read(ROOT/'metadata/generation_gate.json');assert gate['status']=='approved'and gate['manifest_sha256']==sha(ROOT/'metadata/inputs.jsonl')
 spec=importlib.util.spec_from_file_location('h3_fixed_temporal',TIME_CODE);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 nframes=mod.minimax_h3_align_frame_count(4*24);assert nframes==107 and mod.minimax_h3_align_frame_count(10*24)==243
 config=read(PDD/'pdd_config.json');assert config==dict(num_steps=32,block_size=4,fused_steps=8,num_inference_steps=8,video_shift=12.,audio_shift=3.)
 lora=QWEN/'runtime/acceleration_loras/alibaba-pai/MiniMax-H3-Ref2VA-Acc-8Step.safetensors';assert sha(lora)=='111c82e669f6e20e628228172edf39395f1a9fc3ad049793895e542c0f55b18c'
 import subprocess
 assert subprocess.check_output(['git','-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip()=='fa179312e94c764f8637724ea2a364d302888906'
 index=read(PDD/'transformer/model.safetensors.index.json');shards=sorted(set(index['weight_map'].values()));assert len(shards)==13 and all((PDD/'transformer'/p).is_file()for p in shards)
 atomic(ROOT/'metadata/h3_runtime_contract.json',dict(model='MiniMaxAI/MiniMax-H3',base_revision='0919fb684102ef115126612f3fbb84d1027e7317',partition='Ref2VA',pdd_lora_path=str(lora),pdd_lora_revision='335001fb9e5455d68a0caa18ec2e319072150328',pdd_lora_sha256=sha(lora),pdd_directory=str(PDD),pdd_config=config,pdd_config_sha256=sha(PDD/'pdd_config.json'),fused_heads_sha256=sha(PDD/'pdd_fused_heads.safetensors'),transformer_index_sha256=sha(PDD/'transformer/model.safetensors.index.json'),transformer_shards=[dict(name=n,bytes=(PDD/'transformer'/n).stat().st_size)for n in shards],sglang_revision='fa179312e94c764f8637724ea2a364d302888906',time_code_sha256=sha(TIME_CODE),guide=str(GUIDE/'README.md'),guide_sha256=sha(GUIDE/'README.md'),num_inference_steps=8,seed=42,flow_shift=12.,audio_flow_shift=3.,tp_size=2,ulysses_degree=2,num_gpus=4,performance_mode='speed',torch_compile=False,warmup='off',batching_max_size=1,quality='lossless',reference_resize_mode='max',native_frames=nframes,native_fps=24,comparison_timestamps=[n/12 for n in range(49)],comparison_generated_indices=[2*n for n in range(49)],no_negative_prompt=True,no_vace_style_cfg=True,no_quantization_or_approximate_cache=True))
 items=rows(ROOT/'metadata/inputs.jsonl');jobs={};keys={}
 for item in items:
  canvas=[1344,768]if item['canvas_wh'][0]>item['canvas_wh'][1]else[768,1344];p=read(ROOT/'inputs/prepared'/item['id']/'pixels.json');key=digest(dict(frames=p['frame_sha256'],canvas=canvas,native_frames=nframes));keys[item['id']]=key;jobs.setdefault(key,(key,item,canvas,nframes))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:prepared=dict(pool.map(prepare_source,jobs.values()))
 requests=[]
 for item in items:
  r=prepared[keys[item['id']]];conditions=[dict(type='video',role='reference',uri=Path(r['path']).as_uri(),start_time_seconds=0.)]
  for ref in item['references']:
   assert sha(ref['absolute_path'])==ref['sha256'];conditions.append(dict(type='image',role='reference',uri=Path(ref['absolute_path']).as_uri()))
  body=dict(model='MiniMaxAI/MiniMax-H3',task='ref2va',prompt=prompt(item),conditions=conditions,target=dict(short_edge=768,aspect_ratio='16:9'if item['canvas_wh'][0]>item['canvas_wh'][1]else'9:16',duration_seconds=4.),num_outputs_per_prompt=1,num_inference_steps=8,flow_shift=12.,audio_flow_shift=3.,seed=42,quality='lossless',reference_resize_mode='max')
  assert item['prompt']['en']in body['prompt']and len(conditions)==1+len(item['references'])
  atomic(ROOT/'metadata/h3_requests'/f"{item['id']}.json",body);requests.append(dict(id=item['id'],source_audit=r['path'].replace('.mp4','.json'),source_sha256=r['sha256'],request_sha256=digest(body)))
 atomic(ROOT/'metadata/h3_preflight.json',dict(status='passed',tasks=104,unique_source_adapters=len(prepared),native_frames=107,native_fps=24,source_time_window=[0,4],comparison_frames=49,comparison_fps=12,input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),runtime_contract_sha256=sha(ROOT/'metadata/h3_runtime_contract.json'),requests=requests,original_source_pixels_unchanged=True,no_gt_or_masked_source_in_request=True))
 print('H3 CPU preflight complete:104 requests,107 native source frames; comparison0–4s only.',flush=True)
if __name__=='__main__':main()
