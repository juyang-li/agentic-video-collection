"""Inference projection and lossless 24fps adapter; GT never enters requests."""
from common import *
import time,hashlib,argparse
from concurrent.futures import ThreadPoolExecutor

def prompt(item):
 definitions=['<Video 1> is the source video to edit. <Subject 1> is the target person in that source.']
 for i,r in enumerate(item['references'],1):
  definitions.append(f'<Picture {i}> is the supplied {r["type"]} reference, view: {r.get("view") or "unspecified"}. Use only the attributes requested in the original edit specification. Multiple views describe the same requested reference item.')
 original=item['prompt']['en']
 return ('subject_definitions:\n'+'\n'.join(definitions)+'\n\nsummary:\nCreate the edited version of <Video 1> according to the original edit specification.\n\nretention_analysis:\nPreserve all source content, people, movement, camera, timing, lighting and appearance except the changes explicitly requested below. A reference is not a new scene or a new camera unless that is the requested edit. Preserve the original ordering of all requested edits.\n\ndetailed_description:\nOriginal edit specification (complete):\n'+original+'\nKeep the source timeline from 0 to 4 seconds without time compression. Frames after 4 seconds are a held endpoint for native temporal alignment; do not introduce a new shot or additional action there.\n\noverall_soundscape:\nNo dialogue is requested. Audio is not part of this visual editing evaluation.\n\nnon_diegetic_music:\nNone.')

def prepare_one(item):
 from media import video_frames
 from PIL import Image,ImageOps
 import numpy as np,av
 sid=item['id'];w,h=item['canvas_wh'];canvas=[1344,768] if w>h else [768,1344]
 dst=ROOT/'inputs'/f'{sid}_source.mp4';status=ROOT/'metadata/input_audits'/f'{sid}.json'
 identity={'source_sha256':item['source']['sha256'],'input_manifest_sha256':sha(OLD/'metadata/inputs.jsonl'),'adapter_sha256':sha(__file__),'media_sha256':sha(ROOT/'scripts/media.py'),'canvas_wh':canvas,'fps':24,'frames':107,'index_mapping':[min(48,(j+1)//2) for j in range(107)],'tail_policy':'hold t=4 source frame; no material after t=4'}
 source=dict(item['source'])
 if not Path(source['absolute_path']).is_file():
  relocation=next(r for r in read(ROOT/'metadata/source_relocations.json') if r['id']==sid)
  assert relocation['sha256']==source['sha256']
  source['absolute_path']=relocation['resolved_path']
 assert sha(source['absolute_path'])==source['sha256']
 for r in item['references']:assert sha(r['absolute_path'])==r['sha256']
 old=read(status) if status.exists() else None
 if old:
  assert {k:v for k,v in old['identity'].items() if k!='adapter_sha256'}=={k:v for k,v in identity.items() if k!='adapter_sha256'} and sha(dst)==old['sha256']
  if old['identity']==identity:return old
 t=time.monotonic();frames,pixel=video_frames(item,source);assert len(frames)==49
 converted=[]
 for f in frames:
  image=Image.new('RGB',tuple(canvas));im=ImageOps.contain(f,tuple(canvas),method=Image.Resampling.LANCZOS)
  image.paste(im,((canvas[0]-im.width)//2,(canvas[1]-im.height)//2));converted.append(image)
 hashes=[hashlib.sha256(np.asarray(f).tobytes()).hexdigest() for f in converted]
 tmp=dst if old else dst.with_suffix('.part.mp4')
 if not old:
  with av.open(str(tmp),'w') as c:
   s=c.add_stream('libx264rgb',rate=24);s.width,s.height=canvas;s.pix_fmt='rgb24';s.thread_count=2;s.options={'crf':'0','preset':'fast'}
   for idx in identity['index_mapping']:
    for packet in s.encode(av.VideoFrame.from_image(converted[idx])):c.mux(packet)
   for packet in s.encode():c.mux(packet)
 with av.open(str(tmp)) as c:
  assert c.streams.video[0].average_rate==24 and len(c.streams.audio)==0
  n=0
  for f in c.decode(video=0):
   assert (f.width,f.height)==tuple(canvas)
   assert hashlib.sha256(f.to_ndarray(format='rgb24').tobytes()).hexdigest()==hashes[identity['index_mapping'][n]],(sid,n,'lossless pixel mismatch')
   n+=1
  assert n==107
 if not old:tmp.replace(dst)
 result={'status':'complete','identity':identity,'sha256':sha(dst),'path':str(dst),'original_pixel_audit':pixel,'frames_full_decoded':107,'seconds':time.monotonic()-t}
 result['source_resolved_path']=source['absolute_path']
 if old and old['identity']!=identity:
  atomic(ROOT/'metadata/input_audit_history'/f'{sid}_{old["identity"]["adapter_sha256"]}.json',old)
  result['reuse_validation']='all 107 decoded frames rechecked against current source adapter'
 atomic(status,result);print(sid,'prepared',flush=True);return result

def main(limit):
 require_job();items=[r for r in rows(OLD/'metadata/inputs.jsonl') if r['group']!='action']
 assert len(items)==90 and {r['group'] for r in items}==set(GROUPS)
 frozen=[r for r in rows(OLD/'metadata/selection.jsonl') if r['group']!='action'];assert [r['id'] for r in items]==[r['id'] for r in frozen]
 freeze_jsonl(ROOT/'metadata/selection.jsonl',frozen)
 if limit:items=[next(r for r in items if r['id']==sid) for sid in ['hair_color_01','single_piece_01','hair_01','site_01']][:limit]
 with ThreadPoolExecutor(max_workers=4) as pool:audits=list(pool.map(prepare_one,items))
 output=[]
 for r,a in zip(items,audits,strict=True):
  conditions=[{'type':'video','role':'reference','uri':Path(a['path']).as_uri(),'start_time_seconds':0.0}]+[{'type':'image','role':'reference','uri':Path(v['absolute_path']).as_uri()} for v in r['references']]
  p=prompt(r);request={'model':'MiniMaxAI/MiniMax-H3','task':'ref2va','prompt':p,'conditions':conditions,'target':{'short_edge':768,'aspect_ratio':'16:9' if a['identity']['canvas_wh'][0]>a['identity']['canvas_wh'][1] else '9:16','duration_seconds':4.0},'num_outputs_per_prompt':1,'num_inference_steps':50,'flow_shift':12.0,'audio_flow_shift':3.0,'seed':42,'quality':'lossless','reference_resize_mode':'max'}
  output.append({**r,'h3_canvas_wh':a['identity']['canvas_wh'],'h3_prompt':p,'prepared_source':{'path':a['path'],'sha256':a['sha256']},'request':request})
  atomic(ROOT/'metadata/requests'/f'{r["id"]}.json',request)
 freeze_jsonl(ROOT/'metadata'/('pilot_inputs.jsonl' if limit else 'inputs.jsonl'),output)
 atomic(ROOT/'metadata'/('pilot_prepared.json' if limit else 'preflight.json'),{'status':'passed','samples':len(output),'historical_manifest_sha256':sha(OLD/'metadata/inputs.jsonl'),'no_gt_in_requests':True,'frames':107,'fps':24,'native_endpoint_mapping':True,'comparison_interval_seconds':[0,4]})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);a=p.parse_args();main(a.limit)
