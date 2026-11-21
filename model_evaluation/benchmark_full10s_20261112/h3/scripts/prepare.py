"""Lossless adapter from frozen full ten-second benchmark assets."""
from common import *
import hashlib,time
from concurrent.futures import ThreadPoolExecutor

def encode(item,record,role):
 from media import video_frames
 from PIL import Image,ImageOps
 import av,numpy as np
 assert sha(record['absolute_path'])==record['sha256']
 dst=ROOT/'inputs'/f'{item["id"]}_{role}.mp4';audit=ROOT/'metadata/input_audits'/f'{item["id"]}_{role}.json'
 w,h=item['canvas_wh'];canvas=[1344,768] if w>h else [768,1344]
 mapping=[min(120,(j+1)//2) for j in range(243)]
 identity={'original_sha256':record['sha256'],'frozen_manifest_sha256':sha(OLD/'metadata/inputs.jsonl'),'adapter_sha256':sha(__file__),'pixel_script_sha256':sha(ROOT/'scripts/media.py'),'canvas_wh':canvas,'fps':24,'frames':243,'index_mapping':mapping,'source_interval_seconds':[0,10],'endpoint_policy':'training t=10 endpoint; hold only native alignment tail to 243 frames'}
 if audit.exists():
  previous=read(audit);assert previous['identity']==identity and sha(dst)==previous['sha256'];return previous
 frames,pixels=video_frames(item,record);assert len(frames)==121
 converted=[]
 for f in frames:
  image=Image.new('RGB',tuple(canvas));im=ImageOps.contain(f,tuple(canvas),method=Image.Resampling.LANCZOS)
  image.paste(im,((canvas[0]-im.width)//2,(canvas[1]-im.height)//2));converted.append(image)
 hashes=[hashlib.sha256(np.asarray(f).tobytes()).hexdigest() for f in converted]
 tmp=dst.with_suffix('.part.mp4')
 with av.open(str(tmp),'w') as c:
  s=c.add_stream('libx264rgb',rate=24);s.width,s.height=canvas;s.pix_fmt='rgb24';s.thread_count=2;s.options={'crf':'0','preset':'fast'}
  for idx in mapping:
   for packet in s.encode(av.VideoFrame.from_image(converted[idx])):c.mux(packet)
  for packet in s.encode():c.mux(packet)
 with av.open(str(tmp)) as c:
  assert c.streams.video[0].average_rate==24 and not c.streams.audio
  count=0
  for f in c.decode(video=0):
   assert (f.width,f.height)==tuple(canvas)
   assert hashlib.sha256(f.to_ndarray(format='rgb24').tobytes()).hexdigest()==hashes[mapping[count]]
   count+=1
  assert count==243
 tmp.replace(dst)
 result={'status':'complete','identity':identity,'path':str(dst),'sha256':sha(dst),'frames_full_decoded':243,'original_pixel_audit':pixels}
 atomic(audit,result);return result

def prepare(item):
 source=encode(item,item['source'],'source');conditions=[{'type':'video','role':'reference','uri':Path(source['path']).as_uri(),'start_time_seconds':0.0}]
 definitions=['<Video 1> is the source video to edit. <Subject 1> is its target person.'];ni=0;nv=1;motion=[]
 for i,ref in enumerate(item['references']):
  assert sha(ref['absolute_path'])==ref['sha256']
  if ref['type']=='motion':
   a=encode(item,ref,f'motion_{i}');nv+=1;motion.append(a)
   conditions.append({'type':'video','role':'reference','uri':Path(a['path']).as_uri(),'start_time_seconds':0.0})
   definitions.append(f'<Video {nv}> is the full action reference. Transfer its action to <Subject 1>, preserving the source person appearance, scene and camera.')
  else:
   ni+=1;conditions.append({'type':'image','role':'reference','uri':Path(ref['absolute_path']).as_uri()})
   definitions.append(f'<Picture {ni}> is the supplied {ref["type"]} reference, view: {ref.get("view") or "unspecified"}. Transfer only the attributes requested in the original edit specification.')
 prompt='subject_definitions:\n'+'\n'.join(definitions)+'\n\nsummary:\nEdit <Video 1> according to the original edit specification.\n\nretention_analysis:\nPreserve source content, people, motion, camera, lighting and appearance except explicitly requested changes. Multiple reference views describe the same item. Preserve the ordering of all requested edits.\n\ndetailed_description:\nOriginal edit specification (complete):\n'+item['prompt']['en']+'\nPreserve the full source timeline from 0 to 10 seconds without time compression. Frames after 10 seconds hold the endpoint solely for native temporal alignment.\n\noverall_soundscape:\nNo dialogue requested. Audio is outside this visual evaluation.\n\nnon_diegetic_music:\nNone.'
 request={'model':'MiniMaxAI/MiniMax-H3','task':'ref2va','prompt':prompt,'conditions':conditions,'target':{'short_edge':768,'aspect_ratio':'16:9' if item['canvas_wh'][0]>item['canvas_wh'][1] else '9:16','duration_seconds':10.0},'num_outputs_per_prompt':1,'num_inference_steps':50,'flow_shift':12.0,'audio_flow_shift':3.0,'seed':42,'quality':'lossless','reference_resize_mode':'max'}
 atomic(ROOT/'metadata/requests'/f'{item["id"]}.json',request)
 print(item['id'],'prepared 10s',flush=True)
 return {**item,'prepared_source':{'path':source['path'],'sha256':source['sha256']},'prepared_motion':motion,'h3_canvas_wh':source['identity']['canvas_wh'],'h3_prompt':prompt,'request':request}
if __name__=='__main__':
 require_job();items=rows(OLD/'metadata/inputs.jsonl');assert len(items)==80
 with ThreadPoolExecutor(max_workers=2) as pool:output=list(pool.map(prepare,items))
 freeze_jsonl(ROOT/'metadata/inputs.jsonl',output)
 atomic(ROOT/'metadata/preflight.json',{'status':'passed','samples':80,'frames':243,'fps':24,'requested_duration_seconds':10,'native_duration_seconds':243/24,'no_gt':True,'context_ir':False})
