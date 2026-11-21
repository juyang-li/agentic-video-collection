"""Official multi-reference API inputs; frozen source/motion plus static reference sheet."""
from common import *
from PIL import Image,ImageOps
import av,math,numpy as np
require_job()
inputs=rows(BENCH/'metadata/inputs.jsonl');h3={i['id']:i for i in rows(BENCH/'h3/metadata/inputs.jsonl')};result=[]
for item in inputs:
 other=h3[item['id']];assert all(other[k]==v for k,v in item.items())
 source=other['prepared_source'];assert sha(source['path'])==source['sha256']
 w,h=other['h3_canvas_wh'];refs=item['references'];pictures=[r for r in refs if r['type']!='motion']
 sheet=Image.new('RGB',(w,h));cols=max(1,len(pictures));description=[]
 if pictures:
  for i,ref in enumerate(pictures):
   assert sha(ref['absolute_path'])==ref['sha256']
   with Image.open(ref['absolute_path']) as f:im=ImageOps.contain(f.convert('RGB'),(w//cols,h),Image.Resampling.LANCZOS)
   sheet.paste(im,(i*(w//cols)+(w//cols-im.width)//2,(h-im.height)//2))
   description.append(f'Panel {i+1} shows the supplied {ref["type"]} reference, {ref.get("view") or "unspecified"} view; transfer only the attributes specified below.')
 else:
  with av.open(source['path']) as c:sheet=next(c.decode(video=0)).to_image().convert('RGB')
  description=['The sheet shows the SOURCE appearance and setting to preserve, except changes requested below.']
 path=ROOT/'inputs'/f'{item["id"]}_sheet.png';sheet.save(path)
 video=path.with_suffix('.mp4');temp=video.with_suffix('.part.mp4');expected=hashlib.sha256(np.asarray(sheet).tobytes()).hexdigest()
 with av.open(str(temp),'w') as c:
  s=c.add_stream('libx264rgb',rate=24);s.width=w;s.height=h;s.pix_fmt='rgb24';s.thread_count=2;s.options={'crf':'0','preset':'fast'}
  for _ in range(241):
   for packet in s.encode(av.VideoFrame.from_image(sheet)):c.mux(packet)
  for packet in s.encode():c.mux(packet)
 with av.open(str(temp)) as c:
  n=0
  for frame in c.decode(video=0):
   assert hashlib.sha256(frame.to_ndarray(format='rgb24').tobytes()).hexdigest()==expected;n+=1
  assert n==241
 temp.replace(video)
 guides=[{'path':source['path'],'sha256':source['sha256'],'strength':1.,'role':'complete_10s_source'}, {'path':str(video),'sha256':sha(video),'strength':1.,'role':'static_reference_sheet'}]
 for motion in other['prepared_motion']:
  assert sha(motion['path'])==motion['sha256'];guides.append({'path':motion['path'],'sha256':motion['sha256'],'strength':1.,'role':'complete_10s_motion_reference'})
 prompt='Reference sheet: '+' '.join(description)+' The other conditioning video is the full ten-second SOURCE video. Preserve its person, movement, scene, camera and timing except the requested edit. '+('A third conditioning video specifies the action to transfer. ' if other['prepared_motion'] else '')+'\n\nGenerated video: '+item['prompt']['en']+' Preserve the full 0–10 second SOURCE timeline without time compression. No cuts or extra shots.'
 result.append({**item,'iclora_canvas_wh':[w,h],'guides':guides,'sheet_path':str(path),'prompt_adapter':prompt})
 print(item['id'],'IC-LoRA input ready',flush=True)
jsonl(ROOT/'metadata/inputs.jsonl',result)
atomic(ROOT/'metadata/preflight.json',{'status':'passed','samples':len(result),'frames':241,'fps':24,'timeline_span_seconds':10,'no_gt':True,'source_policy':'full frozen 10s source, official video_conditioning API; no keyframe-only substitution','extrapolation':'Ingredients trained on one static sheet at 121 frames/24fps; full source, multiple guides and 10s are experimental extensions'})
