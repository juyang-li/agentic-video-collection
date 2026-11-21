"""Source-only higher-resolution hair parsing in reviewed head ROIs; no GT access."""
from common import *
require_job(exact=True)
import shutil,time,cv2,numpy as np,torch
from PIL import Image
from human_parsing import HumanParser
from media import video_frames,write_preview
from vace_specs import spec
torch.set_num_threads(2)
ready=read(ROOT/'runtime/vace_ready.json'); parser=HumanParser(ready['human_parsing'],device='cuda')
rois={'hair_05':[.35,0,.70,.55],'hair_07':[.25,0,.80,.40]}
for item in rows(ROOT/'metadata/inputs.jsonl'):
 sid=item['id']
 if sid not in rois:continue
 out=ROOT/'conditions'/sid;old=read(out/'condition.json')
 if old['status']=='complete' and old.get('source_crop_repair'):continue
 archive=ROOT/'metadata/protocol_attempts/hair_parser_full_frame'/sid
 archive.parent.mkdir(parents=True,exist_ok=True)
 if not archive.exists():shutil.copytree(out,archive)
 frames,audit=video_frames(item,item['source']); masks=[];retry_frames=[]
 for f in frames:
  w,h=f.size;x1,y1,x2,y2=[round(v*s) for v,s in zip(rois[sid],[w,h,w,h])]
  crop=f.crop((x1,y1,x2,y2));m=parser.segment(crop,[2]);m=m.astype(np.uint8)*255
  d=32 if sid.startswith('hair_') and not sid.startswith('hair_color') else 6
  m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*d+1,2*d+1)))
  m[parser.segment(crop,[11])>0]=0
  if not m.any() and sid=='hair_07':
   x1,y1,x2,y2=[round(v*sz) for v,sz in zip([.35,.07,.68,.30],[w,h,w,h])]
   crop=f.crop((x1,y1,x2,y2));m=parser.segment(crop,[2]).astype(np.uint8)*255
   m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*d+1,2*d+1)))
   m[parser.segment(crop,[11])>0]=0
   retry_frames.append(len(masks))
  mask=np.zeros((h,w),np.uint8);mask[y1:y2,x1:x2]=m;masks.append(mask)
 empty=[i for i,m in enumerate(masks) if not m.any()]
 print(sid,'empty crop parses',empty,flush=True)
 flow_audit=[]
 for i in empty:
  assert sid=='hair_07' and i in (31,32) and masks[i-1].any(),f'Unreviewed empty frame {sid}:{i}'
  current=cv2.cvtColor(np.asarray(frames[i]),cv2.COLOR_RGB2GRAY)
  previous=cv2.cvtColor(np.asarray(frames[i-1]),cv2.COLOR_RGB2GRAY)
  flow=cv2.calcOpticalFlowFarneback(current,previous,None,.5,5,25,5,7,1.5,0)
  assert np.isfinite(flow).all()
  yy,xx=np.mgrid[:h,:w].astype(np.float32)
  transported=cv2.remap(masks[i-1],xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
  assert transported.any() and (transported>0).mean()<.1
  masks[i]=transported
  flow_audit.append({'frame':i,'previous_source_frame':i-1,'method':'current-to-previous source-only Farneback flow, nearest binary-mask remap','max_flow_pixels':float(np.linalg.norm(flow,axis=2).max())})
 overlays=[]
 for i,(f,m) in enumerate(zip(frames,masks)):
  Image.fromarray(m).save(out/f'mask_{i:03d}.png');rgb=np.asarray(f);tint=np.zeros_like(rgb);tint[:,:,0]=255;alpha=(m>127)[:,:,None]*.45
  overlays.append(Image.fromarray((rgb*(1-alpha)+tint*alpha).astype(np.uint8)))
 write_preview(overlays,out/'overlay.mp4',12)
 s=spec(item);files=[out/f'mask_{i:03d}.png' for i in range(121)];areas=[float((m>127).mean()) for m in masks]
 old.update(status='complete',identity={'input_manifest_sha256':sha(ROOT/'metadata/inputs.jsonl'),'spec':s,'script_sha256':sha(__file__)},mask_semantics={'255':'editable','0':'preserved'},mask_frames=[str(f) for f in files],mask_sha256=[sha(f) for f in files],mask_area_min=min(areas),mask_area_max=max(areas),mask_area_mean=sum(areas)/121,overlay_video=str(out/'overlay.mp4'),source_pixel_audit=audit,adapted_prompt=s['adapted_prompt'],original_instruction=item['prompt']['en'],human_protocol_review='pending',completed_unix=time.time(),source_crop_repair={'roi':rois[sid],'method':'source head ROI each-frame semantic hair parsing; no GT; source face parsed and protected; avoids small full-body resolution failure and SAM2 drift onto shirt'})
 old['source_crop_repair']['initial_empty_frames']=empty
 old['source_crop_repair']['flow_transport_for_reviewed_two_frame_gap']=flow_audit
 old['source_crop_repair']['final_empty_frames']=[]
 old['hair_preservation']='Source crop hair parsing with face protection; reviewed two-frame parse gap uses source optical-flow transport of protected hair mask. Human quality acceptance pending.'
 old['source_crop_repair']['tighter_head_roi_retry_frames']=retry_frames
 old['source_crop_repair']['tighter_head_roi']=[.35,.07,.68,.30]
 old['source_crop_repair']['source_only_visual_inspection']='metadata/hair_source_probe.jpg; broader head ROI accommodates full 10s camera and movement'
 old.pop('error',None);atomic(out/'condition.json',old);print(sid,'repaired',min(areas),max(areas),flush=True)
