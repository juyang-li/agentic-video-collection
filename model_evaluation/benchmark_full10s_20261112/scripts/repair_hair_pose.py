"""Source-only per-frame face-landmark head crops; no target assets read."""
from common import *
require_job(exact=True)
import cv2,numpy as np,time,shutil,torch
from PIL import Image
from human_parsing import HumanParser
from prepare_conditions import pose_model
from media import video_frames,write_preview
from vace_specs import spec
ready=read(ROOT/'runtime/vace_ready.json');pose=pose_model(ready);parser=HumanParser(ready['human_parsing'],device='cuda')
item=next(i for i in rows(ROOT/'metadata/inputs.jsonl') if i['id']=='hair_07')
frames,audit=video_frames(item,item['source']);out=ROOT/'conditions'/item['id'];old=read(out/'condition.json');masks=[];details=[];gap=0
for i,f in enumerate(frames):
 rgb=np.asarray(f);h,w=rgb.shape[:2]
 points,scores,det=pose.pose_estimation(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
 assert len(points)>0,f'No source person at {i}'
 actor=int(np.argmax((scores[:,24:92]>.3).sum(axis=1)))
 valid=scores[actor,24:92]>.3;face=points[actor,24:92][valid]
 assert len(face)>=8,f'Insufficient source face landmarks at {i}: {len(face)}'
 lo=face.min(axis=0);hi=face.max(axis=0);fh=max(float(hi[1]-lo[1]),12.);fw=max(float(hi[0]-lo[0]),fh*.55)
 x1=max(0,round(lo[0]-.8*fw));x2=min(w,round(hi[0]+.8*fw));y1=max(0,round(lo[1]-.9*fh));y2=min(h,round(hi[1]+.05*fh))
 crop=f.crop((x1,y1,x2,y2));hair=parser.segment(crop,[2]).astype(np.uint8)*255
 hair=cv2.dilate(hair,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(65,65)))
 hair[parser.segment(crop,[11])>0]=0
 mask=np.zeros((h,w),np.uint8);mask[y1:y2,x1:x2]=hair
 flow_used=False
 if not mask.any():
  gap+=1;assert i>0 and gap<=3,f'Source crop parse gap too long at {i}'
  previous=cv2.cvtColor(np.asarray(frames[i-1]),cv2.COLOR_RGB2GRAY);current=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
  flow=cv2.calcOpticalFlowFarneback(current,previous,None,.5,5,25,5,7,1.5,0);yy,xx=np.mgrid[:h,:w].astype(np.float32)
  warped=cv2.remap(masks[-1],xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
  mask[y1:y2,x1:x2]=warped[y1:y2,x1:x2];flow_used=True
 else:gap=0
 assert mask.any() and (mask>127).mean()<.1
 masks.append(mask);details.append({'frame':i,'face_landmarks':len(face),'head_roi_xyxy':[x1,y1,x2,y2],'source_flow_gap':flow_used})
 if i%24==0:print('head-constrained source parsing',i,flush=True)
overlays=[]
for i,(f,m) in enumerate(zip(frames,masks,strict=True)):
 Image.fromarray(m).save(out/f'mask_{i:03d}.png');rgb=np.asarray(f);red=np.zeros_like(rgb);red[:,:,0]=255;a=(m>127)[:,:,None]*.45
 overlays.append(Image.fromarray((rgb*(1-a)+red*a).astype(np.uint8)))
write_preview(overlays,out/'overlay.mp4',12)
files=[out/f'mask_{i:03d}.png' for i in range(121)];areas=[float((m>127).mean()) for m in masks]
old.update(status='complete',identity={'input_manifest_sha256':sha(ROOT/'metadata/inputs.jsonl'),'spec':spec(item),'script_sha256':sha(__file__)},mask_frames=[str(p) for p in files],mask_sha256=[sha(p) for p in files],mask_area_min=min(areas),mask_area_max=max(areas),mask_area_mean=sum(areas)/121,overlay_video=str(out/'overlay.mp4'),source_pixel_audit=audit,adapted_prompt=spec(item)['adapted_prompt'],original_instruction=item['prompt']['en'],human_protocol_review='pending',completed_unix=time.time(),source_crop_repair={'method':'per-source-frame official DWPose facial-landmark head ROI then high-resolution semantic hair parsing; source face protected; flow only for <=3 missing parses and confined to current source head ROI','frames':details,'no_gt':True},hair_preservation='Per-frame source head ROI excludes torso; crop parsing protects face. Manual output quality pending.')
old.pop('error',None);old.pop('revision_reason',None);atomic(out/'condition.json',old)
print('REPAIRED',item['id'],'source flow frames',[d['frame'] for d in details if d['source_flow_gap']],flush=True)
