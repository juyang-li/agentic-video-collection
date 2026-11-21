"""SOURCE hair semantics intersected with visible silver-white color; CPU-only."""
from common import *
require_job()
import os
cpus=sorted(os.sched_getaffinity(0));chosen=cpus[len(cpus)//2:];os.sched_setaffinity(0,chosen)
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,cv2,numpy as np,time
from PIL import Image
from human_parsing import HumanParser
from prepare_conditions import pose_model
from media import video_frames,write_preview
from vace_specs import spec
torch.set_num_threads(min(8,len(chosen)))
item=next(i for i in rows(ROOT/'metadata/inputs.jsonl') if i['id']=='hair_color_10')
assert item['source']['attributes']['hair_color_id']==7
assert not (ROOT/'outputs/vace/hair_color_10.json').exists(),'Do not change a condition already consumed by inference'
ready=read(ROOT/'runtime/vace_ready.json');pose=pose_model(ready);parser=HumanParser(ready['human_parsing'],device='cpu')
frames,audit=video_frames(item,item['source']);out=ROOT/'conditions/hair_color_10/semantic_white_cpu';out.mkdir(exist_ok=True)
old=read(out.parent/'condition.json');masks=[];records=[];gap=0
for idx,frame in enumerate(frames):
 rgb=np.asarray(frame);h,w=rgb.shape[:2];pts,scores,_=pose.pose_estimation(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));assert len(pts)>0
 actor=int(np.argmax((scores[:,24:92]>.3).sum(axis=1)));face=pts[actor,24:92][scores[actor,24:92]>.3];assert len(face)>=8
 lo=face.min(0);hi=face.max(0);fh=max(float(hi[1]-lo[1]),12.);fw=max(float(hi[0]-lo[0]),fh*.55)
 box=[max(0,round(lo[0]-.8*fw)),max(0,round(lo[1]-.9*fh)),min(w,round(hi[0]+.8*fw)),min(h,round(hi[1]+.05*fh))];x1,y1,x2,y2=box
 crop=frame.crop(tuple(box));hair=parser.segment(crop,[2]).astype(np.uint8)*255
 hint=np.zeros((h,w),np.uint8);hint[y1:y2,x1:x2]=cv2.dilate(hair,np.ones((7,7),np.uint8))
 hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);white=((hsv[:,:,1]<110)&(hsv[:,:,2]>90)).astype(np.uint8)*255
 mask=cv2.bitwise_and(hint,white);used_flow=False
 if not mask.any():
  gap+=1;assert idx and gap<=3,f'Missing source hair semantics {idx}'
  gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);prev=cv2.cvtColor(np.asarray(frames[idx-1]),cv2.COLOR_RGB2GRAY)
  flow=cv2.calcOpticalFlowFarneback(gray,prev,None,.5,5,25,5,7,1.5,0);yy,xx=np.mgrid[:h,:w].astype(np.float32)
  mask=cv2.bitwise_and(cv2.remap(masks[-1],xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_NEAREST),white);used_flow=True
 else:gap=0
 mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)));mask=cv2.bitwise_and(mask,white)
 valid=np.zeros((h,w),np.uint8);valid[y1:y2,x1:x2]=255;mask=cv2.bitwise_and(mask,valid)
 assert mask.any() and (mask>127).mean()<.1
 masks.append(mask);records.append({'frame':idx,'head_roi':box,'semantic_hair_pixels':int((hair>0).sum()),'source_flow_gap':used_flow})
 if idx%24==0:print('CPU source hair refinement',idx,flush=True)
overlays=[]
for idx,(frame,mask) in enumerate(zip(frames,masks,strict=True)):
 Image.fromarray(mask).save(out/f'mask_{idx:03d}.png');rgb=np.asarray(frame);red=np.zeros_like(rgb);red[:,:,0]=255;a=(mask>127)[:,:,None]*.45;overlays.append(Image.fromarray((rgb*(1-a)+red*a).astype(np.uint8)))
write_preview(overlays,out/'overlay.mp4',12)
files=[out/f'mask_{i:03d}.png' for i in range(121)];areas=[float((m>127).mean()) for m in masks]
assert not (ROOT/'outputs/vace/hair_color_10.json').exists()
atomic(ROOT/'metadata/protocol_attempts/white_color_only_condition.json',old)
old.update(status='complete',identity={'input_manifest_sha256':sha(ROOT/'metadata/inputs.jsonl'),'spec':spec(item),'script_sha256':sha(__file__)},mask_frames=[str(p) for p in files],mask_sha256=[sha(p) for p in files],overlay_video=str(out/'overlay.mp4'),source_pixel_audit=audit,mask_area_min=min(areas),mask_area_max=max(areas),mask_area_mean=sum(areas)/121,completed_unix=time.time(),source_crop_repair={'method':'per-frame source head ROI, source hair semantic labels intersected with visible silver-white color; CPU-only; no GT; excludes adjacent white architecture','cpu_affinity':chosen,'frames':records},hair_preservation='Source-only semantic hair plus color constraint; brown skin excluded; no human output-quality approval')
old.pop('error',None);atomic(out.parent/'condition.json',old);print('SOURCE_SEMANTIC_WHITE_MASK_COMPLETE',flush=True)
