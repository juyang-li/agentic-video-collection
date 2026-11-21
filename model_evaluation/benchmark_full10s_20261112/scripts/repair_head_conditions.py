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
import argparse
cli=argparse.ArgumentParser();cli.add_argument('--only',action='append');args=cli.parse_args()
ids=args.only or ['hair_05','hair_07','hair_color_01','hair_color_07']
items=[i for i in rows(ROOT/'metadata/inputs.jsonl') if i['id'] in ids]
for item in items:
 frames,audit=video_frames(item,item['source']);out=ROOT/'conditions'/item['id'];old=read(out/'condition.json');masks=[];details=[];gap=0
 archive=ROOT/'metadata/protocol_attempts/before_face_hull'/item['id'];archive.parent.mkdir(parents=True,exist_ok=True)
 if not archive.exists():shutil.copytree(out,archive)
 for i,f in enumerate(frames):
  rgb=np.asarray(f);h,w=rgb.shape[:2]
  points,scores,det=pose.pose_estimation(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
  assert len(points)>0,f'No source person at {i}'
  actor=int(np.argmax((scores[:,24:92]>.3).sum(axis=1)))
  valid=scores[actor,24:92]>.3;face=points[actor,24:92][valid]
  if len(face)>=8:
   lo=face.min(axis=0);hi=face.max(axis=0);fh=max(float(hi[1]-lo[1]),12.);fw=max(float(hi[0]-lo[0]),fh*.55);head_basis='face_landmarks'
  else:
   head_ids=[0,14,15,16,17];head=points[actor,head_ids][scores[actor,head_ids]>.3]
   assert len(head)>=1,f'No source head landmarks at {i}'
   c=np.median(head,axis=0);shoulder=np.linalg.norm(points[actor,2]-points[actor,5]);fh=max(float(shoulder*.4),16.);fw=fh*.8
   lo=c-np.array([fw*.5,fh*.2]);hi=c+np.array([fw*.5,fh*.6]);head_basis='body_head_landmarks_for_occluded_face' 
  x1=max(0,round(lo[0]-.8*fw));x2=min(w,round(hi[0]+.8*fw));y1=max(0,round(lo[1]-.9*fh));y2=min(h,round(hi[1]+.05*fh))
  crop=f.crop((x1,y1,x2,y2));hair=parser.segment(crop,[2]).astype(np.uint8)*255
  d=6 if item['group']=='hair_color' else 32
  hair=cv2.dilate(hair,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*d+1,2*d+1)))
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
  if len(face)>=8:
   face_protection=np.zeros((h,w),np.uint8)
   cv2.fillConvexPoly(face_protection,cv2.convexHull(np.round(face).astype(np.int32)),255)
   face_protection=cv2.dilate(face_protection,np.ones((3,3),np.uint8));mask[face_protection>0]=0
  color_refinement=False
  if item['id']=='hair_color_10':
   assert item['source']['attributes']['hair_color_id']==7
   hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
   white=((hsv[:,:,1]<95)&(hsv[:,:,2]>120)).astype(np.uint8)*255
   allowed=np.zeros((h,w),np.uint8);allowed[y1:y2,x1:x2]=255
   white=cv2.bitwise_and(white,allowed)
   count,labels,stats,centroids=cv2.connectedComponentsWithStats(white,8)
   choices=[j for j in range(1,count) if stats[j,cv2.CC_STAT_AREA]>=5]
   assert choices,f'No visible SOURCE white hair component at {i}'
   expected=np.array([(lo[0]+hi[0])/2,lo[1]-.1*fh])
   chosen=min(choices,key=lambda j:float(np.linalg.norm(centroids[j]-expected)))
   mask=(labels==chosen).astype(np.uint8)*255
   mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
   skin_safe=((hsv[:,:,1]<110)&(hsv[:,:,2]>95)).astype(np.uint8)*255
   mask=cv2.bitwise_and(cv2.bitwise_and(mask,allowed),skin_safe)
   color_refinement=True
  assert mask.any() and (mask>127).mean()<.1,f'Empty after face protection: {item["id"]}:{i}' 
  masks.append(mask);details.append({'frame':i,'face_landmarks':len(face),'head_basis':head_basis,'face_geometry_protected':len(face)>=8 and item['id']!='hair_color_10','head_roi_xyxy':[x1,y1,x2,y2],'source_flow_gap':flow_used,'source_silver_white_refinement':color_refinement})
  if i%24==0:print('head-constrained source parsing',i,flush=True)
 overlays=[]
 for i,(f,m) in enumerate(zip(frames,masks,strict=True)):
  Image.fromarray(m).save(out/f'mask_{i:03d}.png');rgb=np.asarray(f);red=np.zeros_like(rgb);red[:,:,0]=255;a=(m>127)[:,:,None]*.45
  overlays.append(Image.fromarray((rgb*(1-a)+red*a).astype(np.uint8)))
 write_preview(overlays,out/'overlay.mp4',12)
 files=[out/f'mask_{i:03d}.png' for i in range(121)];areas=[float((m>127).mean()) for m in masks]
 old.update(status='complete',identity={'input_manifest_sha256':sha(ROOT/'metadata/inputs.jsonl'),'spec':spec(item),'script_sha256':sha(__file__)},mask_frames=[str(p) for p in files],mask_sha256=[sha(p) for p in files],mask_area_min=min(areas),mask_area_max=max(areas),mask_area_mean=sum(areas)/121,overlay_video=str(out/'overlay.mp4'),source_pixel_audit=audit,adapted_prompt=spec(item)['adapted_prompt'],original_instruction=item['prompt']['en'],human_protocol_review='pending',completed_unix=time.time(),source_crop_repair={'method':'per-source-frame official DWPose head ROI, source hair parsing, semantic face protection plus facial-landmark convex hull; flow only for <=3 missing parses inside current head ROI','frames':details,'no_gt':True},hair_preservation='Per-frame source head ROI excludes torso; crop parsing protects face. Manual output quality pending.')
 old.pop('error',None);old.pop('revision_reason',None);atomic(out/'condition.json',old)
 print('REPAIRED',item['id'],'source flow frames',[d['frame'] for d in details if d['source_flow_gap']],flush=True)
