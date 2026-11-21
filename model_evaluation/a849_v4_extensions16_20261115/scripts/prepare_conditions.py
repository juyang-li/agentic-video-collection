"""Source-only masks; insertion ROIs run on login, parsing/tracking only on testqueue."""
import argparse,sys,traceback
from common import *

def run(shard,cpu=False,candidates=False):
 manifest=ROOT/'metadata'/('candidates.jsonl' if candidates else 'inputs.jsonl')
 if not cpu:require_job()
 import cv2,numpy as np
 from PIL import Image
 from media import write_preview
 parser=tracker=None
 ready=read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json')
 def get_parser():
  nonlocal parser
  if parser is None:
   sys.path.insert(0,str(ROOT.parent/'ltx25_vace_a100_20261111/scripts'))
   from human_parsing import HumanParser
   parser=HumanParser(ready['human_parsing'])
  return parser
 def get_tracker():
  nonlocal tracker
  if tracker is None:
   sys.path.insert(0,ready['sam2']['path']);sys.path.insert(0,str(ROOT.parent/'omni_baselines_20260926/vace'))
   from generate_vace_masks import MaskModels
   tracker=MaskModels(None,Path(ready['sam2']['checkpoint']),manual_boxes=True)
  return tracker
 def active_box(box,audit,w,h):
  l,t,r,b=audit['pixel_audit']['geometry']['padding_ltrb'];aw=w-l-r;ah=h-t-b
  x1,y1,x2,y2=box;return [l+x1*aw,t+y1*ah,l+x2*aw,t+y2*ah]
 items=rows(manifest)
 if candidates:items=[r for r in items if not r.get('old_id')]
 selected=[r for r in items if (r['mask_spec'].get('kind')=='insertion_roi')==cpu]
 if not cpu:selected=selected[shard::4]
 for item in selected:
  out=ROOT/'conditions'/item['id'];out.mkdir(parents=True,exist_ok=True);status=out/'condition.json'
  if status.exists() and read(status).get('status')=='complete':
   prior=read(status)
   if prior.get('input_manifest_sha256')==sha(manifest) and prior.get('spec')==item['mask_spec']:continue
  try:
   prepared=ROOT/'inputs/prepared'/item['id'];audit=read(prepared/'pixels.json')
   frames=[Image.open(prepared/f'{i:03d}.png').convert('RGB') for i in range(49)]
   w,h=frames[0].size;s=item['mask_spec'];kind=s.get('kind');masks=[];evidence={}
   if kind=='insertion_roi':
    x1,y1,x2,y2=map(round,active_box(s['source_normalized_box'],audit,w,h));m=np.zeros((h,w),np.uint8);m[y1:y2,x1:x2]=1
    masks=[m.copy() for _ in frames];evidence=dict(method='source-planned insertion region in active image; all time steps reviewed',source_normalized_box=s['source_normalized_box'])
   elif kind=='tracked_object':
    tr=get_tracker();box=np.array([active_box(s['source_normalized_box'],audit,w,h)],dtype=np.float32)
    bgr=[cv2.cvtColor(np.asarray(f),cv2.COLOR_RGB2BGR) for f in frames];keys={};methods={}
    for k in [0,24,48]:keys[k],methods[k]=tr.segment_keyframe(bgr[k],box,tuple(s['source_normalized_box']))
    masks=tr.track(bgr,keys);evidence=dict(method='SAM2 segmented object at0,24,48, propagated all49 source frames',keyframe_methods=methods,source_normalized_box=s['source_normalized_box'])
   elif kind=='face_relative_accessory':
    parser=get_parser();rois=[];face_boxes=[];previous_box=None
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    for f in frames:
     face=parser.segment(f,[11],main_person=True)>0
     ys,xs=np.nonzero(face);assert len(xs)>100
     detected=detector.detectMultiScale(cv2.cvtColor(np.asarray(f),cv2.COLOR_RGB2GRAY),scaleFactor=1.05,minNeighbors=4)
     if len(detected):
      fx,fy,fw,fh=map(int,max(detected,key=lambda b:int(b[2])*int(b[3])));previous_box=[fx,fy,fw,fh]
     elif previous_box is not None:fx,fy,fw,fh=previous_box
     else:
      fx,fy=xs.min(),ys.min();fw=xs.max()-fx+1;fh=round((ys.max()-fy+1)*.70)
     face_boxes.append([int(v) for v in [fx,fy,fw,fh]])
     a,b,c,d=s['relative_roi'];x1=max(0,round(fx+a*fw));x2=min(w,round(fx+c*fw));y1=max(0,round(fy+b*fh));y2=min(h,round(fy+d*fh));m=np.zeros((h,w),np.uint8);m[y1:y2,x1:x2]=1
     if s['accessory']=='earrings':
      m[:,round(fx+.14*fw):round(fx+.86*fw)]=0
      m[parser.segment(f,[2])>0]=0
     if s['accessory'] in ['necklace','scarf']:m[parser.segment(f,[14,15])>0]=0
     masks.append(m);rois.append([x1,y1,x2,y2])
    evidence=dict(method='per-frame source Haar face detection (excluding neck), parser-assisted occlusion and accessory-specific region',face_boxes=face_boxes,frame_rois=rois,accessory=s['accessory'])
   else:
    parser=get_parser();actors=None
    if s.get('actor_box'):
     tr=get_tracker();box=np.array([[s['actor_box'][i]*([w,h,w,h][i]) for i in range(4)]],dtype=np.float32)
     bgr=[cv2.cvtColor(np.asarray(f),cv2.COLOR_RGB2BGR) for f in frames]
     seed,method=tr.segment_keyframe(bgr[0],box,(0,0,1,1));actors=tr.track(bgr,{0:seed});evidence=dict(actor_tracking=True,seed_method=method)
    for i,f in enumerate(frames):
     m=parser.segment(f,s['parsing_labels'],main_person=s.get('main_person',False) and actors is None)
     if actors is not None:m &= actors[i]
     if s.get('invert'):m=1-m
     masks.append(m)
   if item['group']=='hair_color' and any(np.count_nonzero(m)<10 for m in masks):
    tr=get_tracker();bgr=[cv2.cvtColor(np.asarray(f),cv2.COLOR_RGB2BGR) for f in frames]
    keys={k:masks[k] for k in [0,12,24,36,48] if np.count_nonzero(masks[k])>=10}
    assert 0 in keys and len(keys)>=2,'No reliable hair tracking seeds'
    masks=tr.track(bgr,keys);evidence.update(hair_repair='SAM2 hair-only propagation where per-frame parsing was empty',hair_keyframes=list(keys))
   overlays=[];final=[]
   for i,(f,m) in enumerate(zip(frames,masks,strict=True)):
    d=s['dilation_pixels'];m=cv2.dilate((m>0).astype(np.uint8)*255,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*d+1,2*d+1)))
    if item['group'] in ['hair','hair_color']:m[get_parser().segment(f,[11])>0]=0
    left,top,right,bottom=audit['pixel_audit']['geometry']['padding_ltrb']
    if left:m[:,:left]=0
    if top:m[:top]=0
    if right:m[:,-right:]=0
    if bottom:m[-bottom:]=0
    assert np.count_nonzero(m)>100,f"Empty mask {item['id']} frame{i}"
    Image.fromarray(m).save(out/f'mask_{i:03d}.png');final.append(m)
    rgb=np.asarray(f).copy();sel=m>127;rgb[sel]=(rgb[sel]*.55+np.array([255,0,0])*.45).astype(np.uint8);overlays.append(Image.fromarray(rgb))
   write_preview(overlays,out/'overlay.mp4')
   paths=[out/f'mask_{i:03d}.png' for i in range(49)]
   atomic(status,dict(status='complete',id=item['id'],input_manifest_sha256=sha(manifest),candidate_staging=candidates,mask_frames=[str(p) for p in paths],mask_sha256=[sha(p) for p in paths],mask_semantics={'255':'editable','0':'preserved'},mask_area_min=min(float((m>127).mean()) for m in final),mask_area_max=max(float((m>127).mean()) for m in final),overlay_video=str(out/'overlay.mp4'),adapted_prompt='A continuous video. '+item['vace_description']+'\nOriginal edit specification: '+item['prompt']['en'],spec=s,source_pixels_sha256=sha(prepared/'pixels.json'),source_sha256=item['source']['sha256'],frame_sha256=audit['frame_sha256'],mask_evidence=evidence,script_sha256=sha(__file__),human_protocol_review='pending'))
   print(item['id'],'complete',flush=True)
  except Exception:
   atomic(status,dict(status='failed',id=item['id'],error=traceback.format_exc()));traceback.print_exc()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--cpu',action='store_true');p.add_argument('--candidates',action='store_true');a=p.parse_args();run(a.shard,a.cpu,a.candidates)
