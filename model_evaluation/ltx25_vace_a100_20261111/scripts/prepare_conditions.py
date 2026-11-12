"""Source-only tracked unions, editable=255, preserve=0; official pose adapter."""
import argparse
import importlib
import sys
import time
import traceback
import types
from common import *


def pose_model(ready):
    # Import official pose module without importing unrelated annotator backends.
    directory=Path(ready['pose_code']['path']).parent
    package=types.ModuleType('a100_official_annotations');package.__path__=[str(directory)]
    sys.modules[package.__name__]=package
    module=importlib.import_module('a100_official_annotations.pose')
    # Keep the official detector/pose algorithm, with bounded ORT CPU pools.
    import onnxruntime as ort
    OriginalWholebody=module.Wholebody
    class BoundedWholebody(OriginalWholebody):
        def __init__(self,onnx_det,onnx_pose,device='cpu'):
            options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
            self.session_det=ort.InferenceSession(onnx_det,sess_options=options,providers=['CPUExecutionProvider'])
            self.session_pose=ort.InferenceSession(onnx_pose,sess_options=options,providers=['CPUExecutionProvider'])
    module.Wholebody=BoundedWholebody
    root=Path(ready['pose']['path'])/'pose'
    return module.PoseBodyFaceVideoAnnotator({'DETECTION_MODEL':str(root/'yolox_l.onnx'),'POSE_MODEL':str(root/'dw-ll_ucoco_384.onnx'),'RESIZE_SIZE':1024},device='cpu')


def prepare(shard,stage,only=None):
    require_job(exact=True)
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from media import video_frames,write_preview
    from vace_specs import spec
    ready=read(ROOT/'runtime/vace_ready.json')
    sys.path.insert(0,ready['sam2']['path'])
    sys.path.insert(0,str(ROOT.parent/'omni_baselines_20260926/vace'))
    from generate_vace_masks import MaskModels
    from native_worker import SMOKE,SECOND
    inputs=rows(ROOT/'metadata/inputs.jsonl')
    ids=SMOKE if stage=='smoke' else SECOND if stage=='second' else [r['id'] for r in inputs]
    selected=[r for r in inputs if r['id'] in ids][shard::4]
    if only:selected=[r for r in inputs if r['id'] in only]
    models=None;pose=None;parser=None
    for item in selected:
        out=ROOT/'conditions'/item['id'];out.mkdir(parents=True,exist_ok=True)
        manifest=out/'condition.json';s=spec(item)
        identity={'input_manifest_sha256':sha(ROOT/'metadata/inputs.jsonl'),'spec':s,'script_sha256':sha(__file__)}
        if manifest.exists() and read(manifest).get('status')=='complete':
            previous=read(manifest)['identity']
            assert previous['input_manifest_sha256']==identity['input_manifest_sha256'] and previous['spec']==identity['spec']
            continue
        state={'id':item['id'],'status':'running','identity':identity,'started_unix':time.time()}
        atomic(manifest,state)
        try:
            frames,pixel_audit=video_frames(item,item['source'])
            rgb=[np.asarray(f) for f in frames];bgr=[cv2.cvtColor(f,cv2.COLOR_RGB2BGR) for f in rgb]
            h,w=rgb[0].shape[:2]
            detection_audit={}
            if s.get('pose'):
                if pose is None:pose=pose_model(ready)
                motion=next(r for r in item['references'] if r['type']=='motion')
                motion_frames,motion_audit=video_frames(item,motion)
                poses=pose.forward(motion_frames)
                assert len(poses)==49 and min(np.count_nonzero(f) for f in poses)>20,'Pose missing in a frame'
                for i,f in enumerate(poses):Image.fromarray(f).save(out/f'pose_{i:03d}.png')
                pose_path=out/'pose.mp4';write_preview([Image.fromarray(f) for f in poses],pose_path,12)
                state['pose_video']=str(pose_path);state['motion_pixel_audit']=motion_audit
                state['pose_adapter']='official PoseBodyFaceVideoAnnotator; source frames 0/24/48 for appearance; pose mask all 255'
                masks=[np.full((h,w),255,np.uint8) for _ in rgb]
            elif s.get('full_frame'):
                masks=[np.full((h,w),255,np.uint8) for _ in rgb]
            else:
                if models is None:
                    models=MaskModels(Path(ready['dino']['path']),Path(ready['sam2']['checkpoint']))
                masks=[np.zeros((h,w),np.uint8) for _ in rgb]
                if s.get('parsing_labels'):
                    from human_parsing import HumanParser
                    if parser is None:parser=HumanParser(ready['human_parsing'])
                    if s.get('main_person'):
                        # Lock the target actor before other people enter; do not reselect the largest person per keyframe.
                        initial=parser.segment(frames[0],list(range(1,18)),main_person=True)
                        actor_masks=models.track(bgr,{0:initial})
                        keys={idx:(parser.segment(frames[idx],s['parsing_labels']) & actor_masks[idx]) for idx in (0,24,48)}
                    else:
                        keys={idx:parser.segment(frames[idx],s['parsing_labels']) for idx in (0,24,48)}
                    assert all(m.any() and not m.all() for m in keys.values()), 'Skin parsing empty/invalid'
                    masks=[(1-m if s.get('parsing_invert') else m)*255 for m in models.track(bgr,keys)]
                    state['human_parsing']={'repo_id':parser.record['repo_id'],'revision':parser.record['revision'],'labels':s['parsing_labels'],'names':[parser.record['id2label'][str(i)] for i in s['parsing_labels']],'main_person':s.get('main_person',False),'code_sha256':sha(ROOT/'scripts/human_parsing.py')}
                for object_index,box in enumerate(s.get('manual_objects',[])):
                    x1,y1,x2,y2=box
                    boxes=np.array([[x1*w,y1*h,x2*w,y2*h]],dtype=np.float32)
                    keys={}
                    for idx in (0,24,48):
                        keys[idx],_=models.segment_keyframe(bgr[idx],boxes,(0,0,1,1))
                    assert all(m.any() and not m.all() for m in keys.values())
                    masks=[np.maximum(m,t*255) for m,t in zip(masks,models.track(bgr,keys),strict=True)]
                    detection_audit['manual_'+str(object_index)]={'method':'explicit_source_ROI_seed_then_SAM2','box_normalized':box,'keyframes':[0,24,48],'provenance':'reviewed source 0/2/4s only; no GT read'}
                queries=[(q,False) for q in s['queries']]
                if s.get('background') and not s.get('parsing_labels'):queries.append((s['protect'],True))
                for qi,(query,invert) in enumerate(queries):
                    keys={};audit=[]
                    for idx in (0,24,48):
                        boxes,scores=models.detect(bgr[idx],query.strip().rstrip('.')+'.',12)
                        if len(boxes)==0:
                            raise RuntimeError(f'No detection for {query!r} at frame {idx}; manual condition repair needed')
                        m,method=models.segment_keyframe(bgr[idx],boxes,(0,0,1,1))
                        assert method=='grounding_dino+sam2','No silent full-frame fallback allowed'
                        assert m.any() and not m.all()
                        keys[idx]=m
                        audit.append({'frame':idx,'query':query,'boxes':boxes.tolist(),'scores':scores})
                    tracked=models.track(bgr,keys)
                    detection_audit[str(qi)]={'invert_to_background':invert,'keyframes':audit}
                    for i,m in enumerate(tracked):
                        if invert:m=1-m
                        masks[i]=np.maximum(masks[i],m*255)
                for box in s.get('boxes',[]):
                    x1,y1,x2,y2=box
                    for m in masks:m[round(y1*h):round(y2*h),round(x1*w):round(x2*w)]=255
                d=s['dilation_pixels']
                kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*d+1,2*d+1))
                masks=[cv2.dilate(m,kernel) for m in masks]
                if s.get('protect_source_face_neck'):
                    # Expand hairstyle contour without deleting the source facial identity.
                    for i,m in enumerate(masks):
                        protected=parser.segment(frames[i],[11])
                        m[protected>0]=0
                    state['hair_preservation']='Source parsed face/neck pixels never erased; contour expansion stays outside them'
                # Preserve the fixed letterbox, never hallucinate over padded bars.
                geometry=pixel_audit['geometry'];left,top,right,bottom=geometry['padding_ltrb']
                for m in masks:
                    if left:m[:,:left]=0
                    if top:m[:top]=0
                    if right:m[:,-right:]=0
                    if bottom:m[-bottom:]=0
            areas=[float((m>127).mean()) for m in masks]
            assert min(areas)>0 and (s.get('full_frame') or s.get('pose') or max(areas)<.995)
            overlays=[]
            for i,(f,m) in enumerate(zip(rgb,masks,strict=True)):
                Image.fromarray(m).save(out/f'mask_{i:03d}.png')
                tint=np.zeros_like(f);tint[:,:,0]=255
                alpha=(m>127)[:,:,None]*.45
                overlays.append(Image.fromarray((f*(1-alpha)+tint*alpha).astype(np.uint8)))
            overlay=out/'overlay.mp4';write_preview(overlays,overlay,12)
            mask_files=[out/f'mask_{i:03d}.png' for i in range(49)]
            state.update(status='complete',mask_semantics={'255':'editable','0':'preserved'},mask_frames=[str(p) for p in mask_files],mask_sha256=[sha(p) for p in mask_files],mask_area_min=min(areas),mask_area_max=max(areas),mask_area_mean=sum(areas)/49,overlay_video=str(overlay),detections=detection_audit,source_pixel_audit=pixel_audit,adapted_prompt=s['adapted_prompt'],original_instruction=item['prompt']['en'],human_protocol_review='pending',completed_unix=time.time())
        except Exception:
            state.update(status='failed',error=traceback.format_exc());traceback.print_exc()
        atomic(manifest,state)
        print(item['id'],state['status'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,choices=range(4),required=True);p.add_argument('--stage',choices=['smoke','second','all'],required=True);p.add_argument('--only',action='append');a=p.parse_args();prepare(a.shard,a.stage,a.only)
