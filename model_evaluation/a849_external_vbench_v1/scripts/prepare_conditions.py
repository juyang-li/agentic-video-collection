"""Source-only per-frame human parsing; SAM2 actor tracking for multi-person clips."""
import argparse
import sys
import traceback
from common import *

def run(stage,shard):
    require_job()
    import cv2
    import numpy as np
    from PIL import Image
    from media import write_preview
    sys.path.insert(0,str(ROOT.parent/'ltx25_vace_a100_20261111/scripts'))
    from human_parsing import HumanParser
    ready=read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json')
    parser=HumanParser(ready['human_parsing']);tracker=None
    items=rows(ROOT/'metadata/inputs.jsonl');smoke=read(ROOT/'metadata/freeze.json')['smoke']
    selected=[r for r in items if stage=='all' or r['id'] in smoke][shard::4]
    for item in selected:
        out=ROOT/'conditions'/item['id'];out.mkdir(parents=True,exist_ok=True);status=out/'condition.json'
        if status.exists() and read(status).get('status')=='complete':continue
        try:
            prepared=ROOT/'inputs/prepared'/item['id'];audit=read(prepared/'pixels.json')
            frames=[Image.open(prepared/f'{i:03d}.png').convert('RGB') for i in range(49)]
            s=item['mask_spec'];actors=None
            if s.get('actor_box'):
                if tracker is None:
                    sys.path.insert(0,ready['sam2']['path']);sys.path.insert(0,str(ROOT.parent/'omni_baselines_20260926/vace'))
                    from generate_vace_masks import MaskModels
                    tracker=MaskModels(Path(ready['dino']['path']),Path(ready['sam2']['checkpoint']))
                w,h=frames[0].size;x1,y1,x2,y2=s['actor_box']
                boxes=np.array([[x1*w,y1*h,x2*w,y2*h]],dtype=np.float32)
                seed,method=tracker.segment_keyframe(cv2.cvtColor(np.asarray(frames[0]),cv2.COLOR_RGB2BGR),boxes,(0,0,1,1))
                actors=tracker.track([cv2.cvtColor(np.asarray(f),cv2.COLOR_RGB2BGR) for f in frames],{0:seed})
            masks=[];overlays=[]
            for i,f in enumerate(frames):
                m=parser.segment(f,s['parsing_labels'],main_person=s.get('main_person',False) and actors is None)
                if actors is not None:m &= actors[i]
                if s['invert']:m=1-m
                d=s['dilation_pixels'];kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*d+1,2*d+1));m=cv2.dilate(m*255,kernel)
                if item['group']=='hair':m[parser.segment(f,[11])>0]=0
                left,top,right,bottom=audit['pixel_audit']['geometry']['padding_ltrb']
                if left:m[:,:left]=0
                if top:m[:top]=0
                if right:m[:,-right:]=0
                if bottom:m[-bottom:]=0
                assert np.count_nonzero(m)>100,f'Empty mask {item["id"]} frame{i}'
                masks.append(m);Image.fromarray(m).save(out/f'mask_{i:03d}.png')
                rgb=np.asarray(f).copy();editable=m>127;rgb[editable]=(rgb[editable]*.55+np.array([255,0,0])*.45).astype(np.uint8);overlays.append(Image.fromarray(rgb))
            write_preview(overlays,out/'overlay.mp4')
            paths=[out/f'mask_{i:03d}.png' for i in range(49)]
            atomic(status,dict(status='complete',id=item['id'],input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),mask_frames=[str(p) for p in paths],mask_sha256=[sha(p) for p in paths],mask_semantics={'255':'editable','0':'preserved'},mask_area_min=min(float((m>127).mean()) for m in masks),mask_area_max=max(float((m>127).mean()) for m in masks),overlay_video=str(out/'overlay.mp4'),adapted_prompt='A continuous video. '+item['vace_description']+'\nOriginal edit specification: '+item['prompt']['en'],parser=ready['human_parsing'],actor_tracking=bool(actors is not None),spec=s,source_pixels_sha256=sha(prepared/'pixels.json'),script_sha256=sha(__file__),human_protocol_review='pending'))
            print(item['id'],'complete',flush=True)
        except Exception:
            atomic(status,dict(status='failed',id=item['id'],error=traceback.format_exc()));traceback.print_exc()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','all'],required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args();run(a.stage,a.shard)
