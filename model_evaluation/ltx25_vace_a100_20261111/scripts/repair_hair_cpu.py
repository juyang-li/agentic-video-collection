"""Source-only higher-resolution hair parsing in reviewed head ROIs; no GT access."""
from common import *
require_job(exact=True)
import shutil,time,cv2,numpy as np,torch
from PIL import Image
from human_parsing import HumanParser
from media import video_frames,write_preview
from vace_specs import spec
torch.set_num_threads(8)
ready=read(ROOT/'runtime/vace_ready.json'); parser=HumanParser(ready['human_parsing'],device='cpu')
rois={'hair_07':[.35,.07,.68,.30],'hair_color_07':[.39,.015,.60,.26],'hair_color_10':[.36,.015,.60,.44],'hair_color_06':[.17,.09,.83,.45]}
for item in rows(ROOT/'metadata/inputs.jsonl'):
 sid=item['id']
 if sid not in rois:continue
 out=ROOT/'conditions'/sid;old=read(out/'condition.json')
 archive=ROOT/'metadata/protocol_attempts/hair_parser_full_frame'/sid
 archive.parent.mkdir(parents=True,exist_ok=True)
 if not archive.exists():shutil.copytree(out,archive)
 frames,audit=video_frames(item,item['source']); masks=[]
 for f in frames:
  w,h=f.size;x1,y1,x2,y2=[round(v*s) for v,s in zip(rois[sid],[w,h,w,h])]
  crop=f.crop((x1,y1,x2,y2));m=parser.segment(crop,[2]);m=m.astype(np.uint8)*255
  d=32 if sid.startswith('hair_') and not sid.startswith('hair_color') else 6
  m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*d+1,2*d+1)))
  m[parser.segment(crop,[11])>0]=0
  mask=np.zeros((h,w),np.uint8);mask[y1:y2,x1:x2]=m;masks.append(mask)
 empty=[i for i,m in enumerate(masks) if not m.any()]
 if len(empty)>12:raise RuntimeError(f'{sid}: too many empty crop parses {empty}')
 valid=[i for i,m in enumerate(masks) if m.any()]
 for i in empty:
  masks[i]=masks[min(valid,key=lambda j:abs(i-j))].copy()
 overlays=[]
 for i,(f,m) in enumerate(zip(frames,masks)):
  Image.fromarray(m).save(out/f'mask_{i:03d}.png');rgb=np.asarray(f);tint=np.zeros_like(rgb);tint[:,:,0]=255;alpha=(m>127)[:,:,None]*.45
  overlays.append(Image.fromarray((rgb*(1-alpha)+tint*alpha).astype(np.uint8)))
 write_preview(overlays,out/'overlay.mp4',12)
 s=spec(item);files=[out/f'mask_{i:03d}.png' for i in range(49)];areas=[float((m>127).mean()) for m in masks]
 old.update(status='complete',identity={'input_manifest_sha256':sha(ROOT/'metadata/inputs.jsonl'),'spec':s,'script_sha256':sha(__file__)},mask_semantics={'255':'editable','0':'preserved'},mask_frames=[str(f) for f in files],mask_sha256=[sha(f) for f in files],mask_area_min=min(areas),mask_area_max=max(areas),mask_area_mean=sum(areas)/49,overlay_video=str(out/'overlay.mp4'),source_pixel_audit=audit,adapted_prompt=s['adapted_prompt'],original_instruction=item['prompt']['en'],human_protocol_review='pending',completed_unix=time.time(),source_crop_repair={'roi':rois[sid],'method':'source head ROI each-frame semantic hair parsing; no GT; source face parsed and protected; avoids small full-body resolution failure and SAM2 drift onto shirt'})
 old['source_crop_repair']['nearest_valid_source_mask_for_empty_frames']=empty
 old.pop('error',None);atomic(out/'condition.json',old);print(sid,'repaired',min(areas),max(areas),flush=True)
