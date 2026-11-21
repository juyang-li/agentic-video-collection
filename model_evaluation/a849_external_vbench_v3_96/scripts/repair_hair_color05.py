"""GPU source-only repair for the thin frontal hairline; preserve small valid masks."""
import sys
from common import *

def main():
 require_job()
 import cv2,numpy as np
 from PIL import Image
 from media import write_preview
 sys.path.insert(0,str(ROOT.parent/'ltx25_vace_a100_20261111/scripts'))
 from human_parsing import HumanParser
 ready=read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json');parser=HumanParser(ready['human_parsing'])
 sid='hair_color_05';item=next(r for r in rows(ROOT/'metadata/inputs.jsonl') if r['id']==sid);out=ROOT/'conditions'/sid;before=read(out/'condition.json');atomic(out/'before_thin_hair_repair.json',before)
 old=ROOT.parent/'a849_external_vbench_v2_96';oldc=read(old/'conditions/hair_07/condition.json');px=read(ROOT/'inputs/prepared'/sid/'pixels.json')
 assert px['frame_sha256']==read(old/'inputs/prepared/hair_07/pixels.json')['frame_sha256']
 paths=[];overlays=[];counts=[];fallback=[]
 for i in range(49):
  frame=Image.open(ROOT/'inputs/prepared'/sid/f'{i:03d}.png').convert('RGB');hair=parser.segment(frame,[2]);face=parser.segment(frame,[11])>0
  m=cv2.dilate(hair*255,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)));m[face]=0
  # A narrow visible hairline need not cover100 pixels. Never turn a valid small mask into a SAM tracking failure.
  if not np.count_nonzero(m):
   base=np.asarray(Image.open(oldc['mask_frames'][i]));assert sha(oldc['mask_frames'][i])==oldc['mask_sha256'][i]
   for radius in [20,16,12,8,4,0]:
    candidate=cv2.erode(base,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)));candidate[face]=0
    if np.count_nonzero(candidate):m=candidate;fallback.append(dict(frame=i,erosion_radius=radius));break
  assert np.count_nonzero(m)>0,(sid,i)
  dst=out/f'mask_{i:03d}.png';Image.fromarray(m).save(dst);paths.append(dst);counts.append(int(np.count_nonzero(m)))
  rgb=np.asarray(frame).copy();sel=m>127;rgb[sel]=(rgb[sel]*.55+np.array([255,0,0])*.45).astype(np.uint8);overlays.append(Image.fromarray(rgb))
 write_preview(overlays,out/'overlay.mp4')
 atomic(out/'condition.json',dict(status='complete',id=sid,input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),mask_frames=[str(p) for p in paths],mask_sha256=[sha(p) for p in paths],mask_semantics={'255':'editable','0':'preserved'},mask_area_min=min(counts)/(frame.width*frame.height),mask_area_max=max(counts)/(frame.width*frame.height),overlay_video=str(out/'overlay.mp4'),adapted_prompt='A continuous video. '+item['vace_description']+'\nOriginal edit specification: '+item['prompt']['en'],spec=item['mask_spec'],source_pixels_sha256=sha(ROOT/'inputs/prepared'/sid/'pixels.json'),source_sha256=item['source']['sha256'],frame_sha256=px['frame_sha256'],parser=ready['human_parsing'],script_sha256=sha(__file__),repair=dict(method='Keep nonempty source-parsed hairline masks below100 pixels; do not force them through SAM2',per_frame_pixel_counts=counts,fallback=fallback,fallback_condition=str(old/'conditions/hair_07/condition.json'),no_gt=True),human_protocol_review='pending'))
 print(sid,'complete',min(counts),max(counts),'fallback',fallback,flush=True)
if __name__=='__main__':main()
