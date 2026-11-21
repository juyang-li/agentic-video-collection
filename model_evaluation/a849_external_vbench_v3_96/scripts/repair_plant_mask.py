"""Track the plant within a source-reviewed corridor, preserving the vase silhouette."""
import sys
from common import *

def main():
 require_job()
 import cv2,numpy as np
 from PIL import Image
 from media import write_preview
 sid='object_removal_06';item=next(r for r in rows(ROOT/'metadata/inputs.jsonl') if r['id']==sid);out=ROOT/'conditions'/sid;c=read(out/'condition.json');atomic(out/'before_vase_exclusion.json',c)
 ready=read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json');sys.path.insert(0,ready['sam2']['path']);sys.path.insert(0,str(ROOT.parent/'omni_baselines_20260926/vace'))
 from generate_vace_masks import MaskModels
 tracker=MaskModels(None,Path(ready['sam2']['checkpoint']),manual_boxes=True)
 frames=[Image.open(ROOT/'inputs/prepared'/sid/f'{i:03d}.png').convert('RGB') for i in range(49)];w,h=frames[0].size;px=read(ROOT/'inputs/prepared'/sid/'pixels.json');l,t,r,b=px['pixel_audit']['geometry']['padding_ltrb'];aw=w-l-r;ah=h-t-b
 polygons=[[(.28,.07),(.47,.07),(.51,.32),(.45,.43),(.35,.53),(.27,.48),(.25,.31)],[(.30,.40),(.34,.46),(.29,.59),(.24,.83),(.18,.83),(.20,.71),(.23,.59)]]
 allowed=np.zeros((h,w),np.uint8)
 for polygon in polygons:
  points=np.array([[round(l+x*aw),round(t+y*ah)] for x,y in polygon],dtype=np.int32);cv2.fillPoly(allowed,[points],1)
 seeds={i:((np.asarray(Image.open(c['mask_frames'][i]))>127)&(allowed>0)).astype(np.uint8) for i in [0,24,48]}
 tracked=tracker.track([cv2.cvtColor(np.asarray(f),cv2.COLOR_RGB2BGR) for f in frames],seeds)
 paths=[];overlays=[]
 for i,(frame,m) in enumerate(zip(frames,tracked,strict=True)):
  m=((m>0)&(allowed>0)).astype(np.uint8)*255
  assert np.count_nonzero(m)>500
  dst=out/f'mask_{i:03d}.png';Image.fromarray(m).save(dst);paths.append(dst)
  rgb=np.asarray(frame).copy();sel=m>127;rgb[sel]=(rgb[sel]*.55+np.array([255,0,0])*.45).astype(np.uint8);overlays.append(Image.fromarray(rgb))
 write_preview(overlays,out/'overlay.mp4')
 c.update(mask_frames=[str(p) for p in paths],mask_sha256=[sha(p) for p in paths],repair=dict(method='Source-only SAM2 plant track constrained to reviewed plant-head/stem corridor; glass vase silhouette is preserved',polygons_source_normalized=polygons,keyframes=[0,24,48],script_sha256=sha(__file__),note='Stems inside the vase need local reconstruction; plant shadows remain a limitation of the object-only mask.'),human_protocol_review='pending')
 atomic(out/'condition.json',c);print(sid,'plant corridor and vase exclusion complete',flush=True)
if __name__=='__main__':main()
