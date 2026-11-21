"""Repair source parser collapse between inspected hair keyframes before generation."""
from common import *
from refine_masks import save
import sys

def main():
    require_job();assert not (ROOT/'metadata/generation_gate.json').exists()
    import cv2,numpy as np
    from PIL import Image
    ready=read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json')
    sys.path.insert(0,ready['sam2']['path']);sys.path.insert(0,str(ROOT.parent/'omni_baselines_20260926/vace'))
    from generate_vace_masks import MaskModels
    model=MaskModels(None,Path(ready['sam2']['checkpoint']),manual_boxes=True)
    item=next(r for r in rows(ROOT/'metadata/inputs.jsonl')if r['id']=='hair_color_22')
    c=read(ROOT/'conditions/hair_color_22/condition.json');frames=[cv2.imread(str(ROOT/'inputs/prepared/hair_color_22'/f'{n:03}.png'))for n in range(49)]
    h,w=frames[0].shape[:2];keys={}
    for n in [0,12,24,48]:
        m=np.array(Image.open(c['mask_frames'][n]));m[round(.04*h):round(.36*h),round(.28*w):round(.62*w)]=0
        assert (m>127).sum()>30000,(n,(m>127).sum())
        keys[n]=(m>127).astype('uint8')
    tracked=model.track(frames,keys);masks=[]
    for m in tracked:
        m=m.astype('uint8')*255;m[round(.04*h):round(.36*h),round(.28*w):round(.62*w)]=0;masks.append(m)
    save(item,masks,dict(method='SAM2 source-only hair tracking between directly inspected reliable source hair keyframes; repair per-frame parser collapse at43–44; preserve source lens envelope',hair_keyframes=list(keys),source_only=True,no_gt=True,minimum_area_pixels=min(int((m>127).sum())for m in masks)),'hair_source_tracking')

if __name__=='__main__':main()
