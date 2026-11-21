"""Pre-inference source-only mask repairs after direct temporal inspection."""
import sys
from common import *

if __name__=='__main__':
    require_job()
    import cv2
    import numpy as np
    from PIL import Image
    from media import write_preview
    ready=read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json')
    sys.path.insert(0,ready['sam2']['path']);sys.path.insert(0,str(ROOT.parent/'omni_baselines_20260926/vace'))
    from generate_vace_masks import MaskModels
    tracker=MaskModels(None,Path(ready['sam2']['checkpoint']),manual_boxes=True)
    for sid in ['native_04','tripvvt_04']:
        item=next(r for r in rows(ROOT/'metadata/inputs.jsonl') if r['id']==sid)
        assert all(not (ROOT/'outputs'/m/f'{sid}.json').exists() for m in ['ltx_base','a849','vace'])
        out=ROOT/'conditions'/sid;old=read(out/'condition.json');atomic(out/'before_repair.json',old)
        frames=[Image.open(ROOT/'inputs/prepared'/sid/f'{i:03d}.png').convert('RGB') for i in range(49)]
        if sid=='native_04':
            masks=[]
            for i,f in enumerate(frames):
                m=np.asarray(Image.open(out/f'mask_{i:03d}.png')).copy();w,h=f.size
                # The gray beard extends below the parser's skin class. The
                # stationary lower-face ROI is reviewed on source 0..4s.
                m[round(.48*h):round(.81*h),round(.385*w):round(.65*w)]=255
                masks.append(m)
            repair={'method':'Union existing face mask with manually reviewed lower-face beard ROI','source_normalized_roi':[.385,.48,.65,.81],'reason':'Human parser excludes beard fibers from face label11; original instruction removes all beard.'}
        else:
            keys={i:(np.asarray(Image.open(out/f'mask_{i:03d}.png'))>127).astype(np.uint8) for i in [0,24,46]}
            masks=[m*255 for m in tracker.track([cv2.cvtColor(np.asarray(f),cv2.COLOR_RGB2BGR) for f in frames],keys)]
            assert all(m.sum()>100 for m in masks)
            repair={'method':'SAM2 tracks the source target garment directly from valid source parser masks at0,24,46','keyframes':[0,24,46],'reason':'Actor-and-parser intersection lost garment at47; track the garment itself; no GT loaded.'}
        overlays=[]
        for i,(f,m) in enumerate(zip(frames,masks)):
            Image.fromarray(m).save(out/f'mask_{i:03d}.png')
            rgb=np.asarray(f).copy();sel=m>127;rgb[sel]=(rgb[sel]*.55+np.array([255,0,0])*.45).astype(np.uint8);overlays.append(Image.fromarray(rgb))
        write_preview(overlays,out/'overlay.mp4')
        paths=[out/f'mask_{i:03d}.png' for i in range(49)]
        prepared=ROOT/'inputs/prepared'/sid
        old.update(status='complete',id=sid,input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),mask_frames=[str(p) for p in paths],mask_sha256=[sha(p) for p in paths],mask_semantics={'255':'editable','0':'preserved'},mask_area_min=min(float((m>127).mean()) for m in masks),mask_area_max=max(float((m>127).mean()) for m in masks),overlay_video=str(out/'overlay.mp4'),adapted_prompt='A continuous video. '+item['vace_description']+'\nOriginal edit specification: '+item['prompt']['en'],spec=item['mask_spec'],parser=ready['human_parsing'],source_pixels_sha256=sha(prepared/'pixels.json'),repair=repair,repair_script_sha256=sha(__file__),human_protocol_review='pending')
        old.pop('error',None);atomic(out/'condition.json',old)
        print(sid,'repaired',flush=True)
