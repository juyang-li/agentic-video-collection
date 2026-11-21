"""Source-only garment-mask repair before the immutable generation gate."""
import argparse,sys
from common import *

def main():
    require_job();p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args()
    assert (ROOT/'runtime/cache_ready').exists() and not (ROOT/'metadata/generation_gate.json').exists()
    import numpy as np,cv2
    from PIL import Image
    from media import write_preview
    ready=read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json')
    sys.path.insert(0,str(ROOT.parent/'ltx25_vace_a100_20261111/scripts'))
    from human_parsing import HumanParser
    sys.path.insert(0,ready['sam2']['path']);sys.path.insert(0,str(ROOT.parent/'omni_baselines_20260926/vace'))
    from generate_vace_masks import MaskModels
    parser=HumanParser(ready['human_parsing']);tracker=MaskModels(None,Path(ready['sam2']['checkpoint']),manual_boxes=True)
    for item in rows(ROOT/'metadata/inputs.jsonl')[a.shard::4]:
        out=ROOT/'conditions'/item['id'];path=out/'condition.json'
        if read(path)['status']=='complete':continue
        atomic(ROOT/'metadata/mask_attempt_history'/f"{item['id']}_initial.json",read(path))
        prepared=ROOT/'inputs/prepared'/item['id'];audit=read(prepared/'pixels.json')
        frames=[Image.open(prepared/f'{n:03}.png').convert('RGB')for n in range(49)];w,h=frames[0].size;s=item['mask_spec']
        box=s.get('actor_box',[0,0,1,1]);x1,y1,x2,y2=[round(v*d)for v,d in zip(box,[w,h,w,h])]
        seeds={}
        for n in [0,12,24,36,48]:
            # Crop only source pixels to disambiguate the predeclared person;
            # source semantic labels initialize garment propagation, never GT.
            crop=frames[n].crop((x1,y1,x2,y2));raw=parser.segment(crop,s['parsing_labels'],main_person=True)
            mask=np.zeros((h,w),dtype=np.uint8);mask[y1:y2,x1:x2]=raw
            if np.count_nonzero(mask)>100:seeds[n]=mask
        assert 0 in seeds and len(seeds)>=3,(item['id'],list(seeds))
        bgr=[cv2.cvtColor(np.array(f),cv2.COLOR_RGB2BGR)for f in frames];masks=tracker.track(bgr,seeds)
        overlays=[];final=[]
        for n,(frame,mask)in enumerate(zip(frames,masks,strict=True)):
            d=s['dilation_pixels'];mask=cv2.dilate(mask*255,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*d+1,2*d+1)))
            l,t,r,b=audit['pixel_audit']['geometry']['padding_ltrb']
            if l:mask[:,:l]=0
            if t:mask[:t]=0
            if r:mask[:,-r:]=0
            if b:mask[-b:]=0
            assert np.count_nonzero(mask)>100,(item['id'],n)
            Image.fromarray(mask).save(out/f'mask_{n:03}.png');final.append(mask)
            rgb=np.array(frame);sel=mask>127;rgb[sel]=(rgb[sel]*.55+np.array([255,0,0])*.45).astype('uint8');overlays.append(Image.fromarray(rgb))
        write_preview(overlays,out/'overlay.mp4');paths=[out/f'mask_{n:03}.png'for n in range(49)]
        atomic(path,dict(status='complete',id=item['id'],input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),mask_frames=[str(p)for p in paths],mask_sha256=[sha(p)for p in paths],
            mask_semantics={'255':'editable','0':'preserved'},mask_area_min=min(float((m>127).mean())for m in final),mask_area_max=max(float((m>127).mean())for m in final),
            overlay_video=str(out/'overlay.mp4'),adapted_prompt='A continuous video. '+item['vace_description']+'\nOriginal edit specification: '+item['prompt']['en'],spec=s,
            source_pixels_sha256=sha(prepared/'pixels.json'),source_sha256=item['source']['sha256'],frame_sha256=audit['frame_sha256'],
            mask_evidence=dict(method='Source-only parser on predeclared target crop; garment-mask SAM2 propagation from five source keyframes',keyframes=list(seeds),actor_box=box,no_gt=True),
            script_sha256=sha(__file__),human_protocol_review='pending'))
        print('Repaired',item['id'],flush=True)

if __name__=='__main__':main()
