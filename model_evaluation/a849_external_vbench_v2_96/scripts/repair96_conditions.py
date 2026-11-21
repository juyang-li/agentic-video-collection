"""Pre-inference source-only repairs identified from actual mask contact sheets."""
import argparse,sys
from common import *

def main(shard,only=None):
    require_job()
    import cv2,numpy as np
    from PIL import Image
    from media import write_preview
    ready=read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json')
    sys.path.insert(0,str(ROOT.parent/'ltx25_vace_a100_20261111/scripts'))
    from human_parsing import HumanParser
    parser=HumanParser(ready['human_parsing'])
    sys.path.insert(0,ready['sam2']['path']);sys.path.insert(0,str(ROOT.parent/'omni_baselines_20260926/vace'))
    from generate_vace_masks import MaskModels
    tracker=None
    jobs=['identity_11','identity_12','identity_15','identity_16','hair_10','hair_12','hair_13','native_07','native_11','native_24','tripvvt_08','tripvvt_17','tripvvt_23']
    if only:jobs=[x for x in jobs if x in only.split(',')]
    items={r['id']:r for r in rows(ROOT/'metadata/inputs.jsonl')}
    def largest(mask):
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        assert n>1,'No semantic component found'
        return (labels==1+np.argmax(stats[1:,cv2.CC_STAT_AREA])).astype(np.uint8)
    def hull(mask):
        pts=cv2.findNonZero(mask.astype(np.uint8));assert pts is not None
        out=np.zeros_like(mask,dtype=np.uint8);cv2.fillConvexPoly(out,cv2.convexHull(pts),255);return out
    for sid in jobs[shard::4]:
        assert all(not (ROOT/'outputs'/m/f'{sid}.json').exists() for m in ['ltx_base','a849','vace'])
        folder=ROOT/'conditions'/sid;old=read(folder/'condition.json');item=items[sid]
        if old.get('repair96') and not only:continue
        if not (folder/'before_repair96.json').exists():atomic(folder/'before_repair96.json',old)
        frames=[Image.open(ROOT/'inputs/prepared'/sid/f'{i:03d}.png').convert('RGB') for i in range(49)]
        w,h=frames[0].size;masks=[]
        if sid.startswith('tripvvt'):
            if tracker is None:tracker=MaskModels(None,Path(ready['sam2']['checkpoint']),manual_boxes=True)
            roi={'tripvvt_08':[.49,.34,.84,.64],'tripvvt_17':[.23,.40,.57,.92],'tripvvt_23':[.35,.33,.88,.91]}[sid]
            keys={}
            for k in [0,24,48]:
                if sid=='tripvvt_23':
                    box={0:[.40,.33,.72,.77],24:[.39,.33,.74,.85],48:[.38,.39,.74,.91]}[k]
                    pixels=np.array([[box[0]*w,box[1]*h,box[2]*w,box[3]*h]],dtype=np.float32)
                    seed,_=tracker.segment_keyframe(cv2.cvtColor(np.asarray(frames[k]),cv2.COLOR_RGB2BGR),pixels,(0,0,1,1))
                    seed[parser.segment(frames[k],[1,2,3,9,10,11,12,13,14,15])>0]=0
                    keys[k]=seed
                    continue
                raw=parser.segment(frames[k],item['mask_spec']['parsing_labels']);keep=np.zeros_like(raw)
                x1,y1,x2,y2=roi;keep[int(y1*h):int(y2*h),int(x1*w):int(x2*w)]=1
                seed=largest(raw&keep);assert seed.sum()>100,(sid,k)
                keys[k]=seed
            tracked=tracker.track([cv2.cvtColor(np.asarray(f),cv2.COLOR_RGB2BGR) for f in frames],keys)
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25));masks=[cv2.dilate(m*255,kernel) for m in tracked]
            method=dict(method='Track garment directly from largest source semantic garment component inside manually reviewed foreground garment ROI, not a failing whole-person intersection',source_roi=roi,keyframes=[0,24,48],no_gt=True)
        else:
            for i,f in enumerate(frames):
                if sid in ['hair_10','hair_13']:
                    roi=[.25,.02,.68,.99] if sid=='hair_10' else [.35,.15,.65,.98]
                    x1,y1,x2,y2=roi;box=[int(x1*w),int(y1*h),int(x2*w),int(y2*h)]
                    crop=f.crop(tuple(box));local=parser.segment(crop,[2]);local=largest(local)
                    raw=np.zeros((h,w),np.uint8);raw[box[1]:box[3],box[0]:box[2]]=local
                    m=cv2.dilate(raw*255,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(49,49)))
                    face=np.zeros_like(raw);face[box[1]:box[3],box[0]:box[2]]=parser.segment(crop,[11]);m[face>0]=0
                else:
                    m=np.asarray(Image.open(folder/f'mask_{i:03d}.png')).copy()
                    if sid in ['identity_11','identity_12','identity_16','native_11']:m=hull(m)
                    if sid=='native_07':m[round(.65*h):]=0
                    if sid=='hair_12':m[round(.60*h):]=0
                    if sid=='native_11':
                        pts=np.array([[.43,.55],[.73,.48],[.76,.70],[.66,.96],[.43,.98],[.37,.76]])*np.array([w,h]);cv2.fillPoly(m,[pts.astype(np.int32)],255)
                        m[parser.segment(f,[1,14,15])>0]=0
                    if sid in ['identity_15','native_24']:
                        m=np.zeros((h,w),np.uint8)
                        pts=np.array([[.39,0],[.99,0],[.99,.78],[.85,.99],[.47,.99],[.37,.82],[.34,.56],[.40,.35]])*np.array([w,h]);cv2.fillPoly(m,[pts.astype(np.int32)],255)
                        hair=parser.segment(f,[2])>0;hair[round(.20*h):]=False;m[hair]=0
                masks.append(m)
            method=dict(method='Source-only semantic close-up for small/foreground hairstyle; clear remote false positives; fill facial holes or manually include requested beard/face region as identified in source masks',source_roi=roi if sid in ['hair_10','hair_13'] else None,no_gt=True)
        overlays=[]
        pad=read(ROOT/'inputs/prepared'/sid/'pixels.json')['pixel_audit']['geometry']['padding_ltrb']
        for i,(f,m) in enumerate(zip(frames,masks)):
            left,top,right,bottom=pad
            if left:m[:,:left]=0
            if top:m[:top]=0
            if right:m[:,-right:]=0
            if bottom:m[-bottom:]=0
            assert (m>127).sum()>100,(sid,i)
            # Replace a new file so any retained prior mask hardlink is not modified.
            dest=folder/f'mask_{i:03d}.png';tmp=dest.with_name(dest.stem+'.repair.png');Image.fromarray(m).save(tmp);tmp.replace(dest)
            rgb=np.asarray(f).copy();sel=m>127;rgb[sel]=(rgb[sel]*.55+np.array([255,0,0])*.45).astype(np.uint8);overlays.append(Image.fromarray(rgb))
        write_preview(overlays,folder/'overlay_repaired.mp4');(folder/'overlay_repaired.mp4').replace(folder/'overlay.mp4')
        paths=[folder/f'mask_{i:03d}.png' for i in range(49)]
        old.update(status='complete',id=sid,input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),mask_frames=[str(p) for p in paths],mask_sha256=[sha(p) for p in paths],mask_semantics={'255':'editable','0':'preserved'},mask_area_min=min(float((m>127).mean()) for m in masks),mask_area_max=max(float((m>127).mean()) for m in masks),overlay_video=str(folder/'overlay.mp4'),adapted_prompt='A continuous video. '+item['vace_description']+'\nOriginal edit specification: '+item['prompt']['en'],parser=ready['human_parsing'],spec=item['mask_spec'],source_pixels_sha256=sha(ROOT/'inputs/prepared'/sid/'pixels.json'),repair96=method,repair_script_sha256=sha(__file__),human_protocol_review='pending')
        old.pop('error',None);atomic(folder/'condition.json',old);print(sid,'repaired; visual recheck pending',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--only');a=p.parse_args();main(a.shard,a.only)
