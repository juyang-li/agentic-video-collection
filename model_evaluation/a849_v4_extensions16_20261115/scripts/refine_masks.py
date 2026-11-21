"""Before generation: source-inspected scalp limits and foreground-object protection."""
import argparse,sys
from common import *

def save(item,masks,evidence,tag):
    import numpy as np
    from PIL import Image
    from media import write_preview
    sid=item['id'];out=ROOT/'conditions'/sid;c=read(out/'condition.json')
    history=ROOT/'metadata/mask_attempt_history'/f'{sid}_{tag}_before.json'
    if history.exists():
        assert tag not in c['mask_evidence'],f'Refinement already complete: {history}'
        assert read(history)['input_manifest_sha256']==c['input_manifest_sha256']
    else:atomic(history,c)
    overlays=[];areas=[]
    for n,m in enumerate(masks):
        m=(m>127).astype('uint8')*255;assert np.count_nonzero(m)>100
        Image.fromarray(m).save(out/f'mask_{n:03}.png');rgb=np.array(Image.open(ROOT/'inputs/prepared'/sid/f'{n:03}.png').convert('RGB'));sel=m>127
        rgb[sel]=(rgb[sel]*.55+np.array([255,0,0])*.45).astype('uint8');overlays.append(Image.fromarray(rgb));areas.append(float(sel.mean()))
    write_preview(overlays,out/'overlay.mp4',12);c['mask_sha256']=[sha(p)for p in c['mask_frames']];c.update(mask_area_min=min(areas),mask_area_max=max(areas))
    c['mask_evidence'][tag]=evidence;atomic(out/'condition.json',c);print('Refined',sid,tag,flush=True)

def cpu():
    import cv2,numpy as np
    from PIL import Image,ImageDraw
    by={r['id']:r for r in rows(ROOT/'metadata/inputs.jsonl')}
    polygons={
        'hair_color_17':[(.43,0),(.76,0),(.76,.30),(.43,.30)],
        'hair_color_19':[(.30,0),(.68,0),(.68,.40),(.30,.40)],
        'hair_color_20':[(.46,0),(.80,0),(.80,.57),(.70,.57),(.70,.42),(.46,.42)],
    }
    for sid,points in polygons.items():
        item=by[sid];c=read(ROOT/'conditions'/sid/'condition.json');masks=[]
        if 'scalp_limit'in c['mask_evidence']:continue
        for p in c['mask_frames']:
            m=np.array(Image.open(p));h,w=m.shape;allow=Image.new('L',(w,h));ImageDraw.Draw(allow).polygon([(round(x*w),round(y*h))for x,y in points],fill=255);m[np.array(allow)==0]=0;masks.append(m)
        save(item,masks,dict(method='Direct source scalp polygon, excluding beard/mouth/wrist false positives',normalized_polygon=points,source_evidence=str(ROOT/'review'/f'{sid}_source_grid.jpg'),no_gt=True),'scalp_limit')
    for sid,limit in [('background_17',.86),('background_22',.50)]:
        item=by[sid];c=read(ROOT/'conditions'/sid/'condition.json');masks=[]
        if 'foreground_surface'in c['mask_evidence']:continue
        for p in c['mask_frames']:
            m=np.array(Image.open(p));m[round(m.shape[0]*limit):]=0;masks.append(m)
        save(item,masks,dict(method='Source-planned foreground interaction surface preservation',normalized_y_lower_bound=limit,foreground='keyboard/front desk'if sid.endswith('17')else'pool table, balls and cue below the table horizon',no_gt=True),'foreground_surface')
    for item in by.values():
        if item['group']!='background':continue
        c=read(ROOT/'conditions'/item['id']/'condition.json');masks=[];changed=False
        for p in c['mask_frames']:
            m=np.array(Image.open(p));count,components,stats,_=cv2.connectedComponentsWithStats((m>127).astype('uint8'),connectivity=8)
            for label in range(1,count):
                ys,xs=np.where(components==label)
                if stats[label,cv2.CC_STAT_AREA]<4096 and xs.min()>0 and xs.max()<m.shape[1]-1 and ys.min()>0 and ys.max()<m.shape[0]-1:
                    m[components==label]=0;changed=True
            masks.append(m)
        if changed:save(item,masks,dict(method='Protect small isolated editable holes within source foreground, e.g. mouth/skin; do not alter border-connected background',maximum_component_pixels=4096,no_gt=True),'foreground_holes')

def gpu(shard):
    require_job();assert (ROOT/'runtime/cache_ready').exists()
    import numpy as np,cv2
    from PIL import Image
    ready=read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json')
    sys.path.insert(0,ready['sam2']['path']);sys.path.insert(0,str(ROOT.parent/'omni_baselines_20260926/vace'))
    from generate_vace_masks import MaskModels
    models=MaskModels(None,Path(ready['sam2']['checkpoint']),manual_boxes=True)
    # Boxes are normalized to frozen source prepared canvases, not GT.
    specs=[
        ('hair_color_22','sunglasses',{0:[.075,.07,.59,.34],24:[.04,.055,.52,.34],48:[.065,.055,.56,.35]}),
        ('background_19','held_cash',{0:[.44,.58,.60,.77],24:[.40,.59,.60,.78],48:[.43,.58,.60,.79]}),
        ('background_19','foreground_shoes',{0:[.70,.20,.95,.96],24:[.70,.20,.95,.96],48:[.70,.20,.95,.96]}),
        ('background_20','held_cup',{0:[.51,.54,.57,.64],24:[.46,.42,.53,.52],48:[.43,.41,.50,.51]}),
        ('background_20','bottle',{0:None,24:None,36:[.53,.26,.64,.43],48:[.50,.23,.61,.40]}),
        ('background_24','racket',{0:None,6:[.32,.60,.46,.69],12:[.475,.325,.525,.46],18:[.475,.31,.515,.49],24:[.41,.56,.50,.66],30:[.12,.50,.32,.61],36:[.48,.52,.57,.64],42:[.455,.28,.51,.46],48:[.40,.43,.53,.60]}),
    ]
    by={r['id']:r for r in rows(ROOT/'metadata/inputs.jsonl')}
    # Keep all refinements of one case on the same worker.
    ids=sorted({sid for sid,_,_ in specs})[shard::4]
    for sid in ids:
        item=by[sid];frames=[Image.open(ROOT/'inputs/prepared'/sid/f'{n:03}.png').convert('RGB')for n in range(49)];w,h=frames[0].size;bgr=[cv2.cvtColor(np.array(f),cv2.COLOR_RGB2BGR)for f in frames]
        c=read(ROOT/'conditions'/sid/'condition.json');masks=[np.array(Image.open(p))for p in c['mask_frames']];proofs=[]
        if 'foreground_objects'in c['mask_evidence']:continue
        for case,label,boxes in specs:
            if case!=sid:continue
            seeds={}
            for n,box in boxes.items():
                if box is None:seeds[n]=np.zeros((h,w),dtype='uint8');continue
                pixels=np.array([[v*d for v,d in zip(box,[w,h,w,h])]],dtype='float32');seed,method=models.segment_keyframe(bgr[n],pixels,tuple(box));assert seed.sum()>10,(sid,label,n)
                seeds[n]=seed
            tracked=models.track(bgr,seeds)
            for n,foreground in enumerate(tracked):
                foreground=cv2.dilate(foreground,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))>0;masks[n][foreground]=0
            proofs.append(dict(object=label,source_boxes=boxes,keyframes=list(seeds),method='SAM2 source-only object segmentation and propagation; protect object pixels'))
        save(item,masks,dict(objects=proofs,no_gt=True),'foreground_objects')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu',action='store_true');p.add_argument('--shard',type=int);a=p.parse_args()
    assert not (ROOT/'metadata/generation_gate.json').exists(),'All refinements must precede generation gate'
    if a.cpu:cpu()
    else:gpu(a.shard)
