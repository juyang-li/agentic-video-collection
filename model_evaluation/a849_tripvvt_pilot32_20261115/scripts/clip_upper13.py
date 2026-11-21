"""CPU source-inspected waist boundary protects13's skirt from parser dress labels."""
from common import *

def main():
    assert not (ROOT/'metadata/generation_gate.json').exists()
    import numpy as np
    from PIL import Image
    from media import write_preview
    out=ROOT/'conditions/tripvvt_13';c=read(out/'condition.json')
    keys=[(0,.39,.492,.53,.495),(12,.427,.484,.543,.481),(24,.443,.498,.582,.505),(36,.514,.484,.622,.463),(48,.478,.489,.647,.519)]
    atomic(ROOT/'metadata/mask_attempt_history/tripvvt_13_before_waist_clip.json',c)
    overlays=[];areas=[]
    for n in range(49):
        p=out/f'mask_{n:03}.png';m=np.array(Image.open(p));h,w=m.shape
        x1,y1,x2,y2=[np.interp(n,[k[0]for k in keys],[k[j]for k in keys])for j in range(1,5)]
        boundary=(y1+(np.arange(w)/w-x1)*(y2-y1)/(x2-x1))*h
        m[np.arange(h)[:,None]>boundary[None,:]]=0;Image.fromarray(m).save(p)
        f=np.array(Image.open(ROOT/'inputs/prepared/tripvvt_13'/f'{n:03}.png').convert('RGB'));sel=m>127;f[sel]=(f[sel]*.55+np.array([255,0,0])*.45).astype('uint8');overlays.append(Image.fromarray(f));areas.append(float(sel.mean()))
    write_preview(overlays,out/'overlay.mp4');c['mask_sha256']=[sha(p)for p in c['mask_frames']]
    c.update(mask_area_min=min(areas),mask_area_max=max(areas))
    c['mask_evidence']['source_inspected_waist_clip']=dict(keyframes=keys,coordinate_system='normalized source prepared canvas',reason='Parser labels merged gray top with purple skirt; skirt must be preserved',source_evidence=str(ROOT/'review/upper13_source_waist.jpg'),no_gt=True)
    atomic(out/'condition.json',c)

if __name__=='__main__':main()
