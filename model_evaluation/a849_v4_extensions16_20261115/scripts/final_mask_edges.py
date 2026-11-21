"""Source-only foreground edge correction before the immutable generation gate."""
from common import *
from refine_masks import save

def main():
    require_job()
    assert not (ROOT/'metadata/generation_gate.json').exists()
    import sys, cv2, numpy as np
    from PIL import Image, ImageDraw
    sys.path.insert(0,str(ROOT.parent/'ltx25_vace_a100_20261111/scripts'))
    from human_parsing import HumanParser
    parser=HumanParser(read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json')['human_parsing'])
    items=rows(ROOT/'metadata/inputs.jsonl')
    for item in items:
        sid=item['id'];c=read(ROOT/'conditions'/sid/'condition.json')
        if item['group']!='background' and sid!='hair_color_22':continue
        if 'source_foreground_edges' in c['mask_evidence']:continue
        masks=[]
        for n,path in enumerate(c['mask_frames']):
            im=Image.open(ROOT/'inputs/prepared'/sid/f'{n:03d}.png').convert('RGB');w,h=im.size
            m=np.array(Image.open(path))
            if item['group']=='background':
                # Undo the initial editable-region dilation which consumed narrow
                # fingers/arms; all twenty human labels include both shoes.
                m=cv2.erode(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
                if sid in ['background_17','background_18','background_19','background_24']:
                    fg=parser.segment(im,list(range(1,20)))
                    fg=cv2.dilate(fg,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0
                    m[fg]=0
                if sid=='background_19':
                    # Hands, fanned cash and trouser legs are foreground objects.
                    # The source camera is fixed; this region is within the seated
                    # subject/chair silhouette throughout all 49 source frames.
                    keep=Image.new('L',(w,h));ImageDraw.Draw(keep).polygon([(round(x*w),round(y*h)) for x,y in [(.39,.57),(.62,.57),(.72,.77),(.72,.88),(.38,.88)]],fill=255)
                    m[np.array(keep)>0]=0
                    m[round(.88*h):]=0
            else:
                # Rebuild scalp pixels removed by an overbroad SAM sunglasses
                # segment. Preserve source-traced lenses and the thin glasses arm.
                m=parser.segment(im,[2],main_person=True)*255
                m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
                m[parser.segment(im,[11])>0]=0
                points=np.array([[.08,.22],[.405,.105],[.475,.105],[.54,.15],[.57,.25],[.54,.30],[.47,.31],[.42,.25],[.40,.17],[.10,.265]])
                anchors=np.array([[0,0],[-.065,-.015],[-.025,-.035]])
                k=min(n//24,1);a=(n-k*24)/24;delta=anchors[k]*(1-a)+anchors[k+1]*a
                points+=delta
                keep=Image.new('L',(w,h));ImageDraw.Draw(keep).polygon([(round(x*w),round(y*h)) for x,y in points],fill=255)
                m[np.array(keep)>0]=0
            masks.append(m)
        save(item,masks,dict(method='Direct source edge review; preserve narrow limbs, shoes and held objects; sunglasses polygon follows source keyframes',no_gt=True,source_times=[0,1,2,3,4],script_sha256=sha(__file__)),'source_foreground_edges')

if __name__=='__main__':main()
