"""Face segmentation onto RGB128; single frontal photo per hair task."""
import sys
from common import *

def main():
    require_job()
    import requests
    import cv2
    import numpy as np
    from PIL import Image,ImageOps,ImageDraw
    sys.path.insert(0,str(ROOT.parent/'ltx25_vace_a100_20261111/scripts'))
    from human_parsing import HumanParser
    ready=read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json')
    parser=HumanParser(ready['human_parsing'])
    records=[]; thumbnails=[]
    for r in read(ROOT/'metadata/portrait_sources.json'):
        if 'error' in r:raise RuntimeError(r)
        original=Image.open(r['path']).convert('RGB')
        original.thumbnail((1280,1280))
        mask=parser.segment(original,[11])
        # A face detector bounds the semantic face label to exclude the neck.
        detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        boxes=detector.detectMultiScale(cv2.cvtColor(np.asarray(original),cv2.COLOR_RGB2GRAY),scaleFactor=1.1,minNeighbors=5)
        assert len(boxes),r['name']
        x,y,w,h=max(boxes,key=lambda b:int(b[2])*int(b[3]))
        roi=np.zeros_like(mask);roi[y:y+h,x:x+w]=1
        mask &= roi
        assert mask.sum()>2000
        pixels=np.full_like(np.asarray(original),128);pixels[mask>0]=np.asarray(original)[mask>0]
        # Tight square face canvas, neutral gray only outside the parsed face.
        face=Image.fromarray(pixels).crop((x,y,x+w,y+h))
        dst=ROOT/'references'/f'{r["name"]}_face.png';face.save(dst)
        maskpath=ROOT/'references'/f'{r["name"]}_face_mask.png';Image.fromarray(mask*255).save(maskpath)
        # Hair reference is one frontal head photo, including complete hairstyle.
        head=original.crop((max(0,x-w//3),max(0,y-h//2),min(original.width,x+w+w//3),min(original.height,y+h+h//5)))
        hairpath=ROOT/'references'/f'{r["name"]}_hair.png';head.save(hairpath)
        records.append(dict(**r,face_path=str(dst),face_sha256=sha(dst),hair_path=str(hairpath),hair_sha256=sha(hairpath),face_box=[int(v) for v in [x,y,w,h]],mask=str(maskpath),parser=ready['human_parsing'],processing='Segformer face label11 intersect frontal face detection; non-face RGB128; crop detector ROI; hair single frontal head crop',review='pending'))
        thumbnails.extend([(r['name']+' original',original),(r['name']+' segmented face',face),(r['name']+' frontal hair',head)])
    session=requests.Session();session.trust_env=False
    backgrounds={'living_room':'1600210492486-724fe5c67fb0','kitchen':'1556912172-45b7abe8b7e1','forest':'1448375240586-882707db888b','beach':'1507525428034-b723cf961d3e'}
    for name,pid in backgrounds.items():
        url=f'https://images.unsplash.com/photo-{pid}?auto=format&fit=max&w=1280&q=90'
        dst=ROOT/'references'/f'background_{name}.jpg'
        if not dst.exists():
            res=session.get(url,timeout=120);res.raise_for_status();dst.write_bytes(res.content)
        img=Image.open(dst).convert('RGB')
        records.append(dict(name=name,role='background',url=url,path=str(dst),sha256=sha(dst),review='pending'))
        thumbnails.append((name,img))
    out=Image.new('RGB',(1200,((len(thumbnails)+3)//4)*330),'#20242a');draw=ImageDraw.Draw(out)
    for i,(label,img) in enumerate(thumbnails):
        x=(i%4)*300;y=(i//4)*330
        out.paste(ImageOps.contain(img,(295,300)),(x,y+25));draw.text((x+4,y+4),label,fill='white')
    out.save(ROOT/'review/references.jpg',quality=92)
    atomic(ROOT/'metadata/reference_processing.json',records)

if __name__=='__main__':main()
