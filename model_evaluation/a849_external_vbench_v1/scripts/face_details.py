"""Enlarged source-ROI audit views only; never alters inference references."""
from common import *

if __name__=='__main__':
    require_job()
    import av
    import numpy as np
    from PIL import Image,ImageDraw,ImageOps
    for sid in ['native_01','native_02','native_03','native_04']:
        videos=[('SOURCE',ROOT/'inputs/prepared'/sid/'source.mp4')]
        for model in ['ltx_base','a849','vace']:
            path=ROOT/'outputs'/model/f'{sid}.mp4';receipt=path.with_suffix('.json')
            if receipt.is_file() and read(receipt)['status']=='complete':videos.append((model,path))
        sheet=Image.new('RGB',(1500,len(videos)*330),'#20242a');draw=ImageDraw.Draw(sheet)
        for row,(model,path) in enumerate(videos):
            with av.open(str(path)) as c:frames=list(c.decode(video=0))
            for col,i in enumerate([0,12,24,36,48]):
                mask=np.asarray(Image.open(ROOT/'conditions'/sid/f'mask_{i:03d}.png'))>127
                ys,xs=np.where(mask);x1,x2=xs.min(),xs.max()+1;y1,y2=ys.min(),ys.max()+1
                dx=int((x2-x1)*.3);dy=int((y2-y1)*.3)
                # This is an explicitly labeled audit magnification, not a
                # change to the user's full-frame identity input photographs.
                crop=frames[i].to_image().crop((max(0,x1-dx),max(0,y1-dy),min(frames[i].width,x2+dx),min(frames[i].height,y2+dy)))
                sheet.paste(ImageOps.contain(crop,(295,300)),(col*300,row*330+28));draw.text((col*300,row*330+4),f'{sid} {model} {i/12:.1f}s (audit ROI)',fill='white')
        sheet.save(ROOT/'review'/f'{sid}_face_detail.jpg',quality=94)
