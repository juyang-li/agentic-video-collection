"""Small source/reference pages; audit inputs before inference, never read GT."""
from common import *

if __name__=='__main__':
    require_job()
    import av
    from PIL import Image,ImageDraw,ImageOps
    items=[r for r in rows(ROOT/'metadata/inputs.jsonl') if r['group']=='tripvvt']
    for start in range(0,len(items),6):
        page=Image.new('RGB',(1200,1050),'#20242a');draw=ImageDraw.Draw(page)
        for j,r in enumerate(items[start:start+6]):
            with av.open(r['source']['absolute_path']) as c:source=next(c.decode(video=0)).to_image()
            images=[('source',source),('garment',Image.open(r['references'][0]['absolute_path']).convert('RGB'))]
            x=j%2*600;y=j//2*350
            for col,(label,im) in enumerate(images):
                page.paste(ImageOps.contain(im,(295,320)),(x+col*300,y+25));draw.text((x+col*300,y+3),r['id']+' '+label,fill='white')
        page.save(ROOT/'review'/f'garment_references_{start//6:02d}.jpg',quality=94)
