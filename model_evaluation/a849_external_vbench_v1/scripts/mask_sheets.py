"""Source/overlay temporal sheets for direct protocol review, no outputs read."""
from common import *

if __name__=='__main__':
    require_job()
    import av
    from PIL import Image,ImageOps,ImageDraw
    items=rows(ROOT/'metadata/inputs.jsonl')
    for start in range(0,len(items),4):
        canvas=Image.new('RGB',(1440,4*210),'#20242a');draw=ImageDraw.Draw(canvas)
        for row,item in enumerate(items[start:start+4]):
            out=ROOT/'conditions'/item['id'];p=out/'overlay.mp4'
            draw.text((4,row*210+2),item['id'],fill='white')
            if not p.exists():
                draw.text((4,row*210+28),str(read(out/'condition.json'))[:140],fill='white');continue
            with av.open(str(p)) as c:frames=list(c.decode(video=0))
            for col,i in enumerate([0,6,12,18,24,30,42,48]):
                canvas.paste(ImageOps.contain(frames[i].to_image(),(178,175)),(col*180,row*210+24));draw.text((col*180,row*210+194),f'{i/12:.1f}s',fill='white')
        canvas.save(ROOT/'review'/f'masks_{start//4:02d}.jpg',quality=93)
