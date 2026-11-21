"""All49 output frames for uncertain temporal artifacts, recorded on demand."""
import argparse
from common import *

if __name__=='__main__':
    require_job()
    import av
    from PIL import Image,ImageOps,ImageDraw
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--model',choices=['ltx_base','a849','vace'],required=True);a=p.parse_args()
    path=ROOT/'outputs'/a.model/(a.id+'.mp4')
    with av.open(str(path)) as c:frames=list(c.decode(video=0))
    assert len(frames)==49
    w,h=frames[0].width,frames[0].height
    tile=(200,180) if w>=h else (145,230)
    out=Image.new('RGB',(tile[0]*7,tile[1]*7),'#20242a');draw=ImageDraw.Draw(out)
    for i,f in enumerate(frames):
        x=i%7*tile[0];y=i//7*tile[1]
        out.paste(ImageOps.contain(f.to_image(),(tile[0]-3,tile[1]-22)),(x,y+20));draw.text((x,y),f'{i}  {i/12:.3f}s',fill='white')
    dest=ROOT/'review'/f'{a.id}_{a.model}_all49.jpg';out.save(dest,quality=93)
    print(dest,flush=True)
