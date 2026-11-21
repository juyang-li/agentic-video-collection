from common import *
import av,argparse
from PIL import Image,ImageDraw
require_job();p=argparse.ArgumentParser();p.add_argument('sid');a=p.parse_args();r=next(r for r in rows(ROOT/'metadata/inputs.jsonl') if r['id']==a.sid)
paths=[r['prepared_source']['path'],str(ROOT/'outputs'/f'{a.sid}.mp4')];samples=[]
for path in paths:
 chosen={}
 with av.open(path) as c:
  for i,f in enumerate(c.decode(video=0)):
   if i in (0,24,48,72,96):chosen[i]=f.to_image()
 samples.append(chosen)
w=320;h=round(w*r['h3_canvas_wh'][1]/r['h3_canvas_wh'][0]);canvas=Image.new('RGB',(w*2,(h+24)*5),'#ddd');draw=ImageDraw.Draw(canvas)
for row,i in enumerate((0,24,48,72,96)):
 for col,frames in enumerate(samples):
  image=frames[i].resize((w,h),Image.Resampling.LANCZOS);canvas.paste(image,(col*w,row*(h+24)+24));draw.text((col*w+4,row*(h+24)+3),('SOURCE' if col==0 else 'H3')+f' t={i/24:g}',fill='black')
dst=ROOT/'metadata/visual'/f'{a.sid}.png';dst.parent.mkdir(exist_ok=True);canvas.save(dst);print(dst)
