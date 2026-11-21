from common import *
import av
from PIL import Image,ImageDraw
require_job()
canvas=Image.new('RGB',(1320,990),'white');draw=ImageDraw.Draw(canvas)
items={r['id']:r for r in rows(ROOT/'metadata/inputs.jsonl')}
for row,sid in enumerate(['identity_01','hair_color_01','action_01']):
 for side,path in enumerate([items[sid]['guides'][0]['path'],str(ROOT/'outputs'/f'{sid}.mp4')]):
  with av.open(path) as c:
   for i,f in enumerate(c.decode(video=0)):
    if i not in [0,120,240]:continue
    col=side*3+[0,120,240].index(i);im=f.to_image();im.thumbnail((220,300));canvas.paste(im,(col*220,row*330+25));draw.text((col*220,row*330),sid+(' SOURCE ' if side==0 else ' IC-LoRA ')+str(i/24)+'s',fill='black')
canvas.save(ROOT/'metadata/pilot_source_output.jpg')
