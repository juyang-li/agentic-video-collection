from common import *
from PIL import Image,ImageDraw
import av
require_job()
item=next(i for i in rows(ROOT/'metadata/inputs.jsonl') if i['id']=='hair_color_01')
canvas=Image.new('RGB',(1320,350),'white');draw=ImageDraw.Draw(canvas)
for side,path in enumerate([item['prepared_source']['path'],str(ROOT/'outputs/hair_color_01.mp4')]):
 with av.open(path) as c:
  for i,f in enumerate(c.decode(video=0)):
   if i not in [0,120,240]:continue
   col=side*3+[0,120,240].index(i);im=f.to_image();im.thumbnail((220,300));canvas.paste(im,(col*220,25));draw.text((col*220,0),('SOURCE ' if side==0 else 'H3 ')+str(i/24)+'s',fill='black')
canvas.save(ROOT/'metadata/source_output_full10s.jpg')
