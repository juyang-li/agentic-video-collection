import sys
from common import *
from media import video_frames
from PIL import Image,ImageDraw
import numpy as np
require_job()
item=next(r for r in rows(ROOT/'metadata/inputs.jsonl') if r['id']=='hair_color_10');frames,_=video_frames(item,item['source'])
canvas=Image.new('RGB',(5*300,2*320),'white');draw=ImageDraw.Draw(canvas)
for col,i in enumerate([59,60,61,87,120]):
 f=frames[i];mask=np.asarray(Image.open(read(ROOT/'conditions/hair_color_10/condition.json')['mask_frames'][i]))>127
 rgb=np.asarray(f).copy();tint=np.zeros_like(rgb);tint[:,:,0]=255;rgb=(rgb*(1-mask[:,:,None]*.45)+tint*(mask[:,:,None]*.45)).astype(np.uint8)
 for row,im in enumerate([f,Image.fromarray(rgb)]):
  w,h=im.size;im=im.crop((int(.35*w),0,int(.65*w),int(.45*h)));im.thumbnail((300,290));canvas.paste(im,(col*300,row*320+25))
 draw.text((col*300,0),str(i)+' t='+str(round(i/12,3)),fill='black')
canvas.save(ROOT/'metadata/hair_color_10_repair_overlay.jpg')
