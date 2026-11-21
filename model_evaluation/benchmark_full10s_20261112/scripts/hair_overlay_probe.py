import sys
from common import *
from media import video_frames
from PIL import Image,ImageDraw
import numpy as np
require_job()
item=next(r for r in rows(ROOT/'metadata/inputs.jsonl') if r['id']=='hair_07');frames,_=video_frames(item,item['source'])
canvas=Image.new('RGB',(5*300,2*320),'white');draw=ImageDraw.Draw(canvas)
for col,i in enumerate([29,30,31,32,33]):
 f=frames[i];mask=np.asarray(Image.open(ROOT/'conditions/hair_07'/f'mask_{i:03d}.png'))>127
 rgb=np.asarray(f).copy();tint=np.zeros_like(rgb);tint[:,:,0]=255;rgb=(rgb*(1-mask[:,:,None]*.45)+tint*(mask[:,:,None]*.45)).astype(np.uint8)
 for row,im in enumerate([f,Image.fromarray(rgb)]):
  w,h=im.size;im=im.crop((int(.3*w),0,int(.75*w),int(.4*h)));im.thumbnail((300,290));canvas.paste(im,(col*300,row*320+25))
 draw.text((col*300,0),str(i)+' t='+str(round(i/12,3)),fill='black')
canvas.save(ROOT/'metadata/hair_repair_overlay.jpg')
