"""CPU direct-inspection evidence for four new source/reference pairings."""
from common import *
from PIL import Image,ImageOps,ImageDraw
import numpy as np

def main():
 selected=[r for r in rows(ROOT/'metadata/inputs.jsonl')if r['id'].startswith('hair_celeb_')]
 for item in selected:
  sid=item['id'];out=Image.new('RGB',(1600,850),'#20242a');draw=ImageDraw.Draw(out)
  out.paste(ImageOps.contain(Image.open(item['references'][0]['absolute_path']).convert('RGB'),(380,380)),(10,25))
  draw.text((410,20),sid+' FROM '+item['source_task_id']+' / '+item['uid'],fill='white')
  draw.text((410,45),'Match reference shape/length ONLY. Preserve source hair color, face, clothes, action and scene.',fill='white')
  c=read(ROOT/'conditions'/sid/'condition.json')
  for k,f in enumerate([0,12,24,36,48]):
   frame=Image.open(ROOT/'inputs/prepared'/sid/f'{f:03d}.png').convert('RGB');mask=np.asarray(Image.open(c['mask_frames'][f]))>127
   rgb=np.asarray(frame).copy();rgb[mask]=(rgb[mask]*.5+np.array([255,0,0])*.5).astype('uint8')
   for row,image in enumerate([frame,Image.fromarray(rgb)]):
    out.paste(ImageOps.contain(image,(310,200)),(k*320,420+row*210));draw.text((k*320,405+row*210),f'{f/12:g}s '+('source'if row==0 else'source hair mask'),fill='white')
  out.save(ROOT/'review'/f'{sid}_input.jpg',quality=97)
 print('New input sheets ready',flush=True)
if __name__=='__main__':main()
