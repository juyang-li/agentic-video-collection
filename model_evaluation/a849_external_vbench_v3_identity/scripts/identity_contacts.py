"""CPU: enlarged facial time-series for direct identity diagnosis."""
from common import *
import av,numpy as np
from PIL import Image,ImageOps,ImageDraw

def main():
 for item in rows(ROOT/'metadata/inputs.jsonl'):
  sid=item['id'];c=read(ROOT/'conditions'/sid/'condition.json')
  paths=[('source',ROOT/'inputs/prepared'/sid/'source.mp4')]+[(m,ROOT/'outputs'/m/f'{sid}.mp4') for m in ['ltx_base','a849','vace']]
  out=Image.new('RGB',(1800,5*240),'#20242a');d=ImageDraw.Draw(out)
  ref=Image.open(item['references'][0]['absolute_path']);out.paste(ImageOps.contain(ref,(230,215)),(10,20));d.text((250,60),sid+' / reference (natural hair; gray128)',fill='white')
  for j,(name,path) in enumerate(paths,1):
   if not path.exists():continue
   with av.open(str(path)) as con:frames=[f.to_image() for f in con.decode(video=0)]
   if len(frames)!=49:continue
   for k,f in enumerate([0,6,12,18,24,30,36,42,48]):
    mask=np.asarray(Image.open(c['mask_frames'][f]))>127;ys,xs=np.nonzero(mask);x1,x2=xs.min(),xs.max();y1,y2=ys.min(),ys.max();w=x2-x1;h=y2-y1
    crop=frames[f].crop((max(0,x1-w*.45),max(0,y1-h*.5),min(frames[f].width,x2+w*.45),min(frames[f].height,y2+h*.25)))
    out.paste(ImageOps.contain(crop,(195,205)),(k*200,j*240+25));d.text((k*200+2,j*240+2),f'{name} {f/12:g}s',fill='white')
  out.save(ROOT/'review'/f'{sid}_face_detail.jpg',quality=97)
if __name__=='__main__':main()
