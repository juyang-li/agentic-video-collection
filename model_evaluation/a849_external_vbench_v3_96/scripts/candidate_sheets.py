"""CPU direct-review sheets: full source time span, no VLM."""
from common import *
import av
from PIL import Image,ImageOps,ImageDraw

def main():
 records=read(ROOT/'metadata/object_candidates.json')
 sheet=Image.new('RGB',(1600,len(records)*170),'#20242a');d=ImageDraw.Draw(sheet)
 for j,r in enumerate(records):
  with av.open(r['path']) as c:
   stream=c.streams.video[0];frames=[(float(f.time),f.to_image()) for f in c.decode(video=0)]
  times=[i*frames[-1][0]/7 for i in range(8)]
  assert frames[-1][0]>=4
  r['probe']=dict(duration_last_pts=frames[-1][0],frames=len(frames),dimensions=list(frames[0][1].size),review_timestamps=times)
  for k,t in enumerate(times):
   _,im=min(frames,key=lambda v:abs(v[0]-t));x=k*200;y=j*170
   sheet.paste(ImageOps.contain(im,(197,143)),(x,y+23));d.text((x+2,y+2),f"{r['id'].replace('Sp_Object_','')} {t:.1f}s",fill='white')
 sheet.save(ROOT/'review/object_candidates.jpg',quality=96);atomic(ROOT/'metadata/object_candidates.json',records)
if __name__=='__main__':main()
