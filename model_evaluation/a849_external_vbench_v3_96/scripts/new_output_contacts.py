"""CPU temporal evidence for new outputs; full frames plus source-mask-based detail."""
import argparse,textwrap
from common import *
import av,numpy as np
from PIL import Image,ImageOps,ImageDraw

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['ltx_base','a849','vace'],required=True);p.add_argument('--limit',type=int,default=999);a=p.parse_args()
 reused={(r['id'],r['model']) for r in read(ROOT/'metadata/reuse_outputs.json')['reused']};count=0
 dest=ROOT/'review/new_contacts';dest.mkdir(parents=True,exist_ok=True)
 for item in rows(ROOT/'metadata/inputs.jsonl'):
  sid=item['id']
  if (sid,a.model) in reused:continue
  video=ROOT/'outputs'/a.model/f'{sid}.mp4';receipt=video.with_suffix('.json')
  if not receipt.exists():continue
  r=read(receipt)
  if r.get('status')!='complete':continue
  out=dest/f'{a.model}--{sid}.jpg';status=out.with_suffix('.json')
  if out.exists() and status.exists() and read(status)['output_sha256']==r['output_sha256']:continue
  assert sha(video)==r['output_sha256']
  with av.open(str(video)) as con:frames=[f.to_image() for f in con.decode(video=0)]
  assert len(frames)==49
  condition=read(ROOT/'conditions'/sid/'condition.json');header=180
  sheet=Image.new('RGB',(1800,header+4*210),'#20242a');draw=ImageDraw.Draw(sheet)
  for i,ref in enumerate(item['references']):sheet.paste(ImageOps.contain(Image.open(ref['absolute_path']).convert('RGB'),(180,165)),(i*190,10))
  left=max(10,len(item['references'])*190)
  draw.text((left,8),sid+' / '+a.model+' / '+item['group']+' / '+str(item.get('subgroup')),fill='white')
  for j,line in enumerate(textwrap.wrap(item['prompt']['en'],width=150)):draw.text((left,35+j*17),line,fill='white')
  crops=[]
  for col,f in enumerate([0,6,12,18,24,30,36,42,48]):
   source=Image.open(ROOT/'inputs/prepared'/sid/f'{f:03d}.png').convert('RGB');mask=np.asarray(Image.open(condition['mask_frames'][f]))>127;ys,xs=np.nonzero(mask);x1,x2=xs.min(),xs.max()+1;y1,y2=ys.min(),ys.max()+1;w=x2-x1;h=y2-y1
   box=(max(0,x1-w*.25),max(0,y1-h*.25),min(source.width,x2+w*.25),min(source.height,y2+h*.25));crops.append([round(x) for x in box])
   for row,(label,image) in enumerate([('SOURCE full',source),(a.model+' full',frames[f]),('SOURCE edit detail',source.crop(box)),(a.model+' edit detail',frames[f].crop(box))]):
    x=col*200;y=header+row*210;sheet.paste(ImageOps.contain(image,(195,180)),(x,y+23));draw.text((x+2,y+2),f'{label} {f/12:g}s',fill='white')
  sheet.save(out,quality=96)
  atomic(status,dict(id=sid,model=a.model,output_sha256=r['output_sha256'],crop_boxes=crops,source='Frozen source frames; source-only mask bounds used for display crops; model input/output pixels unchanged',script_sha256=sha(__file__)))
  print(sid,a.model,out,flush=True);count+=1
  if count>=a.limit:break
 print('New contact sheets',count,flush=True)
if __name__=='__main__':main()
