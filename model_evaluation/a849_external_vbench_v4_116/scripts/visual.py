"""Import exact-output observations, build temporal evidence, record direct human/model inspection."""
import argparse,time,textwrap
from common import *
from score import MODELS

def sync():
 manifest=sha(ROOT/'metadata/inputs.jsonl');path=ROOT/'metadata/visual_observations.json'
 current=read(path)if path.exists()else dict(input_manifest_sha256=manifest,samples={})
 assert current['input_manifest_sha256']==manifest
 for item in rows(ROOT/'metadata/inputs.jsonl'):current['samples'].setdefault(item['id'],dict(models={}))
 origins=read(ROOT/'metadata/input_origins.json')
 for item in rows(ROOT/'metadata/inputs.jsonl'):
  sid=item['id'];parent,old_id=origins[sid];p=Path(parent)/'metadata/visual_observations.json'
  if not p.exists():continue
  sample=read(p)['samples'].get(old_id,{})
  for model,note in sample.get('models',{}).items():
   if not note.get('reviewed'):continue
   r=ROOT/'outputs'/model/f'{sid}.json'
   if not r.exists():continue
   receipt=read(r)
   if receipt['status']=='complete'and note.get('output_sha256')==receipt['output_sha256']:
    existing=current['samples'][sid]['models'].get(model,{})
    if existing.get('reviewed')and existing.get('output_sha256')==receipt['output_sha256']:continue
    current['samples'][sid]['models'][model]=dict(note,imported_from=str(p),imported_case_id=old_id,imported_observations_sha256=sha(p))
 count=0
 for sid,sample in current['samples'].items():
  for m,note in sample['models'].items():
   r=ROOT/'outputs'/m/f'{sid}.json'
   if note.get('reviewed')and r.exists()and read(r).get('output_sha256')==note.get('output_sha256'):count+=1
 current.update(inspection_complete=count==len(current['samples'])*len(MODELS),reviewed_outputs=count,expected_outputs=len(current['samples'])*len(MODELS),summary=f"Direct observations {count}/{len(current['samples'])*len(MODELS)}; pending is not passed",updated_unix=time.time());atomic(path,current);return current

def contacts(model,limit):
 import av,numpy as np
 from PIL import Image,ImageOps,ImageDraw
 observations=sync();dest=ROOT/'review/new_contacts';dest.mkdir(parents=True,exist_ok=True);n=0
 for item in rows(ROOT/'metadata/inputs.jsonl'):
  sid=item['id'];p=ROOT/'outputs'/model/f'{sid}.mp4';r=p.with_suffix('.json')
  if not r.exists()or read(r)['status']!='complete':continue
  receipt=read(r);h=receipt['output_sha256'];note=observations['samples'][sid]['models'].get(model,{})
  if note.get('reviewed')and note.get('output_sha256')==h:continue
  out=dest/f'{model}--{sid}.jpg'
  if out.exists()and out.with_suffix('.json').exists()and read(out.with_suffix('.json'))['output_sha256']==h:continue
  assert sha(p)==h
  with av.open(str(p))as c:frames=[f.to_image()for f in c.decode(video=0)]
  assert len(frames)==49
  cond=read(ROOT/'conditions'/sid/'condition.json');header=180;sheet=Image.new('RGB',(1800,header+4*210),'#20242a');draw=ImageDraw.Draw(sheet)
  for i,ref in enumerate(item['references']):sheet.paste(ImageOps.contain(Image.open(ref['absolute_path']).convert('RGB'),(180,165)),(190*i,10))
  left=max(10,len(item['references'])*190);draw.text((left,8),sid+' / '+model+' / '+item['group'],fill='white')
  for j,line in enumerate(textwrap.wrap(item['prompt']['en'],width=145)):draw.text((left,35+j*17),line,fill='white')
  for col,f in enumerate([0,6,12,18,24,30,36,42,48]):
   source=Image.open(ROOT/'inputs/prepared'/sid/f'{f:03d}.png').convert('RGB');mask=np.asarray(Image.open(cond['mask_frames'][f]))>127;ys,xs=np.nonzero(mask)
   if len(xs):
    x1,x2=xs.min(),xs.max()+1;y1,y2=ys.min(),ys.max()+1;w=x2-x1;hgt=y2-y1;box=(max(0,x1-w*.25),max(0,y1-hgt*.25),min(source.width,x2+w*.25),min(source.height,y2+hgt*.25))
   else:box=(0,0,*source.size)
   output=frames[f];scale=min(output.width/source.width,output.height/source.height);ox=(output.width-round(source.width*scale))/2;oy=(output.height-round(source.height*scale))/2;obox=(box[0]*scale+ox,box[1]*scale+oy,box[2]*scale+ox,box[3]*scale+oy)
   for row,(label,img)in enumerate([('SOURCE full',source),(model+' full',output),('SOURCE edit detail',source.crop(box)),(model+' edit detail',output.crop(obox))]):
    x=col*200;y=header+row*210;sheet.paste(ImageOps.contain(img,(195,180)),(x,y+23));draw.text((x+2,y+2),f'{label} {f/12:g}s',fill='white')
  sheet.save(out,quality=96);atomic(out.with_suffix('.json'),dict(id=sid,model=model,output_sha256=receipt['output_sha256'],frame_indices=[0,6,12,18,24,30,36,42,48],note='Full frames + source-mask display crops. H3 cropping mapped by input letterbox; full frames remain primary evidence. No output manipulation.'));print(out,flush=True);n+=1
  if n>=limit:break
 print('New contacts',n,flush=True)

def record(path):
 observations=sync()
 for note in read(path):
  sid=note['id'];model=note['model'];assert model in MODELS
  assert note['edit']in['success','partial','failed','uncertain']and note['preservation']in['good','partial','poor']
  receipt=read(ROOT/'outputs'/model/f'{sid}.json');assert receipt['status']=='complete'
  evidence=ROOT/'review/new_contacts'/f'{model}--{sid}.jpg';assert evidence.is_file()and read(evidence.with_suffix('.json'))['output_sha256']==receipt['output_sha256']
  observations['samples'][sid]['models'][model]=dict({k:v for k,v in note.items()if k not in['id','model']},reviewed=True,output_sha256=receipt['output_sha256'],evidence=str(evidence),evidence_sha256=sha(evidence),method='Direct inspection of full source/output and reference across nine timestamps 0–4s; no external VLM API',manual_notes_source=str(path),manual_notes_sha256=sha(path),reviewed_unix=time.time())
 atomic(ROOT/'metadata/visual_observations.json',observations);sync()
if __name__=='__main__':
 import fcntl
 lock=(ROOT/'runtime/visual.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['sync','contacts','record']);p.add_argument('--model',choices=MODELS);p.add_argument('--limit',type=int,default=999);p.add_argument('--notes',type=Path);a=p.parse_args()
 if a.mode=='sync':print('Direct observations',sync()['reviewed_outputs'],'/464',flush=True)
 elif a.mode=='contacts':assert a.model;contacts(a.model,a.limit)
 else:assert a.notes;record(a.notes)
