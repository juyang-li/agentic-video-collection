"""CPU/login local parsing; screen additional CelebA-HQ full-head hairstyle references."""
from common import *
import sys,time

def main():
 import torch,numpy as np
 from PIL import Image,ImageOps,ImageDraw
 assert not os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4)
 sys.path.insert(0,str(ROOT.parent/'ltx25_vace_a100_20261111/scripts'))
 from human_parsing import HumanParser
 ready=read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json');parser=HumanParser(ready['human_parsing'],device='cpu');audit=[];passing=[]
 prior=read(ROOT/'metadata/celeb_candidate_screen.json')if(ROOT/'metadata/celeb_candidate_screen.json').exists()else dict(audit=[],passing=[]);old_audit={r['id']:r for r in prior['audit']};old_passing={r['name']:r for r in prior['passing']};old_sources={Path(r['path']).stem:r for r in read(ROOT/'metadata/first240_portraits.json')}if(ROOT/'metadata/first240_portraits.json').exists()else{}
 for r in read(ROOT/'metadata/portrait_candidates.json'):
  if int(Path(r['path']).stem)<=0:continue
  sid=Path(r['path']).stem
  if sid in old_audit and sid in old_sources:
   assert sha(r['path'])==old_sources[sid]['sha256'];audit.append(old_audit[sid])
   if sid in old_passing:
    old=old_passing[sid];assert old.get('face_interior_repair')and sha(old['processed_path'])==old['processed_sha256'];passing.append(old)
   continue
  path=Path(r['path']);image=Image.open(path).convert('RGB');assert image.size==(1024,1024);assert sha(path)==r['sha256'];raw=np.asarray(image)
  # Keep actual facial pixels in holes that the parser mistakes for background (notably eyes).
  import cv2
  with torch.inference_mode():
   tensors=parser.processor(images=image,return_tensors='pt').to('cpu');logits=parser.model(**tensors).logits;values=torch.nn.functional.interpolate(logits,size=image.size[::-1],mode='bilinear',align_corners=False).argmax(dim=1)[0].numpy()
  count,components,stats,_=cv2.connectedComponentsWithStats((values!=0).astype(np.uint8),connectivity=8);assert count>1;largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));mask=np.isin(values,list(range(1,18)))&(components==largest);face=((values==11)&(components==largest)).astype(np.uint8);points=cv2.findNonZero(face)
  if points is not None:
   hull=cv2.convexHull(points);restore=np.zeros_like(face);cv2.fillConvexPoly(restore,hull,1);mask|=restore>0
  n=int(mask[0].sum());audit.append(dict(id=path.stem,top_foreground_pixels=n,foreground_fraction=float(mask.mean())))
  if n or mask.mean()<.15:continue
  top_y=int(np.nonzero(mask)[0].min())
  if top_y<20:continue
  out=np.full_like(raw,128);out[mask]=raw[mask];dst=ROOT/'references'/f'{path.stem}_hair_gray128.png';mp=dst.with_name(dst.stem+'_foreground.png');Image.fromarray(out).save(dst);Image.fromarray(mask.astype(np.uint8)*255).save(mp)
  assert np.array_equal(out[mask],raw[mask])and np.all(out[~mask]==128)
  passing.append(dict(**r,processed_path=str(dst),processed_sha256=sha(dst),foreground_mask=str(mp),foreground_mask_sha256=sha(mp),dimensions=[1024,1024],parser=ready['human_parsing'],pixel_verification=dict(background_rgb=[128]*3,foreground_bitwise_original=True,natural_hair_preserved=True,no_crop_or_resize=True),face_interior_repair='Original pixels restored inside parsed face convex hull; no generative edits',head_top_margin=top_y,visual_review='pending',name=path.stem,role='hair'))
  print('Full-head candidate',path.stem,flush=True)
 atomic(ROOT/'metadata/celeb_candidate_screen.json',dict(audit=audit,passing=passing,device='cpu',no_vlm_api=True,completed_unix=time.time()))
 im=Image.new('RGB',(1600,((len(passing)+3)//4)*420),'#222');d=ImageDraw.Draw(im)
 for i,r in enumerate(passing):
  x=(i%4)*400;y=(i//4)*420;im.paste(ImageOps.contain(Image.open(r['processed_path']).convert('RGB'),(395,390)),(x,y+25));d.text((x+4,y+3),r['name'],fill='white')
 im.save(ROOT/'review/celeb_new_fullhead_candidates.jpg',quality=96)
 print('CPU reference preparation complete',len(passing),flush=True)
if __name__=='__main__':main()
