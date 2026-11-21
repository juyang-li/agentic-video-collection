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
 for r in read(ROOT/'metadata/portrait_candidates.json'):
  if int(Path(r['path']).stem)<=24:continue
  path=Path(r['path']);image=Image.open(path).convert('RGB');assert image.size==(1024,1024);assert sha(path)==r['sha256'];raw=np.asarray(image);mask=parser.segment(image,list(range(1,18)),main_person=True)>0
  n=int(mask[0].sum());audit.append(dict(id=path.stem,top_foreground_pixels=n,foreground_fraction=float(mask.mean())))
  if n or mask.mean()<.15:continue
  out=np.full_like(raw,128);out[mask]=raw[mask];dst=ROOT/'references'/f'{path.stem}_hair_gray128.png';mp=dst.with_name(dst.stem+'_foreground.png');Image.fromarray(out).save(dst);Image.fromarray(mask.astype(np.uint8)*255).save(mp)
  assert np.array_equal(out[mask],raw[mask])and np.all(out[~mask]==128)
  passing.append(dict(**r,processed_path=str(dst),processed_sha256=sha(dst),foreground_mask=str(mp),foreground_mask_sha256=sha(mp),dimensions=[1024,1024],parser=ready['human_parsing'],pixel_verification=dict(background_rgb=[128]*3,foreground_bitwise_original=True,natural_hair_preserved=True,no_crop_or_resize=True),visual_review='pending',name=path.stem,role='hair'))
  print('Full-head candidate',path.stem,flush=True)
 atomic(ROOT/'metadata/celeb_candidate_screen.json',dict(audit=audit,passing=passing,device='cpu',no_vlm_api=True,completed_unix=time.time()))
 im=Image.new('RGB',(1600,((len(passing)+3)//4)*420),'#222');d=ImageDraw.Draw(im)
 for i,r in enumerate(passing):
  x=(i%4)*400;y=(i//4)*420;im.paste(ImageOps.contain(Image.open(r['processed_path']).convert('RGB'),(395,390)),(x,y+25));d.text((x+4,y+3),r['name'],fill='white')
 im.save(ROOT/'review/celeb_new_fullhead_candidates.jpg',quality=96)
 print('CPU reference preparation complete',len(passing),flush=True)
if __name__=='__main__':main()
