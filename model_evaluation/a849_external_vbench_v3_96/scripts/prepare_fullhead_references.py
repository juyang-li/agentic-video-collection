"""GPU segmentation only; preserve every foreground pixel including hair."""
import sys
from common import *
def main():
 require_job()
 import numpy as np
 from PIL import Image,ImageOps,ImageDraw
 sys.path.insert(0,str(ROOT.parent/'ltx25_vace_a100_20261111/scripts'))
 from human_parsing import HumanParser
 ready=read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json')
 parser=HumanParser(ready['human_parsing']);records=[]
 sheet=Image.new('RGB',(1600,800),'#20242a');draw=ImageDraw.Draw(sheet)
 for i,r in enumerate(read(ROOT/'metadata/fullhead_portrait_sources.json')):
  original=Image.open(r['path']).convert('RGB');assert original.size==(1024,1024)
  raw=np.asarray(original);mask=parser.segment(original,list(range(1,18)),main_person=True)>0
  assert mask.mean()>.15 and not mask[0].any(), 'Reference crown touches top border'
  out=np.full_like(raw,128);out[mask]=raw[mask]
  assert np.array_equal(out[mask],raw[mask]) and np.all(out[~mask]==128)
  dst=ROOT/'references'/f"{Path(r['path']).stem}_identity_gray128.png"
  mp=dst.with_name(dst.stem+'_foreground.png')
  Image.fromarray(out).save(dst);Image.fromarray(mask.astype(np.uint8)*255).save(mp)
  records.append(dict(**r,processed_path=str(dst),processed_sha256=sha(dst),foreground_mask=str(mp),foreground_mask_sha256=sha(mp),dimensions=[1024,1024],parser=ready['human_parsing'],pixel_verification={'background_rgb':[128]*3,'foreground_bitwise_original':True,'natural_hair_preserved':True,'no_crop_or_resize':True},visual_review='pending'))
  for j,pic in enumerate([original,Image.fromarray(out)]):
   sheet.paste(ImageOps.contain(pic,(395,370)),(i*400,j*400+25));draw.text((i*400+4,j*400+4),Path(r['path']).stem+(' original' if j==0 else ' gray128'),fill='white')
 sheet.save(ROOT/'review/fullhead_reference_processing.jpg',quality=96)
 atomic(ROOT/'metadata/fullhead_reference_processing.json',records)
if __name__=='__main__':main()
