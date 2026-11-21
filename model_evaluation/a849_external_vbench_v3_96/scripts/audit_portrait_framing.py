"""Source-only segmentation boundaries for the direct full-head reference audit."""
import sys
from common import *
def main():
 require_job()
 import numpy as np
 from PIL import Image,ImageDraw,ImageOps
 sys.path.insert(0,str(ROOT.parent/'ltx25_vace_a100_20261111/scripts'))
 from human_parsing import HumanParser
 ready=read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json');parser=HumanParser(ready['human_parsing']);records=[]
 candidates=read(ROOT/'metadata/portrait_candidates.json')
 for r in candidates:
  im=Image.open(r['path']).convert('RGB');m=parser.segment(im,list(range(1,18)),main_person=True)>0
  record=dict(**r,top_foreground_pixels=int(m[0].sum()),upper_left_foreground_pixels=int(m[:650,0].sum()),upper_right_foreground_pixels=int(m[:650,-1].sum()))
  records.append(record);print(Path(r['path']).name,record['top_foreground_pixels'],record['upper_left_foreground_pixels'],record['upper_right_foreground_pixels'],flush=True)
 atomic(ROOT/'metadata/portrait_framing_audit.json',records)
 selected=[r for r in records if r['top_foreground_pixels']==0]
 sheet=Image.new('RGB',(1600,((len(selected)+3)//4)*430),'#20242a');d=ImageDraw.Draw(sheet)
 for i,r in enumerate(selected):
  x=i%4*400;y=i//4*430;sheet.paste(ImageOps.contain(Image.open(r['path']),(395,395)),(x,y+30));d.text((x+4,y+5),Path(r['path']).name+f" edges:{r['upper_left_foreground_pixels']},{r['upper_right_foreground_pixels']}",fill='white')
 sheet.save(ROOT/'review/fullhead_candidates.jpg',quality=97)
if __name__=='__main__':main()
