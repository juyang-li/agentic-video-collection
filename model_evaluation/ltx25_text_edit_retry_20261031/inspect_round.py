from pathlib import Path
import argparse,json,os
import av,numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent

def read_frames(path,indices):
 result={}
 with av.open(str(path)) as c:
  for i,f in enumerate(c.decode(video=0)):
   if i in indices:result[i]=f.to_image()
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--round',type=int,required=True);a=p.parse_args()
 records=[]
 for label in ['target','empty']:
  sid=f'appearance_002_attempt{a.round}_{label}';d=json.loads((R/f'outputs/records/{sid}.json').read_text());assert d['status'] in ['generated','recovered_tmp'];records.append(d)
 n=records[0]['media']['frames'];indices=[0,(n-1)//2,n-1]
 os.environ['LTX_BENCH_ROOT']=str(R)
 from bench_nodes import BenchSource
 source=R.parents[1]/records[0]['source_video'];images,w,h,n=BenchSource().load(str(source),97,24.0)
 sources={i:Image.fromarray((images[i].numpy()*255).round().clip(0,255).astype('uint8')) for i in indices}
 target=read_frames(records[0]['output'],indices);empty=read_frames(records[1]['output'],indices)
 sheet=Image.new('RGB',(1152,750),'#202020');draw=ImageDraw.Draw(sheet)
 metrics=[]
 for row,i in enumerate(indices):
  for col,(frames,label) in enumerate([(sources,'SOURCE'),(target,'TARGET PROMPT'),(empty,'EMPTY PROMPT CONTROL')]):
   draw.text((col*384+6,row*250+4),f'round {a.round} frame {i} | {label}',fill='white')
   im=frames[i].copy();im.thumbnail((384,224));sheet.paste(im,(col*384+(384-im.width)//2,row*250+25))
  x,y,z=(np.asarray(frames[i],dtype=float) for frames in [sources,target,empty])
  metrics.append(dict(frame=i,source_target_mae=float(np.abs(x-y).mean()),target_empty_mae=float(np.abs(y-z).mean())))
 sheet.save(R/f'metadata/attempt{a.round}_comparison.jpg')
 (R/f'metadata/attempt{a.round}_pixel_diagnostics.json').write_text(json.dumps(metrics,indent=2)+'\n')
 print(json.dumps(dict(round=a.round,image=str(R/f'metadata/attempt{a.round}_comparison.jpg'),metrics=metrics)),flush=True)
if __name__=='__main__':main()
