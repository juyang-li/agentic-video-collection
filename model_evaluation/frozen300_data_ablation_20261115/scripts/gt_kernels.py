from common import PKG
import av,numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

def geometry(w,h,cw,ch):
 scale=min(cw/w,ch/h);rw=round(w*scale);rh=round(h*scale)
 return [(cw-rw)//2,(ch-rh)//2,rw,rh]

def decode(path):
 with av.open(str(path)) as c:
  s=c.streams.video[0];s.codec_context.thread_count=2
  for f in c.decode(s):
   assert f.pts is not None
   yield f,float(f.pts*f.time_base)

def normalize(frame,box,size):
 im=frame.to_image();x,y,w,h=box
 return np.asarray(im.crop((x,y,x+w,y+h)).resize(size,Image.Resampling.LANCZOS),dtype=np.uint8)

def frame_metrics(pair):
 a,b=pair;a=a.astype(np.float32)/255;b=b.astype(np.float32)/255
 mse=np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
 return float(structural_similarity(a,b,data_range=1.,channel_axis=-1,gaussian_weights=True,sigma=1.5,win_size=11,use_sample_covariance=False)),None if mse==0 else float(-10*np.log10(mse)),bool(mse==0)

def gt_frames(gt,size):
 indices=gt['sample_frame_indices'];wanted=set(indices);images={};pts={}
 for i,(f,t) in enumerate(decode(PKG/gt['target']['path'])):
  if i in wanted:images[i]=normalize(f,[0,0,f.width,f.height],size);pts[i]=t
 assert wanted<=images.keys()
 return [images[i] for i in indices],[pts[i] for i in indices]

def output_frames(path,item,metadata,model,size):
 # Validate entire native stream, then select by closest presentation time. Store RGB only for selected frames.
 wanted=[i/12 for i in range(item['num_frames'])];iterator=iter(decode(path));prev=None;current=next(iterator,None);chosen=[];audit=[];count=0;last=-1
 cw,ch=metadata['canvas_wh'] if 'canvas_wh' in metadata else metadata['contract']['canvas_wh']
 ow,oh=item['source']['original_wh'];box=geometry(ow,oh,cw,ch)
 for target in wanted:
  while current is not None and current[1]<target-1e-9:
   assert current[1]>last;last=current[1];assert (current[0].width,current[0].height)==(cw,ch)
   prev=(current[0],current[1],count);count+=1;current=next(iterator,None)
  candidates=([prev] if prev else [])+([(current[0],current[1],count)] if current else [])
  assert candidates
  chosen_frame=min(candidates,key=lambda z:(round(abs(z[1]-target),9),-z[1]))
  assert abs(chosen_frame[1]-target)<=1/48+1e-6 if model=='h3_pdd8' else abs(chosen_frame[1]-target)<=1e-5
  chosen.append(normalize(chosen_frame[0],box,size));audit.append({'target':target,'index':chosen_frame[2],'pts':chosen_frame[1]})
 while current is not None:
  assert current[1]>last;last=current[1];assert (current[0].width,current[0].height)==(cw,ch)
  count+=1;current=next(iterator,None)
 expected=metadata.get('num_frames',metadata.get('contract',{}).get('num_frames'));assert count==expected,(count,expected)
 assert last>=wanted[-1]-1e-6
 fps=metadata.get('fps',metadata.get('contract',{}).get('fps'));assert float(fps)==(24 if model=='h3_pdd8' else 12)
 return chosen,{'selected':audit,'crop_xywh':box,'native_frames':count,'last_pts':last}
