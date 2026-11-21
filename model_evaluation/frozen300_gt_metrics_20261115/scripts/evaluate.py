from common import *
import av,numpy as np,time,sys,math,concurrent.futures as cf,importlib.metadata
from PIL import Image
from skimage.metrics import structural_similarity
import torch,lpips

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
def location(model,sid):
 if model=='ltx_distilled_v7':p=EVAL/f'ltx25_distilled_v7_300/outputs/{sid}'
 elif model=='a849_v7':p=EVAL/f'frozen300_20261113/outputs/a849/{sid}'
 else:p=ROOT/f'remote/results/{"h3" if model=="h3_pdd8" else "vace"}/{sid}'
 return p.with_suffix('.mp4'),read(p.with_suffix('.json'))
def validate(item,model,p,d):
 assert d['status']=='complete' and d.get('uid',item['uid'])==item['uid']
 manifest=d.get('input_manifest_sha256',d.get('contract',{}).get('input_manifest_sha256'))
 assert manifest==sha(PKG/'inputs.jsonl')
 assert d.get('original_prompt',item['prompt']['en'])==item['prompt']['en']
 assert d.get('source_sha256',item['source']['sha256'])==item['source']['sha256']
 assert d.get('ordered_reference_sha256s',[r['sha256'] for r in item['references']])==[r['sha256'] for r in item['references']]
 assert sha(p)==d['output_sha256']

def main():
 assert os.environ.get('SLURM_JOB_PARTITION')=='testqueue' and torch.cuda.device_count()==1
 assert 'H200' in torch.cuda.get_device_name(0)
 config=read(ROOT/'metadata/protocol.json');configsha=sha(ROOT/'metadata/protocol.json')
 items=rows(PKG/'inputs.jsonl');gts={r['id']:r for r in rows(PKG/'ground_truth.jsonl')};byid={r['id']:r for r in items}
 network=lpips.LPIPS(net='alex',version='0.1').cuda().eval();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 atomic(ROOT/'metadata/environment.json',{'versions':{k:importlib.metadata.version(k) for k in ['torch','torchvision','lpips','scikit-image','numpy','av','Pillow']},'gpu':torch.cuda.get_device_name(0),'lpips_weights':{str(p):sha(p) for p in Path(lpips.__file__).parent.rglob('alex.pth')},'alexnet_sha256':sha(Path(os.environ['TORCH_HOME'])/'hub/checkpoints/alexnet-owt-7be5be79.pth')})
 # Kernel identity and controlled distortion checks.
 rng=np.random.default_rng(42);a=rng.integers(0,210,(128,128,3),dtype=np.uint8);b=a+20
 same=frame_metrics((a,a));diff=frame_metrics((a,b));assert same[0]==1 and same[2] and diff[0]<1
 x=torch.from_numpy(a.copy()).permute(2,0,1)[None].cuda().float()/127.5-1;y=torch.from_numpy(b.copy()).permute(2,0,1)[None].cuda().float()/127.5-1
 with torch.inference_mode():
  zero=network(x,x).item();dist=network(x,y).item();reverse=network(y,x).item();batch=network(torch.cat([x,x]),torch.cat([y,y])).flatten()
 assert abs(zero)<1e-6 and dist>zero and abs(dist-reverse)<1e-5 and bool((abs(batch-dist)<1e-5).all())
 atomic(ROOT/'metadata/kernel_tests.json',{'status':'passed','identity_ssim':same[0],'identity_psnr_infinite':same[2],'identity_lpips':zero,'lpips_batch_agreement':True})
 batchsize=4;start=time.time();completed=0;smokestart=start
 order=SMOKE+[r['id'] for r in items if r['id'] not in SMOKE]
 with cf.ThreadPoolExecutor(max_workers=12) as pool:
  for sid in order:
   item=byid[sid];gt=gts[sid];assert gt['uid']==item['uid'];assert sha(PKG/gt['target']['path'])==gt['target']['sha256']
   size=(1280,720) if item['canvas_wh'][0]>item['canvas_wh'][1] else (720,1280)
   truth,gtpts=gt_frames(gt,size)
   for model in MODELS:
    path,meta=location(model,sid);validate(item,model,path,meta)
    receipt=ROOT/f'scores/{model}/{sid}.json'
    if receipt.exists():
     old=read(receipt);assert old['protocol_sha256']==configsha and old['output_sha256']==meta['output_sha256'] and old['gt_sha256']==gt['target']['sha256'];completed+=1;continue
    t=time.time();frames,alignment=output_frames(path,item,meta,model,size)
    cpu=read(ROOT/f'cpu_scores/{model}/{sid}.json')
    assert cpu['protocol_sha256']==configsha and cpu['output_sha256']==meta['output_sha256'] and cpu['gt_sha256']==gt['target']['sha256']
    basic=cpu['values'];assert len(basic)==len(frames)
    percept=[]
    pos=0
    while pos<len(frames):
     try:
      chunk=frames[pos:pos+batchsize];ref=truth[pos:pos+batchsize]
      x=torch.from_numpy(np.stack(chunk)).permute(0,3,1,2).cuda().float()/127.5-1;y=torch.from_numpy(np.stack(ref)).permute(0,3,1,2).cuda().float()/127.5-1
      with torch.inference_mode():v=network(x,y).flatten().cpu().tolist()
      assert all(math.isfinite(z) for z in v);percept+=v;pos+=len(chunk);del x,y
     except torch.cuda.OutOfMemoryError:
      if batchsize==1:raise
      batchsize//=2;torch.cuda.empty_cache()
    perframe=[{'frame':i,'ssim':s,'psnr_db':p,'psnr_infinite':inf,'lpips':l} for i,((s,p,inf),l) in enumerate(zip(basic,percept))]
    mean={'ssim':float(np.mean([v[0] for v in basic])),'psnr_db':None if any(v[2] for v in basic) else float(np.mean([v[1] for v in basic])),'psnr_infinite':any(v[2] for v in basic),'lpips':float(np.mean(percept))}
    atomic(receipt,{'status':'complete','model':model,'id':sid,'uid':item['uid'],'dataset':item['dataset'],'group':item['group'],'num_frames':len(frames),'mean':mean,'frames':perframe,'alignment':alignment,'gt_indices':gt['sample_frame_indices'],'gt_pts':gtpts,'gt_sha256':gt['target']['sha256'],'output_sha256':meta['output_sha256'],'protocol_sha256':configsha,'seconds':time.time()-t,'batch_size':batchsize})
    if sid in SMOKE:
     for index in [0,len(frames)//2,len(frames)-1]:
      preview=ROOT/f'preview/{model}/{sid}_{index}.jpg';preview.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.concatenate([truth[index],frames[index]],axis=1)).save(preview)
    completed+=1;atomic(ROOT/'metadata/run_status.json',{'status':'running','completed':completed,'total':1200,'current':sid,'model':model});print('COMPLETE',completed,model,sid,flush=True)
   if sid==SMOKE[-1]:
    elapsed=time.time()-smokestart;atomic(ROOT/'metadata/smoke.json',{'status':'passed','samples':SMOKE,'combinations':24,'seconds':elapsed,'estimated_total_seconds':elapsed/24*1200,'note':'same six samples across all models; alignment preview retained'})
 atomic(ROOT/'metadata/run_status.json',{'status':'complete','completed':1200,'seconds':time.time()-start})
 from report import main as report
 report()
if __name__=='__main__':
 try:main()
 except Exception as e:
  import traceback
  atomic(ROOT/'metadata/failure.json',{'status':'failed','error':traceback.format_exc()});raise
