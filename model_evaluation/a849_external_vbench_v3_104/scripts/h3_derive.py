"""CPU only: actual generated0..4s frames -> lossless scoring clip and H264 browser preview."""
import argparse,time
from common import *

def main():
 import av,hashlib,numpy as np
 parser=argparse.ArgumentParser();parser.add_argument('--loop',action='store_true');parser.add_argument('--interval',type=int,default=60);a=parser.parse_args()
 assert not os.environ.get('SLURM_JOB_ID'),'Run pure CPU projections on login'
 items=rows(ROOT/'metadata/inputs.jsonl')
 while True:
  done=0
  for item in items:
   sid=item['id'];native=ROOT/'outputs/h3_pdd8_native'/f'{sid}.mp4';rpath=native.with_suffix('.json')
   if not rpath.exists()or read(rpath)['status']!='complete':continue
   r=read(rpath);assert sha(native)==r['output_sha256'];out=ROOT/'outputs/h3_pdd8'/f'{sid}.mp4';receipt=out.with_suffix('.json');preview=ROOT/'review/h3_browser'/f'{sid}.mp4'
   if receipt.exists():
    prior=read(receipt);assert prior['status']=='complete'and prior['native_sha256']==r['output_sha256']and sha(out)==prior['output_sha256']and sha(preview)==prior['browser_sha256'];done+=1;continue
   with av.open(str(native))as c:
    stream=c.streams.video[0];assert stream.average_rate==24;frames=[f.to_image()for f in c.decode(video=0)]
   assert len(frames)==107;indices=[2*n for n in range(49)];selected=[frames[n]for n in indices];pixels=[hashlib.sha256(np.asarray(f).tobytes()).hexdigest()for f in selected]
   out.parent.mkdir(parents=True,exist_ok=True);preview.parent.mkdir(parents=True,exist_ok=True)
   def encode(path,codec,pix_fmt,options):
    tmp=path.with_suffix('.part.mp4')
    with av.open(str(tmp),'w')as c:
     s=c.add_stream(codec,rate=12);s.width,s.height=selected[0].size;s.pix_fmt=pix_fmt;s.thread_count=2;s.options=options
     for frame in selected:
      for packet in s.encode(av.VideoFrame.from_image(frame)):c.mux(packet)
     for packet in s.encode():c.mux(packet)
    tmp.replace(path)
   encode(out,'libx264rgb','rgb24',{'crf':'0','preset':'fast'});encode(preview,'libx264','yuv420p',{'crf':'18','preset':'fast'})
   with av.open(str(out))as c:
    decoded=[hashlib.sha256(f.to_ndarray(format='rgb24').tobytes()).hexdigest()for f in c.decode(video=0)]
   assert decoded==pixels
   atomic(receipt,dict(id=sid,model='h3_pdd8',status='complete',output_sha256=sha(out),native_sha256=r['output_sha256'],native_path=str(native),browser_path=str(preview),browser_sha256=sha(preview),input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),contract=r['contract'],comparison=dict(frames=49,fps=12,generated_frame_indices=indices,canvas_wh=list(selected[0].size),lossless_pixel_preservation=True,no_interpolation=True,no_repeated_or_frozen_generated_frames=True),full_decode_frames=49,completed_unix=time.time()));done+=1;print('H3 scoring/browser projected',sid,flush=True)
  atomic(ROOT/'metadata/h3_derivation.json',dict(status='complete'if done==104 else'pending',complete=done,expected=104,checked_unix=time.time()))
  if done==104 or not a.loop:break
  time.sleep(a.interval)
if __name__=='__main__':main()
