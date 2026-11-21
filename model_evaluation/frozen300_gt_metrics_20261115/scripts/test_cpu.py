from common import *
from evaluate import geometry,frame_metrics,output_frames
import tempfile,av,numpy as np

def main():
 assert geometry(1920,1080,1280,736)==[0,8,1280,720]
 assert geometry(1080,1920,736,1280)==[8,0,720,1280]
 a=np.zeros((32,32,3),np.uint8);b=a+20;s,p,inf=frame_metrics((a,a));assert s==1 and inf and p is None
 s,p,inf=frame_metrics((a,b));assert s<1 and abs(p-20*np.log10(255/20))<1e-5
 with tempfile.TemporaryDirectory() as tmp:
  path=Path(tmp)/'test.mp4'
  with av.open(str(path),'w') as c:
   stream=c.add_stream('libx264rgb',rate=24);stream.width=64;stream.height=36;stream.pix_fmt='rgb24';stream.options={'crf':'0'}
   for i in range(27):
    f=av.VideoFrame.from_ndarray(np.full((36,64,3),i*5,dtype=np.uint8),format='rgb24')
    for packet in stream.encode(f):c.mux(packet)
   for packet in stream.encode():c.mux(packet)
  item={'num_frames':13,'source':{'original_wh':[64,36]}}
  frames,audit=output_frames(path,item,{'canvas_wh':[64,36],'num_frames':27,'fps':24},'h3_pdd8',(64,36))
  assert [r['index'] for r in audit['selected']]==list(range(0,25,2))
  assert len(frames)==13 and audit['native_frames']==27
  try:output_frames(path,item,{'canvas_wh':[60,36],'num_frames':27,'fps':24},'h3_pdd8',(64,36))
  except AssertionError:pass
  else:raise AssertionError('Bad geometry accepted')
 atomic(ROOT/'metadata/cpu_tests.json',{'status':'passed','geometry':True,'psnr_reference':True,'h3_time_mapping':True,'tail_excluded':True,'invalid_shape_rejected':True})
if __name__=='__main__':main()
