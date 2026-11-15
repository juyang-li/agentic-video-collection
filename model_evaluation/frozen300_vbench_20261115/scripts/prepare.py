"""Login: exact native media verification, H3 projections and frozen SOURCE clips."""
from common import *
import av,numpy as np,time,subprocess,concurrent.futures as cf
from PIL import Image

def pixel(frame):return hashlib.sha256(frame.tobytes()).hexdigest()
def derived(path,frames):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.part.mp4')
 with av.open(str(tmp),'w') as c:
  s=c.add_stream('libx264rgb',rate=12);s.width=frames[0].shape[1];s.height=frames[0].shape[0];s.pix_fmt='rgb24';s.thread_count=2;s.options={'crf':'0','preset':'fast'}
  for a in frames:
   for p in s.encode(av.VideoFrame.from_ndarray(a,format='rgb24')):c.mux(p)
  for p in s.encode():c.mux(p)
 with av.open(str(tmp)) as c:after=[pixel(f.to_ndarray(format='rgb24')) for f in c.decode(video=0)]
 assert after==[pixel(a) for a in frames]
 tmp.replace(path)
 return {'path':str(path.resolve()),'sha256':sha(path),'pixel_hashes':after,'pixel_preservation':True}

def one(item):
 sid=item['id'];out=[];receipt=ROOT/f'metadata/media/{sid}.json'
 if receipt.exists():
  old=read(receipt)
  for m in old['outputs']+[old['source']]:assert sha(m['path'])==m['output_sha256']
  return old
 for model in MODELS:
  if model=='ltx_distilled_v7':path=EVAL/f'ltx25_distilled_v7_300/outputs/{sid}.mp4'
  elif model=='a849_v7':path=EVAL/f'frozen300_20261113/outputs/a849/{sid}.mp4'
  else:path=METRICS/f'remote/results/{"h3" if model=="h3_pdd8" else "vace"}/{sid}.mp4'
  record=read(path.with_suffix('.json'));assert record['status']=='complete' and sha(path)==record['output_sha256']
  manifest=record.get('input_manifest_sha256',record.get('contract',{}).get('input_manifest_sha256'));assert manifest==sha(PKG/'inputs.jsonl')
  native_frames=0;selected=[];indices=[2*i for i in range(item['num_frames'])];wanted=set(indices);pts=[]
  wh=record.get('canvas_wh',record.get('contract',{}).get('canvas_wh'))
  with av.open(str(path)) as c:
   assert float(c.streams.video[0].average_rate)==(24 if model=='h3_pdd8' else 12)
   for i,f in enumerate(c.decode(video=0)):
    assert [f.width,f.height]==wh;native_frames+=1
    if model=='h3_pdd8' and i in wanted:
     assert abs(float(f.pts*f.time_base)-i/24)<1e-6;selected.append(f.to_ndarray(format='rgb24'));pts.append(float(f.pts*f.time_base))
  assert native_frames==record.get('num_frames',record.get('contract',{}).get('num_frames'))
  media={'id':sid,'uid':item['uid'],'group':item['group'],'dataset':item['dataset'],'model':model,'path':str(path.resolve()),'output_sha256':record['output_sha256'],'native_sha256':record['output_sha256'],'native_path':str(path.resolve()),'generation_status':'complete','num_frames':item['num_frames'],'fps':12,'canvas_wh':wh,'cohort':'frozen300','subgroup':None}
  if model=='h3_pdd8':
   assert len(selected)==item['num_frames'];dest=ROOT/f'media/h3/{sid}.mp4';d=derived(dest,selected)
   media.update(path=d['path'],output_sha256=d['sha256'],generated_frame_indices=indices,selected_pts=pts,pixel_preservation=True,pixel_hashes=d['pixel_hashes'])
  else:assert native_frames==item['num_frames']
  out.append(media)
 # Source only for official static intersection, not ground truth or similarity scoring.
 src=item['source'];path=PKG/src['path'];assert sha(path)==src['sha256'];wanted=set(src['sample_frame_indices']);frames={};cw,ch=item['canvas_wh']
 with av.open(str(path)) as c:
  c.streams.video[0].codec_context.thread_count=2
  for i,f in enumerate(c.decode(video=0)):
   if i in wanted:
    im=f.to_image();scale=min(cw/im.width,ch/im.height);w,h=round(im.width*scale),round(im.height*scale)
    canvas=Image.new('RGB',(cw,ch));canvas.paste(im.resize((w,h),Image.Resampling.LANCZOS),((cw-w)//2,(ch-h)//2));frames[i]=np.asarray(canvas)
 assert set(frames)==wanted
 d=derived(ROOT/f'media/source/{sid}.mp4',[frames[i] for i in src['sample_frame_indices']])
 source={'id':sid,'model':'source','path':d['path'],'output_sha256':d['sha256'],'source_original_sha256':src['sha256'],'pixel_preservation':True,'frames':item['num_frames'],'fps':12}
 result={'status':'complete','id':sid,'outputs':out,'source':source};atomic(receipt,result);print('MEDIA_READY',sid,flush=True);return result

def main():
 assert not os.environ.get('SLURM_JOB_ID')
 revision=subprocess.check_output(['git','-C',str(OLD/'runtime/VBench'),'rev-parse','HEAD'],text=True).strip();assert revision=='fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490'
 weights={str(p.relative_to(OLD/'runtime/vbench_models')):sha(p) for p in (OLD/'runtime/vbench_models').rglob('*') if p.is_file()}
 code={str(p.relative_to(OLD/'runtime/VBench')):sha(p) for p in (OLD/'runtime/VBench/vbench').rglob('*.py')}
 code['static_filter.py']=sha(OLD/'runtime/VBench/static_filter.py')
 implementation={'vbench_revision':revision,'official_sources_sha256':code,'weights_sha256':weights,'imaging_quality_preprocessing_mode':'longer','mode':'custom_input','input_manifest_sha256':sha(PKG/'inputs.jsonl')}
 identity=ROOT/'metadata/implementation.json'
 if identity.exists():assert read(identity)==implementation,'Pinned implementation or weights changed'
 else:atomic(identity,implementation)
 items=rows(PKG/'inputs.jsonl');assert len(items)==300
 workers=int(os.environ.get('VB_PREPARE_WORKERS','4'));assert 1<=workers<=4
 with cf.ThreadPoolExecutor(max_workers=workers) as pool:records=list(pool.map(one,items))
 outputs=[v for r in records for v in r['outputs']];assert len(outputs)==1200
 for name,values in [('inputs',items),('scoring_media',outputs),('source_media',[r['source'] for r in records])]:
  (ROOT/f'metadata/{name}.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
 atomic(ROOT/'metadata/preparation.json',{'status':'complete','outputs':1200,'sources':300,'h3_pixel_checks':300,'interval':'per-sample frozen interval; no tail','gt_used':False})
if __name__=='__main__':main()
