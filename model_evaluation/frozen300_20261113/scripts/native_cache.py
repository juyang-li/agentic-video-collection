"""Project only input caches; derive changed video windows in an allocated GPU job."""
import argparse
import sys
import time
from common import *
sys.path.insert(0,str(NATIVE/'scripts/common'))
from project_layout import resolve_path,add_script_paths
add_script_paths()

def make_index():
 inputs=rows(ROOT/'package/inputs.jsonl');index={}
 for src in read(BUNDLE)['sources']:
  metadata=read(src['test_completion_metadata']);assert metadata['status']=='passed'
  assert metadata['weights']==read(NATIVE/'runtime/models/native_model_revision.json')
  samples={r['sample_id']:r for r in metadata['samples']}
  for item in inputs:
   if item['dataset']!=src['name']:continue
   s=samples[int(item['uid'].split(':')[1])];text=metadata['text'][s['text']]
   assert text['prompt_en']==item['prompt']['en']
   entry={'id':item['id'],'text':{'receipt':text,'path':str(resolve_path(src['encoding_root'])/text['artifact']['path'])},'references':[],'input_sha256':sha(ROOT/'package/inputs.jsonl')}
   def cached(m,key,role):
    r=metadata['media'][key];assert r['media_sha256_verified']==m['sha256']
    matched='sample_frame_indices' not in m
    if not matched:
     pixels=r.get('pixel_metadata',{})
     matched=r['num_frames']==item['num_frames'] and pixels.get('indices')==m['sample_frame_indices'] and pixels.get('timestamps_seconds')==item['timestamps_seconds']
    if matched:return {'path':str(resolve_path(src['encoding_root'])/r['artifact']['path']),'receipt':r}
    return {'path':str(ROOT/'cache'/f'{item["id"]}_{role}.pt'),'derived':True}
   entry['source']=cached(item['source'],s['source'],'source')
   for i,(m,ref) in enumerate(zip(item['references'],s['references'],strict=True)):
    assert (ref['index'],ref['type'],ref.get('view',''))==(i,m['type'],m.get('view',''))
    entry['references'].append(cached(m,ref['media_key'],f'ref_{i}')|{'index':i,'type':m['type'],'view':m.get('view','')})
   index[item['id']]=entry
 write(ROOT/'metadata/native_cache_index.json',index)
 print('INPUT_ONLY_INDEX',len(index),flush=True)

def video_frames(item,m):
 import av
 from PIL import Image
 from prepare_datamake_media import contain_image
 p=ROOT/'package'/m['path'];assert sha(p)==m['sha256']
 ids=m['sample_frame_indices'];needed=set(ids);images={}
 with av.open(str(p)) as c:
  c.streams.video[0].codec_context.thread_count=2
  for i,f in enumerate(c.decode(video=0)):
   if i in needed:images[i]=contain_image(f.to_image(),item['canvas_wh'],{'padding_rgb':[0,0,0]})[0]
   if i>=max(needed):break
 assert needed==images.keys()
 return [images[i] for i in ids]

def derive(shard):
 require_job();import torch,numpy as np
 from ltx_trainer.model_loader import load_video_vae_encoder
 from encoding_helpers import weight_paths
 _,weights=weight_paths(NATIVE/'runtime/models/native_model_revision.json',NATIVE/'runtime/verification/native_weights_verified.json')
 index=read(ROOT/'metadata/native_cache_index.json');model=None
 for item in rows(ROOT/'package/inputs.jsonl')[shard::4]:
  e=index[item['id']]
  for role,ref,m in [('source',e['source'],item['source']),*[(f'ref_{i}',r,item['references'][i]) for i,r in enumerate(e['references'])]]:
   if not ref.get('derived'):continue
   dst=Path(ref['path']);receipt=dst.with_suffix('.json')
   identity={'manifest':sha(ROOT/'package/inputs.jsonl'),'media':m['sha256'],'indices':m['sample_frame_indices'],'canvas_wh':item['canvas_wh'],'pixel_script_sha256':sha(__file__)}
   if receipt.exists():
    old=read(receipt);assert old['identity']==identity and sha(dst)==old['sha256'];continue
   start=time.monotonic()
   if model is None:model=load_video_vae_encoder(weights['vae'],device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
   frames=video_frames(item,m)
   px=torch.from_numpy(np.stack([np.asarray(f) for f in frames])).float().div_(255).sub_(.5).div_(.5).permute(3,0,1,2).unsqueeze(0).to('cuda',torch.bfloat16)
   with torch.inference_mode():latent=model(px)
   w,h=item['canvas_wh'];assert list(latent.shape)==[1,128,(item['num_frames']-1)//8+1,h//32,w//32] and torch.isfinite(latent).all()
   payload={'latents':latent[0].cpu().contiguous(),'num_frames':latent.shape[2],'height':h//32,'width':w//32,'fps':12}
   dst.parent.mkdir(exist_ok=True);tmp=dst.with_suffix('.tmp.pt');torch.save(payload,tmp);tmp.replace(dst)
   write(receipt,{'status':'complete','identity':identity,'sha256':sha(dst),'seconds':time.monotonic()-start});print(item['id'],role,'encoded',flush=True)
   del px,latent,payload,frames

def checked_load(r):
 import torch
 p=Path(r['path']);receipt=read(p.with_suffix('.json')) if r.get('derived') else r['receipt']['artifact']
 assert sha(p)==receipt['sha256'],p
 return torch.load(p,map_location='cpu',weights_only=True)

def load_sample(item):
 from compact_text_cache import unpack_text_features
 entry=read(ROOT/'metadata/native_cache_index.json')[item['id']];assert entry['input_sha256']==sha(ROOT/'package/inputs.jsonl')
 source=checked_load(entry['source']);w,h=item['canvas_wh']
 assert list(source['latents'].shape)==[128,(item['num_frames']-1)//8+1,h//32,w//32]
 conditions=unpack_text_features(checked_load(entry['text']),expected_contract=entry['text']['receipt']['contract'])
 refs=[{k:r[k] for k in ['index','type','view']}|{'latent':checked_load(r)} for r in entry['references']]
 return {'source':source,'references':refs,'conditions':conditions}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--index',action='store_true');p.add_argument('--shard',type=int,choices=range(4));a=p.parse_args()
 if a.index:make_index()
 else:derive(a.shard)
