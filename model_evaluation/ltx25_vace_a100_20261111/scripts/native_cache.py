"""Inference-only cache projection; no target tensor is loaded or encoded."""
import argparse
import sys
import time
from common import *
sys.path.insert(0,str(NATIVE/'scripts/common'))
from project_layout import resolve_path, add_script_paths
add_script_paths()


def make_index():
    inputs=rows(ROOT/'metadata/inputs.jsonl')
    index={}
    for src in read(BUNDLE)['sources']:
        metadata=read(src['test_completion_metadata'])
        assert metadata['status']=='passed'
        assert metadata['weights']==read(NATIVE/'runtime/models/native_model_revision.json')
        samples={r['sample_id']:r for r in metadata['samples']}
        for item in inputs:
            if item['dataset']!=src['name']: continue
            sid=int(item['uid'].split(':')[1]);s=samples[sid]
            text=metadata['text'][s['text']]
            assert text['prompt_en']==item['prompt']['en']
            entry={'id':item['id'],'text':{'receipt':text,'path':str(resolve_path(src['encoding_root'])/text['artifact']['path'])},'references':[],'input_sha256':sha(ROOT/'metadata/inputs.jsonl')}
            for i,(given,cached) in enumerate(zip(item['references'],s['references'],strict=True)):
                assert (cached['type'],cached.get('view',''),cached['index'])==(given['type'],given.get('view',''),i)
                r=metadata['media'][cached['media_key']]
                assert r['media_sha256_verified']==given['sha256']
                entry['references'].append({'type':given['type'],'view':given.get('view',''),'index':i,'receipt':r,'path':str(resolve_path(src['encoding_root'])/r['artifact']['path'])})
            if item['dataset']!='ue':
                r=metadata['media'][s['source']]
                assert r['num_frames']==49 and r['media_sha256_verified']==item['source']['sha256']
                assert r['pixel_metadata']['temporal_window']['timestamps_seconds']==item['timestamps_seconds']
                entry['source']={'receipt':r,'path':str(resolve_path(src['encoding_root'])/r['artifact']['path'])}
            else:
                entry['source']={'path':str(ROOT/'cache'/f'{item["id"]}_source.pt'),'derived':True}
                for i,ref in enumerate(item['references']):
                    if ref['type']=='motion':
                        entry['references'][i]={'type':ref['type'],'view':ref.get('view',''),'index':i,'path':str(ROOT/'cache'/f'{item["id"]}_motion_{i}.pt'),'derived':True}
            index[item['id']]=entry
    atomic(ROOT/'metadata/native_cache_index.json',index)


def derive(shard):
    require_job(exact=True)
    import numpy as np
    import torch
    from ltx_trainer.model_loader import load_video_vae_encoder
    from encoding_helpers import weight_paths
    from media import video_frames
    weights_record,weights=weight_paths(NATIVE/'runtime/models/native_model_revision.json',NATIVE/'runtime/verification/native_weights_verified.json')
    index=read(ROOT/'metadata/native_cache_index.json')
    model=None
    for item in rows(ROOT/'metadata/inputs.jsonl')[shard::4]:
        entry=index[item['id']]
        for role,ref,record in [('source',entry['source'],item['source']),*[(f'ref_{i}',v,item['references'][i]) for i,v in enumerate(entry['references'])]]:
            if not ref.get('derived'):continue
            dst=Path(ref['path']);receipt=dst.with_suffix('.json')
            identity={'original_sha256':record['sha256'],'fps':12,'num_frames':49,'canvas_wh':item['canvas_wh'],'vae_record_sha256':sha(NATIVE/'runtime/models/native_model_revision.json'),'pixel_script_sha256':sha(ROOT/'scripts/media.py')}
            if receipt.is_file():
                old=read(receipt)
                assert old['identity']==identity and sha(dst)==old['sha256']
                continue
            start=time.monotonic()
            assert sha(record['absolute_path'])==record['sha256']
            if model is None:
                model=load_video_vae_encoder(weights['vae'],device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
            frames,pixels_audit=video_frames(item,record)
            px=torch.from_numpy(np.stack([np.asarray(f) for f in frames])).float().div_(255).sub_(.5).div_(.5).permute(3,0,1,2).unsqueeze(0).to('cuda',torch.bfloat16)
            with torch.inference_mode():latent=model(px)
            w,h=item['canvas_wh'];assert list(latent.shape)==[1,128,7,h//32,w//32]
            assert torch.isfinite(latent).all()
            payload={'latents':latent[0].cpu().contiguous(),'num_frames':7,'height':h//32,'width':w//32,'fps':12}
            dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix('.tmp.pt');torch.save(payload,tmp);tmp.replace(dst)
            atomic(receipt,{'status':'complete','identity':identity,'sha256':sha(dst),'pixel_audit':pixels_audit,'seconds':time.monotonic()-start})
            del px,latent,payload,frames
    atomic(ROOT/'metadata'/f'cache_shard{shard}.json',{'status':'complete'})


def checked_load(ref):
    import torch
    p=Path(ref['path'])
    receipt=read(p.with_suffix('.json')) if ref.get('derived') else ref['receipt']['artifact']
    assert sha(p)==receipt['sha256'],f'cache changed {p}'
    return torch.load(p,map_location='cpu',weights_only=True)


def load_sample(item):
    from compact_text_cache import unpack_text_features
    entry=read(ROOT/'metadata/native_cache_index.json')[item['id']]
    assert entry['input_sha256']==sha(ROOT/'metadata/inputs.jsonl')
    source=checked_load(entry['source'])
    w,h=item['canvas_wh'];assert list(source['latents'].shape)==[128,7,h//32,w//32]
    conditions=unpack_text_features(checked_load(entry['text']),expected_contract=entry['text']['receipt']['contract'])
    refs=[{k:r[k] for k in ('type','view','index')}|{'latent':checked_load(r)} for r in entry['references']]
    assert len(refs)==len(item['references'])
    return {'source':source,'references':refs,'conditions':conditions}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',action='store_true');p.add_argument('--shard',type=int,choices=range(4));a=p.parse_args()
    if a.index:make_index()
    else:derive(a.shard)
