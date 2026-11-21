"""Read-only upstream inputs/caches; CPU verifies weights, media and actual latent tensors."""
from common import *
import time,collections,importlib.metadata,struct

def main():
    import torch
    from compact_text_cache import unpack_text_features
    started=time.time();items=rows(FROZEN/'inputs.jsonl');index=read(FROZEN_ROOT/'metadata/native_cache_index.json')
    assert len(items)==len({i['id'] for i in items})==300
    assert collections.Counter(i['dataset'] for i in items)=={'ue':200,'coinve':75,'tripvvt':25}
    target=ROOT/'metadata/inputs.jsonl';content=(FROZEN/'inputs.jsonl').read_bytes()
    if target.exists():assert target.read_bytes()==content
    else:target.write_bytes(content)
    digest=sha(target);checked={};audits=[];projected={}
    def verify_file(p,expected):
        p=Path(p)
        if str(p) not in checked:
            assert sha(p)==expected,str(p);checked[str(p)]=expected
        assert checked[str(p)]==expected
    def cache(ref):
        p=resolve_path(ref['path']);receipt=read(p.with_suffix('.json')) if ref.get('derived') else ref['receipt']['artifact']
        verify_file(p,receipt['sha256'])
        if ref.get('derived'):assert receipt['status']=='complete'
        return torch.load(p,map_location='cpu',weights_only=True),str(p)
    for item in items:
        sid=item['id'];e=index[sid];assert e['input_sha256']==digest
        assert item['fps']==12 and item['num_frames']%8==1
        assert item['canvas_wh'] in [[1280,736],[736,1280]]
        assert item['timestamps_seconds']==[i/12 for i in range(item['num_frames'])]
        if item['dataset']=='ue':assert item['num_frames']==121
        assert e['text']['receipt']['prompt_en']==item['prompt']['en']
        for m in [item['source'],*item['references']]:verify_file(FROZEN/m['path'],m['sha256'])
        text,_=cache(e['text']);features=unpack_text_features(text,expected_contract=e['text']['receipt']['contract'])
        assert torch.isfinite(features['video_prompt_embeds']).all()
        for role,ref,m in [('source',e['source'],item['source']),*[(f'ref_{n}',ref,item['references'][n]) for n,ref in enumerate(e['references'])]]:
            payload,path=cache(ref);v=payload['latents'];w,h=item['canvas_wh']
            assert v.dtype==torch.bfloat16 and torch.isfinite(v).all() and v.shape[0]==128
            isvideo=role=='source' or m['type']=='motion'
            if isvideo:
                assert list(v.shape)==[128,(item['num_frames']-1)//8+1,h//32,w//32]
                evidence=read(Path(path).with_suffix('.json'))['identity'] if ref.get('derived') else ref['receipt']
                if ref.get('derived'):
                    assert evidence['manifest']==digest and evidence['media']==m['sha256'] and evidence['indices']==m['sample_frame_indices'] and evidence['canvas_wh']==item['canvas_wh']
                else:
                    assert evidence['media_sha256_verified']==m['sha256']
                    assert evidence['pixel_metadata']['indices']==m['sample_frame_indices'] and evidence['pixel_metadata']['timestamps_seconds']==item['timestamps_seconds']
            else:
                assert v.shape[1]==1 and ref['receipt']['media_sha256_verified']==m['sha256']
            if role!='source':
                n=int(role.split('_')[1]);assert (ref['index'],ref['type'],ref.get('view',''))==(n,m['type'],m.get('view',''))
        assert len(e['references'])==len(item['references'])
        projected[sid]=e;audits.append({'id':sid,'uid':item['uid'],'num_frames':item['num_frames'],'ordered_references':[{'index':r['index'],'type':r['type'],'view':r.get('view',''),'sha256':m['sha256']} for r,m in zip(e['references'],item['references'])],'input_and_cache_checks':'passed'})
        if len(audits)%25==0:print('INPUTS_VERIFIED',len(audits),flush=True)
    atomic(ROOT/'metadata/cache_index.json',projected)
    # Full digests, not just header/stat checks. Reuse pilot compatibility evidence only after same-file verification.
    pilot=read(PILOT/'metadata/weights.json');assert pilot['status']=='passed'
    _,paths=weights();record,_=weights()
    verify_file(DISTILLED,DISTILLED_SHA)
    actual_files=[]
    for f in record['files']:
        p=paths[f['path'].split('/')[0]];verify_file(p,f['sha256']);actual_files.append({'path':str(p),'sha256':f['sha256'],'bytes':p.stat().st_size})
    # Pilot verified exact projections and same packed Gemma. Full hashes above prove byte identity now.
    assert {f['sha256'] for f in actual_files}=={f['sha256'] for f in pilot['shared_verified_files']}
    assert pilot['text_cache_compatibility']['status']=='passed'
    atomic(ROOT/'metadata/weights.json',{'status':'passed','revision':record['revision'],'dit_sha256':DISTILLED_SHA,'distilled_path':str(DISTILLED),'shared_files':actual_files,'text_cache_compatibility':pilot['text_cache_compatibility'],'pilot_evidence_sha256':sha(PILOT/'metadata/weights.json')})
    versions={p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','peft','safetensors','av','numpy']}
    atomic(ROOT/'metadata/preflight.json',{'status':'passed','input_manifest_sha256':digest,'samples':audits,'verified_files':checked,'software':versions,'seconds':time.time()-started,'pilot_output_reuse':False,'gt_loaded_for_inference':False})
    print('PREFLIGHT_PASSED',len(audits),flush=True)
if __name__=='__main__':main()
