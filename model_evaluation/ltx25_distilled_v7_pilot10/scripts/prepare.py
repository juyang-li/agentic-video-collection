"""CPU-only frozen pairing, complete video decode, weight and cache integrity audit."""
from common import *
import struct, time, importlib.metadata

def header(p):
    with Path(p).open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(size))
def tensor_digests(p, predicate):
    h=header(p); result={}
    with Path(p).open('rb') as f:
        offset=8+len(json.dumps(h).encode()) # actual header length, including padding
        f.seek(0);offset=8+struct.unpack('<Q',f.read(8))[0]
        for k,v in h.items():
            if k=='__metadata__' or not predicate(k): continue
            a,b=v['data_offsets'];f.seek(offset+a);d=hashlib.sha256()
            left=b-a
            while left:
                buf=f.read(min(left,8*1024**2));assert buf;d.update(buf);left-=len(buf)
            result[k]={'sha256':d.hexdigest(),'shape':v['shape'],'dtype':v['dtype']}
    return result

def main():
    import av
    start=time.time();record,paths=weights()
    assert record['revision']=='49b093a39c19169a8a81721dcc541e07e945232f'
    all_inputs={r['id']:r for r in rows(BASE/'metadata/inputs.jsonl')}
    frozen={r['id']:r for r in rows(FROZEN/'inputs.jsonl')}
    index=read(BASE/'metadata/native_cache_index.json'); audits=[]
    for sid in IDS:
        item=all_inputs[sid];other=frozen[sid]
        assert item['uid']==other['uid'] and item['dataset']=='ue'
        for k in ['prompt','canvas_wh','num_frames','fps','timestamps_seconds']: assert item[k]==other[k],(sid,k)
        assert item['num_frames']==121 and item['fps']==12 and item['timestamps_seconds']==[i/12 for i in range(121)]
        assert item['canvas_wh'] in [[1280,736],[736,1280]]
        assert item['source']['sha256']==other['source']['sha256']
        assert [(r['sha256'],r['type'],r.get('view','')) for r in item['references']]==[(r['sha256'],r['type'],r.get('view','')) for r in other['references']]
        for media in [item['source'],*item['references']]: assert sha(media['absolute_path'])==media['sha256']
        entry=index[sid];assert entry['input_sha256']==sha(BASE/'metadata/inputs.jsonl')
        assert entry['text']['receipt']['prompt_en']==item['prompt']['en']
        caches=[]
        for c in [entry['source'],entry['text'],*entry['references']]:
            p=resolve_path(c['path']);receipt=read(p.with_suffix('.json')) if c.get('derived') else c['receipt']['artifact']
            assert sha(p)==receipt['sha256'];caches.append({'path':str(p),'sha256':receipt['sha256']})
        src_receipt=entry['source'].get('receipt',{})
        assert src_receipt.get('media_sha256_verified',item['source']['sha256'])==item['source']['sha256']
        for c,r in zip(entry['references'],item['references']): assert c['receipt']['media_sha256_verified']==r['sha256']
        status=read(BASE/f'outputs/ltx_base/{sid}.json');video=BASE/f'outputs/ltx_base/{sid}.mp4'
        assert status['status']=='complete' and sha(video)==status['output_sha256']
        assert status['contract']['seed']==42 and status['contract']['num_frames']==121 and status['contract']['canvas_wh']==item['canvas_wh']
        with av.open(str(video)) as container:
            n=0
            for frame in container.decode(video=0):
                assert [frame.width,frame.height]==item['canvas_wh'];n+=1
            assert n==121
        audits.append({'id':sid,'uid':item['uid'],'historical_status_path':str(video.with_suffix('.json')),'historical_status_sha256':sha(video.with_suffix('.json')),'historical_video_sha256':status['output_sha256'],'full_decode_frames':n,'cache_files':caches,'temporal_window':other['temporal_window'],'source_frame_indices':other['source']['sample_frame_indices']})
        print('paired',sid,flush=True)
    text=''.join(json.dumps(all_inputs[i],ensure_ascii=False)+'\n' for i in IDS)
    manifest=ROOT/'metadata/inputs.jsonl'
    if manifest.exists(): assert manifest.read_text()==text
    else: manifest.write_text(text)
    atomic(ROOT/'metadata/cache_index.json',{i:index[i] for i in IDS})
    atomic(ROOT/'metadata/pairing.json',{'status':'passed','samples':audits,'base_manifest_sha256':sha(BASE/'metadata/inputs.jsonl'),'frozen300_manifest_sha256':sha(FROZEN/'inputs.jsonl'),'input_manifest_sha256':sha(manifest)})
    print('hashing distilled weights',flush=True)
    digest=sha(DISTILLED);assert digest==DISTILLED_SHA
    verified=[]
    for f in record['files']:
        p=paths[f['path'].split('/')[0]];assert sha(p)==f['sha256'];verified.append({'path':str(p),**f})
    # Split LTX2.5 stores feature-extractor projections in the SAME packed Gemma file.
    # Inspect both transformers to rule out any additional projection weights there.
    dh,bh=header(DISTILLED),header(paths['diffusion_models'])
    projection=lambda k: 'text_embedding_projection' in k or 'feature_extractor' in k
    dproj=tensor_digests(DISTILLED,projection);bproj=tensor_digests(paths['diffusion_models'],projection)
    assert dproj==bproj,'Feature extractor differs: regenerate ten text features before inference'
    gemma_projection=tensor_digests(paths['text_encoders'],projection)
    assert gemma_projection,'No projection weights found in packed Gemma'
    atomic(ROOT/'metadata/weights.json',{'status':'passed','repo_id':record['repo_id'],'revision':record['revision'],'distilled':{'path':str(DISTILLED),'sha256':digest,'bytes':DISTILLED.stat().st_size,'metadata':dh.get('__metadata__')},'shared_verified_files':verified,'text_cache_compatibility':{'status':'passed','stage':'after_feature_extractor_before_create_embeddings','same_packed_gemma_file':str(paths['text_encoders']),'gemma_projection_tensors':gemma_projection,'dev_transformer_projection_tensors':bproj,'distilled_transformer_projection_tensors':dproj,'connector':'loaded afresh from distilled transformer; historical post-connector values never reused'}})
    versions={p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','peft','safetensors','av','numpy']}
    atomic(ROOT/'metadata/software_cpu.json',versions)
    print('CPU pairing/weights passed',time.time()-start,flush=True)
if __name__=='__main__': main()
