"""Freeze verified training cache projection without loading target tensors."""
from common import *
import copy

def main():
    require_job(exact=True)
    from native_cache import checked_load,load_sample
    from media import profile
    old=ROOT.with_name('ltx25_vace_a100_20261111')
    original={r['id']:r for r in rows(old/'metadata/inputs.jsonl')}
    inputs=rows(ROOT/'metadata/inputs.jsonl')
    src=read(BUNDLE)['sources'][0];assert src['name']=='ue'
    metadata=read(src['test_completion_metadata']);assert metadata['status']=='passed'
    assert metadata['weights']==read(NATIVE/'runtime/models/native_model_revision.json')
    samples={r['sample_id']:r for r in metadata['samples']};index={}
    for item in inputs:
        previous=original[item['id']]
        assert {k:v for k,v in item.items() if k not in ('num_frames','timestamps_seconds')}=={k:v for k,v in previous.items() if k not in ('num_frames','timestamps_seconds')}
        sample=samples[int(item['uid'].split(':')[1])]
        text=metadata['text'][sample['text']];assert text['prompt_en']==item['prompt']['en']
        entry={'id':item['id'],'input_sha256':sha(ROOT/'metadata/inputs.jsonl'),'text':{'receipt':text,'path':str(Path(src['encoding_root'])/text['artifact']['path'])},'references':[]}
        def media(record,key):
            r=metadata['media'][key]
            assert r['status']=='passed' and r['num_frames']==121
            assert r['media_sha256_verified']==record['sha256']==sha(record['absolute_path'])
            assert r['media_path']==record['path'] and r['canvas_wh']==item['canvas_wh']
            assert r['pixel_metadata']['timestamps_seconds']==item['timestamps_seconds']
            assert r['pixel_metadata']['indices'][-1]==299 and r['pixel_metadata']['hold_last_at_endpoint']
            w,h=item['canvas_wh'];assert r['tensor']['shape']==[1,128,16,h//32,w//32]
            return {'receipt':r,'path':str(Path(src['encoding_root'])/r['artifact']['path'])}
        entry['source']=media(item['source'],sample['source'])
        for i,(given,cached) in enumerate(zip(item['references'],sample['references'],strict=True)):
            assert given['type']=='motion' and (cached['index'],cached['type'],cached['view'])==(i,'motion',given.get('view',''))
            entry['references'].append(media(given,cached['media_key'])|{'index':i,'type':given['type'],'view':given.get('view','')})
        assert profile('ue')['num_frames']==121
        index[item['id']]=entry
    atomic(ROOT/'metadata/native_cache_index.json',index)
    for item in inputs:load_sample(item)
    atomic(ROOT/'metadata/preflight.json',{'status':'passed','samples':10,'source_and_motion_frames':121,'latent_frames':16,'fps':12,'cache':'training artifacts reused after identity, pixel protocol, hash and tensor checks','historical_manifest_sha256':sha(old/'metadata/inputs.jsonl'),'input_sha256':sha(ROOT/'metadata/inputs.jsonl'),'no_gt_inference':True})

if __name__=='__main__':main()
