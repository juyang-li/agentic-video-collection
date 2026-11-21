"""Freeze exact historical identities and full-duration assets; project input-only caches."""
from common import *
import copy, shutil, time

def main():
    require_job(exact=True)
    old=ROOT.with_name('ltx25_vace_a100_20261111')
    originals=rows(old/'metadata/inputs.jsonl')
    jsonl(ROOT/'metadata/historical_inputs.jsonl',originals)
    selected=[]; exceptions=[]; index={}; assets={}
    src=next(s for s in read(BUNDLE)['sources'] if s['name']=='ue')
    m=read(src['test_completion_metadata']); assert m['status']=='passed'
    assert m['weights']==read(NATIVE/'runtime/models/native_model_revision.json')
    samples={s['sample_id']:s for s in m['samples']}
    def freeze(record):
        r=copy.deepcopy(record); path=Path(r['absolute_path']); digest=r['sha256']
        assert sha(path)==digest,path
        dst=ROOT/'assets'/(digest+path.suffix)
        if not dst.exists():
            tmp=dst.with_suffix(dst.suffix+'.tmp');shutil.copyfile(path,tmp);tmp.replace(dst)
        assert sha(dst)==digest
        assets[digest]={'original_path':str(path),'frozen_path':str(dst),'sha256':digest}
        r['original_absolute_path']=str(path);r['absolute_path']=str(dst)
        return r
    for original in originals:
        if original['dataset']!='ue':
            probe=original['source'].get('historical_technical_probe',{})
            exceptions.append({'id':original['id'],'uid':original['uid'],'reason':'original_shorter_than_10s','source_duration_seconds':probe.get('duration'),'status':'deferred','basis':'historical_full_decode_probe','replacement':False,'padding':False})
            continue
        item=copy.deepcopy(original);item.update(num_frames=121,timestamps_seconds=[i/12 for i in range(121)])
        sample=samples[int(item['uid'].split(':')[1])];text=m['text'][sample['text']]
        assert text['prompt_en']==item['prompt']['en']
        entry={'id':item['id'],'text':{'receipt':text,'path':str(Path(src['encoding_root'])/text['artifact']['path'])},'references':[]}
        def cache(record,key,motion):
            r=m['media'][key];assert r['status']=='passed'
            assert r['media_sha256_verified']==record['sha256'] and r['media_path']==record['path']
            if motion:
                assert r['num_frames']==121 and r['canvas_wh']==item['canvas_wh']
                assert r['pixel_metadata']['timestamps_seconds']==item['timestamps_seconds']
                assert r['pixel_metadata']['original']['frames']/float(r['pixel_metadata']['original']['fps'].split('/')[0])*float(r['pixel_metadata']['original']['fps'].split('/')[1])>=10
                assert r['tensor']['shape'][2]==16
            else: assert r['num_frames']==1 and r['tensor']['shape'][2]==1
            return {'receipt':r,'path':str(Path(src['encoding_root'])/r['artifact']['path'])}
        entry['source']=cache(item['source'],sample['source'],True)
        item['source']=freeze(item['source'])
        for i,(given,cached) in enumerate(zip(item['references'],sample['references'],strict=True)):
            assert (cached['index'],cached['type'],cached.get('view',''))==(i,given['type'],given.get('view',''))
            entry['references'].append(cache(given,cached['media_key'],given['type']=='motion')|{'index':i,'type':given['type'],'view':given.get('view','')})
            item['references'][i]=freeze(given)
        # Freeze GT separately; never place it in the inference input or cache index.
        target=next(r['record']['target'] for r in rows(old/'metadata/selection.jsonl') if r['id']==item['id'])
        gt=freeze(target)
        assets[gt['sha256']]['role_note']='review_only_gt'
        index[item['id']]=entry;selected.append(item)
    jsonl(ROOT/'metadata/inputs.jsonl',selected)
    for entry in index.values():entry['input_sha256']=sha(ROOT/'metadata/inputs.jsonl')
    atomic(ROOT/'metadata/native_cache_index.json',index)
    atomic(ROOT/'metadata/short_clip_exceptions.json',exceptions)
    atomic(ROOT/'metadata/frozen_assets.json',assets)
    from native_cache import load_sample
    for item in selected:load_sample(item)
    atomic(ROOT/'metadata/preflight.json',{'status':'passed','samples':len(selected),'deferred':len(exceptions),'source_frames':121,'fps':12,'source_latent_frames':16,'no_gt_inference':True,'manifest_sha256':sha(ROOT/'metadata/inputs.jsonl'),'completed_unix':time.time()})
    print('Preflight passed',len(selected),'deferred',len(exceptions),flush=True)
if __name__=='__main__':main()
