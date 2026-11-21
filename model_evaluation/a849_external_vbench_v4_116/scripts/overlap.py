"""Audit all available checkpoint lineage manifests and media hash records."""
from collections import defaultdict
from common import *

def main():
    inputs=rows(ROOT/'metadata/inputs.jsonl')
    wanted={}
    for r in inputs:
        for role,m in [('source',r['source']),*[(f'reference_{i}',m) for i,m in enumerate(r['references'])]]:
            assert sha(m['absolute_path'])==m['sha256'],m['absolute_path']
            wanted.setdefault(m['sha256'],[]).append((r['id'],role,m['absolute_path']))
    for m in read(ROOT/'metadata/reference_processing.json'):
        if m.get('role')!='background':wanted.setdefault(m['sha256'],[]).append(('reference_original',m['name'],m['path']))
    manifests={};lineage=[];missing=[];contract=CHECKPOINT.with_suffix('.contract.json');seen=set()
    while contract and str(contract) not in seen:
        seen.add(str(contract))
        if not contract.is_file():
            run=contract.parent.parent/'run.json'
            if not run.is_file():missing.append(str(contract));break
            contract=run
        c=read(contract);lineage.append(dict(path=str(contract),sha256=sha(contract),evidence_kind='retained_run_record' if contract.name=='run.json' else 'checkpoint_contract',step=c.get('step',c.get('global_step')),parent=c.get('resume_parent_checkpoint',c.get('resume_checkpoint')),initialization=c.get('initialization'),config=c['identity']['config']))
        srcpath=Path(c['identity']['config']['data']['sources_config'])
        if not srcpath.is_file():missing.append(str(srcpath));break
        assert sha(srcpath)==c['identity']['record_sha256']['sources_config'],'Training source configuration changed'
        for s in read(srcpath)['sources']:
            for split in ['train','development']:
                p=s.get(split+'_manifest')
                if p:manifests[p]={'dataset':s['name'],'split':split,'contract_expected_sha256':c['identity'].get('record_sha256',{}).get(s['name']+'/'+split+'_manifest')}
        parent=c.get('resume_parent_checkpoint',c.get('resume_checkpoint'))
        contract=Path(parent).with_suffix('.contract.json') if parent else None
    hits=[];inventories=[];hash_count=0
    def walk(x,loc=''):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in ['sha256','media_sha256','media_sha256_verified'] and isinstance(v,str):yield loc+'.'+k,v
                elif isinstance(v,(dict,list)):yield from walk(v,loc+'.'+k)
        elif isinstance(x,list):
            for i,v in enumerate(x):yield from walk(v,loc+f'[{i}]')
    for p,info in manifests.items():
        path=Path(p)
        if not path.is_file():missing.append(p);continue
        digest=sha(path);expected=info['contract_expected_sha256']
        assert not expected or digest==expected, f'Training manifest changed: {p}'
        records=rows(path);inventories.append(dict(path=p,sha256=digest,count=len(records),**info))
        for r in records:
            for loc,h in walk(r):
                hash_count+=1
                if h in wanted:hits.append(dict(evaluation=wanted[h],training_dataset=info['dataset'],split=info['split'],training_sample_id=r.get('sample_id'),record_location=loc,sha256=h))
    source_hits=[h for h in hits if any(x[1]=='source' for x in h['evaluation'])]
    result=dict(status='complete' if not missing else 'incomplete',checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),lineage=lineage,manifest_inventory=inventories,evaluation_unique_hashes=len(wanted),training_hash_records=hash_count,hits=hits,source_hits=source_hits,missing=missing,scope='Exact file SHA256 plus all SHA fields and sample IDs in available A849 checkpoint lineage train/development manifests. Original and processed references included. This does not detect re-encoded/near-duplicate clips or base-model pretraining overlap; training media hashes are recorded manifests, not a new full media rehash.')
    atomic(ROOT/'metadata/training_overlap.json',result)
    print(json.dumps(dict(status=result['status'],lineage=len(lineage),manifests=len(inventories),hits=len(hits),source_hits=len(source_hits),missing=missing)),flush=True)
    assert not missing,'Incomplete training lineage; inspect before inference'
    assert not source_hits,'Training source/target collision; exclude before generation'

if __name__=='__main__':main()
