"""Final artifact verification; keep failures explicit rather than scoring zero."""
import csv
import zipfile
from collections import Counter
from common import *

if __name__=='__main__':
    require_job()
    frozen=read(ROOT/'metadata/freeze.json');inputs=rows(ROOT/'metadata/inputs.jsonl')
    assert sha(ROOT/'metadata/inputs.jsonl')==frozen['input_manifest_sha256']
    assert len(inputs)==24 and Counter(r['group'] for r in inputs)==frozen['groups']
    assert all('target' not in r and 'caption' not in r for r in inputs)
    if frozen.get('revision')=='whitehair_fullframe_v2':
        import numpy as np
        from PIL import Image
        for ref in read(ROOT/'metadata/reference_processing.json'):
            if ref.get('role')=='background':continue
            original=np.asarray(Image.open(ref['path']).convert('RGB'))
            actual=np.asarray(Image.open(ref['identity_reference_path']).convert('RGB'))
            mask=np.asarray(Image.open(ref['identity_hair_mask_path']))>127
            assert original.shape==actual.shape and mask.shape==actual.shape[:2]
            assert np.array_equal(actual[~mask],original[~mask]) and np.all(actual[mask]==255)
            assert sha(ref['identity_reference_path'])==ref['identity_reference_sha256']
    paired=[];failures=[]
    for item in inputs:
        sid=item['id'];records={}
        for model in ['ltx_base','a849','vace']:
            path=ROOT/'outputs'/model/f'{sid}.json'
            r=read(path) if path.is_file() else {'status':'missing'};records[model]=r
            if r['status']!='complete':failures.append(dict(id=sid,model=model,status=r['status']));continue
            assert sha(ROOT/'outputs'/model/f'{sid}.mp4')==r['output_sha256']
            if r['contract']['input_manifest_sha256']!=frozen['input_manifest_sha256']:
                reuse=read(ROOT/'metadata/reuse_outputs.json')
                entry=next(e for e in reuse['reused'] if e['id']==sid and e['model']==model)
                old=next(i for i in rows(ROOT/'metadata/before_whitehair/inputs.jsonl') if i['id']==sid)
                assert old==item and entry['task_record_bitwise_equal']
                assert r['output_sha256']==entry['output_sha256']
                assert r['contract']['input_manifest_sha256']==entry['old_manifest_sha256']
            assert r['contract']['steps']==40 and r['contract']['seed']==42
            assert r['full_decode_frames']==49 and r['protocol_audit']['no_gt']
            if model=='vace':
                assert r['official_repaint_erasure']['masked_rgb']==128
                assert r['protocol_audit']['white_is_editable']
                assert r['protocol_audit']['reference_count']==len(item['references'])
            else:
                assert r['protocol_audit']['target_last']
                assert r['protocol_audit']['optional_reference_count']==len(item['references'])
                assert r['contract']['reference_protocol']=='reference_positions_v7_refs_source_target'
        if all(records[m]['status']=='complete' for m in ['ltx_base','a849']):
            base=records['ltx_base']['contract'];adapted=records['a849']['contract']
            changed=[k for k in set(base)|set(adapted) if base.get(k)!=adapted.get(k)]
            assert set(changed)=={'checkpoint','checkpoint_sha256'},(sid,changed)
            assert base['checkpoint'] is None and adapted['checkpoint']==str(CHECKPOINT)
            paired.append(dict(id=sid,changed_contract_fields=changed))
    metrics=list(csv.DictReader((ROOT/'metadata/per_video_metrics.csv').open()))
    assert len(metrics)==432
    assert all(r['score']=='' for r in metrics if r['status']!='complete')
    group=read(ROOT/'metadata/group_metrics.json')
    assert sum(r['status']=='complete' for r in metrics)==group['valid_scores']
    visual=read(ROOT/'metadata/visual_observations.json')
    assert visual['inspection_complete'] is True
    assert set(visual['samples'])=={r['id'] for r in inputs}
    for sid,sample in visual['samples'].items():
        assert set(sample['models'])=={'ltx_base','a849','vace'}
        assert all(m['reviewed'] is True for m in sample['models'].values())
        for model,observation in sample['models'].items():
            assert observation['output_sha256']==read(ROOT/'outputs'/model/f'{sid}.json').get('output_sha256')
    package=read(ROOT/'metadata/package.json')
    with zipfile.ZipFile(package['path']) as z:
        assert z.testzip() is None
        assert package['offline_page'] in z.namelist()
    validation=read(ROOT/'metadata/all_media_validation.json')
    assert validation['complete_generations']==72-len(failures)
    atomic(ROOT/'metadata/final_validation.json',dict(status='passed' if not failures and group['valid_scores']==432 else 'partial',generation_failures=failures,valid_scores=group['valid_scores'],expected_scores=432,paired_contracts=paired,package_crc='passed',input_manifest_sha256=frozen['input_manifest_sha256'],visual_observations='all24 present; direct inspection evidence in observations',scope='Technical integrity, paired inference settings, missingness and package links/CRC. This is not a claim that all edits succeeded.'))
    print('verified',len(paired),'pairs',group['valid_scores'],'scores',flush=True)
