"""CPU final acceptance: exact reuse, full media decode, native H3 pixels, all metrics/reviews."""
import concurrent.futures,re,time
from collections import Counter,defaultdict
from urllib.parse import urlparse,unquote
from common import *
from contracts import digest,DIMS,finite_score
from score import MODELS,inventory

def decode(spec):
    import av,hashlib
    sid,model,path,expected,canvas=spec
    with av.open(str(path))as c:
        s=c.streams.video[0];assert s.average_rate==12,(sid,model,s.average_rate)
        count=0
        for f in c.decode(video=0):
            assert [f.width,f.height]==canvas,(sid,model,f.width,f.height,canvas);count+=1
    assert count==expected,(sid,model,count)
    return dict(id=sid,model=model,frames=count,path=str(path))

def h3_pixels(item):
    import av,hashlib
    sid=item['id'];native=ROOT/'outputs/h3_pdd8_native'/f'{sid}.mp4';scored=ROOT/'outputs/h3_pdd8'/f'{sid}.mp4';selected=[];indices=set(range(0,97,2))
    with av.open(str(native))as c:
        assert c.streams.video[0].average_rate==24;count=0
        for n,f in enumerate(c.decode(video=0)):
            if n in indices:selected.append(hashlib.sha256(f.to_ndarray(format='rgb24').tobytes()).hexdigest())
            count+=1
    assert count==107,(sid,count)
    with av.open(str(scored))as c:actual=[hashlib.sha256(f.to_ndarray(format='rgb24').tobytes()).hexdigest()for f in c.decode(video=0)]
    assert actual==selected and len(actual)==49,sid
    return dict(id=sid,native_frames=107,comparison_frames=49,generated_indices=list(range(0,97,2)),actual_generated_pixels_preserved=True)

def main():
    items=rows(ROOT/'metadata/inputs.jsonl');manifest=sha(ROOT/'metadata/inputs.jsonl');freeze=read(ROOT/'metadata/freeze.json')
    assert freeze['input_manifest_sha256']==manifest and len(items)==116
    assert [r['id']for r in items]==sorted(r['id']for r in items)
    assert Counter(r['group']for r in items)==dict(hair=24,hair_color=24,background=24,tripvvt=24,native=20)
    assert Counter(r['cohort']for r in items)==dict(retained92=92,new24=24)
    assert all(r['group']!='identity'and not r['id'].startswith('object_removal_')for r in items)
    prefixes=defaultdict(list)
    for r in items:
        match=re.fullmatch(r'(.+)_(\d+)',r['id']);assert match;prefixes[match[1]].append(int(match[2]))
    assert all(numbers==list(range(1,len(numbers)+1))for numbers in prefixes.values())
    parent=ROOT.parent/'a849_external_vbench_v3_116'
    assert sha(parent/'metadata/inputs.jsonl')=='f93cf8c4ae008f5ad57e06a2f6def0bd75ce46f9131c6165fd7bb35165478c6a'
    overlap=read(ROOT/'metadata/training_overlap.json');assert overlap['status']=='complete'and not overlap['source_hits']and not overlap['missing']
    reuse=read(ROOT/'metadata/reuse_outputs.json');assert reuse['status']=='passed'and reuse['complete']==464
    proofs={(r['id'],r['model']):r for r in reuse['proofs']};assert len(proofs)==464
    gate=read(ROOT/'metadata/generation_gate.json');assert gate['status']=='approved'and gate['manifest_sha256']==manifest
    assets={r['id']:r for r in gate['assets']};observations=read(ROOT/'metadata/visual_observations.json')
    assert observations['input_manifest_sha256']==manifest
    videos=inventory();assert len(videos)==464 and all(r['generation_status']=='complete'for r in videos)
    metrics=rows_csv=__import__('csv').DictReader((ROOT/'metadata/per_video_metrics.csv').open());metrics=list(metrics)
    assert len(metrics)==2784 and all(r['status']=='complete'and r['score']!=''and finite_score(float(r['score']))for r in metrics)
    assert len({(r['id'],r['model'],r['dimension'])for r in metrics})==2784
    specs=[]
    for item in items:
        sid=item['id'];assert digest(item)==assets[sid]['input_digest']
        assert sha(ROOT/'inputs/prepared'/sid/'pixels.json')==assets[sid]['source_pixels_receipt_sha256']
        assert sha(ROOT/'conditions'/sid/'condition.json')==assets[sid]['condition_receipt_sha256']
        assert sha(item['source']['absolute_path'])==item['source']['sha256']
        body=read(ROOT/'metadata/h3_requests'/f'{sid}.json');assert body['num_inference_steps']==8 and body['target']['duration_seconds']==4 and item['prompt']['en']in body['prompt']
        assert len(body['conditions'])==1+len(item['references'])and body['conditions'][0]['type']=='video'
        source=Path(unquote(urlparse(body['conditions'][0]['uri']).path));audit=read(source.with_suffix('.json'))
        assert audit['identity']['source_sha256']==item['source']['sha256']and sha(source)==audit['sha256']
        for ref,condition in zip(item['references'],body['conditions'][1:],strict=True):
            assert sha(ref['absolute_path'])==ref['sha256']and Path(unquote(urlparse(condition['uri']).path)).resolve()==Path(ref['absolute_path']).resolve()
        for model in MODELS:
            r=read(ROOT/'outputs'/model/f'{sid}.json');h=r['output_sha256'];proof=proofs[sid,model]
            assert proof['status']=='passed'and proof['new_manifest_sha256']==manifest
            assert proof.get('comparison_sha256',proof['output_sha256'])==h
            note=observations['samples'][sid]['models'][model];assert note['reviewed']is True and note['output_sha256']==h
            canvas=[1344,768]if item['canvas_wh'][0]>item['canvas_wh'][1]else[768,1344]
            specs.append((sid,model,ROOT/'outputs'/model/f'{sid}.mp4',49,canvas if model=='h3_pdd8'else item['canvas_wh']))
            if model=='h3_pdd8':specs.append((sid,'h3_browser',Path(r['browser_path']),49,canvas))
        for model in ['source',*MODELS]:
            path=ROOT/'inputs/prepared'/sid/'source.mp4'if model=='source'else ROOT/'outputs'/model/f'{sid}.mp4';h=sha(path)
            static=read(ROOT/'metadata/vbench/static'/f'{h}.json');assert static['status']=='complete'and static['media_sha256']==h and type(static['is_static'])is bool
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:decoded=list(pool.map(decode,specs));pixels=list(pool.map(h3_pixels,items))
    paired=read(ROOT/'metadata/temporal_flickering_static_paired.json')['all116'];assert paired['filter_complete']and paired['n']==len(paired['task_ids'])
    assert all(not sid.startswith(('identity_','object_removal_'))for sid in paired['task_ids'])
    atomic(ROOT/'metadata/final_validation.json',dict(status='passed',manifest_sha256=manifest,tasks=116,outputs=464,scores=2784,direct_reviews=464,scoring_browser_decodes=len(decoded),native_pixel_checks=len(pixels),decode_checks=decoded,h3_pixel_checks=pixels,static_common_n=paired['n'],static_common_ids=paired['task_ids'],parent_unchanged=True,output_conditioned_TripVVT_disclosed=True,completed_unix=time.time()))
    print('Final116 passed:464 outputs/2784 scores/464 reviews/580 full decodes/116 native pixel checks',flush=True)

if __name__=='__main__':main()
