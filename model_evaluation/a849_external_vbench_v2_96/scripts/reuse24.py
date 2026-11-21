"""Retain proven identical task outputs and source-only caches without editing contracts."""
import shutil
from common import *

OLD=ROOT.parent/'a849_external_vbench_v1'

def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():os.link(src,dst)

def main():
    require_job()
    prior=ROOT/'metadata/prior24';prior.mkdir(exist_ok=True)
    old_items={r['id']:r for r in rows(OLD/'metadata/inputs.jsonl')}
    items=rows(ROOT/'metadata/inputs.jsonl');digest=sha(ROOT/'metadata/inputs.jsonl');old_digest=sha(OLD/'metadata/inputs.jsonl')
    for name in ['inputs.jsonl','freeze.json','reuse_outputs.json','visual_observations.json','reference_processing.json']:
        shutil.copy2(OLD/'metadata'/name,prior/name)
    old_reuse=read(OLD/'metadata/reuse_outputs.json');reused=[];cache=[]
    for item in items:
        sid=item['id']
        if sid not in old_items:continue
        old=old_items[sid]
        for model in ['ltx_base','a849','vace']:
            if item!=old:continue
            record_path=OLD/'outputs'/model/f'{sid}.json';record=read(record_path)
            assert record['status']=='complete'
            video=record_path.with_suffix('.mp4');assert sha(video)==record['output_sha256']
            actual_manifest=record['contract']['input_manifest_sha256']
            if actual_manifest!=old_digest:
                proof=next(r for r in old_reuse['reused'] if r['id']==sid and r['model']==model)
                assert proof['old_manifest_sha256']==actual_manifest and proof['task_record_bitwise_equal'] and proof['output_sha256']==record['output_sha256']
            for src in [record_path,video]:link(src,ROOT/'outputs'/model/src.name)
            reused.append(dict(id=sid,model=model,source_manifest_sha256=old_digest,old_manifest_sha256=actual_manifest,new_manifest_sha256=digest,task_record_bitwise_equal=True,original_generation_record=str(record_path),output_sha256=record['output_sha256'],prior_reuse_proof=str(prior/'reuse_outputs.json') if actual_manifest!=old_digest else None))
        src=OLD/'inputs/prepared'/sid;dst=ROOT/'inputs/prepared'/sid;dst.mkdir(parents=True,exist_ok=True)
        for path in src.iterdir():
            if path.suffix!='.json':link(path,dst/path.name)
        pixel=read(src/'pixels.json');assert pixel['source_sha256']==item['source']['sha256']
        assert old['source']==item['source'] and old['canvas_wh']==item['canvas_wh'] and old['timestamps_seconds']==item['timestamps_seconds']
        pixel.update(input_manifest_sha256=digest,manifest_rebind={'old':old_digest,'new':digest,'source_and_time_grid_unchanged':True});atomic(dst/'pixels.json',pixel)
        for path in (OLD/'cache'/sid).glob('*.pt'):
            role=path.stem
            same=(old['prompt']==item['prompt']) if role=='text' else (old['source']==item['source']) if role=='source' else old['references'][int(role.split('_')[1])]==item['references'][int(role.split('_')[1])]
            if not same:continue
            receipt=read(path.with_suffix('.json'));assert sha(path)==receipt['sha256']
            out=ROOT/'cache'/sid/path.name;link(path,out)
            receipt['identity']['manifest_sha256']=digest
            receipt['manifest_rebind']=dict(old_manifest_sha256=old_digest,new_manifest_sha256=digest,role_input_equal=True,artifact_sha256=receipt['sha256'],original_receipt=str(path.with_suffix('.json')))
            atomic(out.with_suffix('.json'),receipt);cache.append(dict(id=sid,role=role,sha256=receipt['sha256']))
        old_cond=OLD/'conditions'/sid/'condition.json';cond=read(old_cond)
        assert all(old[k]==item[k] for k in ['source','mask_spec','prompt','vace_description'])
        folder=ROOT/'conditions'/sid
        for path in old_cond.parent.iterdir():
            if path.suffix!='.json':link(path,folder/path.name)
        cond.update(input_manifest_sha256=digest,source_pixels_sha256=sha(dst/'pixels.json'),manifest_rebind={'old':old_digest,'new':digest,'source_mask_and_prompt_unchanged':True,'original_condition':str(old_cond)})
        atomic(folder/'condition.json',cond)
    for name in ['task_limitations.json','vace_prompt_corrections.json']:
        if (OLD/'metadata'/name).exists():shutil.copy2(OLD/'metadata'/name,ROOT/'metadata'/name)
    atomic(ROOT/'metadata/reuse_outputs.json',dict(new_manifest_sha256=digest,source_manifest_sha256=old_digest,reused=reused,cache_rebound=cache,note='Only16 complete input-identical tasks reused, original generation contracts unchanged. Identity/hair reference edits force regeneration of all32 group tasks.'))
    old_visual=read(prior/'visual_observations.json');samples={}
    for item in items:
        sid=item['id'];models={}
        for rec in reused:
            if rec['id']!=sid:continue
            observed=old_visual['samples'].get(sid,{}).get('models',{}).get(rec['model'])
            if observed and observed['output_sha256']==rec['output_sha256']:models[rec['model']]=observed
        samples[sid]=dict(evidence='Prior24 direct review for input-identical reused outputs only; new outputs require direct inspection.',models=models)
    atomic(ROOT/'metadata/visual_observations.json',dict(input_manifest_sha256=digest,reviewer='Codex direct inspection; no VLM API',inspection_complete=False,summary='96-task extension in progress; only prior direct observations tied to unchanged output SHA retained.',trend='Pending all output review and scoring.',samples=samples))
    print('Reused',len(reused),'outputs;',288-len(reused),'pending;',len(cache),'cache artifacts retained',flush=True)

if __name__=='__main__':main()
