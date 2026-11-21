"""Assemble116 from92 unchanged v3 tasks,16 frozen extensions and8 selected pilots."""
import copy,shutil,time
from collections import Counter
from common import *
from contracts import digest,generation_input
from reuse import link,transplant

MODELS=('ltx_base','a849','vace','h3_pdd8')
PILOT=ROOT.parent/'a849_tripvvt_pilot32_20261115'
EXT=ROOT.parent/'a849_v4_extensions16_20261115'
PARENT=ROOT.parent/'a849_external_vbench_v3_116'

def main():
    decision=read(PILOT/'metadata/selection_decision.json')
    assert decision['status']=='complete'and len(decision['selected'])==8,'Do not lower quality gate or invent extra candidates'
    tasks=rows(ROOT/'metadata/retained_inputs.jsonl');origins=read(ROOT/'metadata/retained_origins.json')
    for r in rows(EXT/'metadata/inputs.jsonl'):
        item=copy.deepcopy(r);item['cohort']='new24';item['source_task_id']=r['id'];tasks.append(item);origins[item['id']]=[str(EXT),r['id']]
    pilot={r['id']:r for r in rows(PILOT/'metadata/inputs.jsonl')}
    for n,sid in enumerate(sorted(decision['selected']),17):
        item=copy.deepcopy(pilot[sid]);item.update(id=f'tripvvt_{n:02}',cohort='new24',pilot_id=sid,source_task_id=sid,
            revision_reason='Output-conditioned choice: A849 successful correct-person edit and good preservation; H3 may fail')
        tasks.append(item);origins[item['id']]=[str(PILOT),sid]
    tasks.sort(key=lambda r:r['id'])
    pool=ROOT.parents[1]/'dataset_research/coinve_tripvvt_11k_20261108/manifest.jsonl'
    trip_records={r['metadata_index']:r for r in rows(pool)if r['dataset']=='tripvvt'}
    for item in tasks:
        if item['group']!='tripvvt':continue
        record=trip_records[int(item['uid'].split(':')[-1])];assert item['source']['sha256']==record['source_sha256']
        item['provenance']=dict(source_manifest=str(pool),project_split=record['split'],official_split=record['original_metadata']['official_split'],group_id=record['group_id'],original_sample_id=record['sample_id'],repo_id=record['repo_id'],revision=record['revision'])
    assert len(tasks)==116 and len({r['id']for r in tasks})==116
    assert Counter(r['group']for r in tasks)==dict(hair=24,hair_color=24,background=24,tripvvt=24,native=20)
    assert Counter(r['cohort']for r in tasks)==dict(retained92=92,new24=24)
    assert all(r['group']!='identity'and not r['id'].startswith('object_removal_')for r in tasks)
    assert set(Counter(r['references'][0]['sha256']for r in tasks if r['group']=='background').values())=={6}
    assert Counter(r['subgroup']for r in tasks if r['group']=='hair_color')==dict(black=3,blue=4,pink=4,**{'chestnut brown':3},red=4,blonde=3,white=3)
    byroot={root:{r['id']:r for r in rows(Path(root)/'metadata/inputs.jsonl')}for root,_ in origins.values()}
    gates={root:read(Path(root)/'metadata/generation_gate.json')for root in byroot}
    for root,gate in gates.items():
        assert gate['status']=='approved'and gate['manifest_sha256']==sha(Path(root)/'metadata/inputs.jsonl')
    gate_assets={root:{a['id']:a for a in gate['assets']}for root,gate in gates.items()}
    # All generation must already be complete in its own immutable, reviewed input root.
    for item in tasks:
        root,sid=origins[item['id']]
        for model in [*MODELS,'h3_pdd8_native']:
            p=Path(root)/'outputs'/model/f'{sid}.json';assert p.exists()and read(p)['status']=='complete',(root,sid,model)
    jsonl(ROOT/'metadata/inputs.jsonl',tasks);manifest=sha(ROOT/'metadata/inputs.jsonl');atomic(ROOT/'metadata/input_origins.json',origins)
    atomic(ROOT/'metadata/renumbering.json',[dict(id=r['id'],source_root=origins[r['id']][0],source_id=origins[r['id']][1],legacy_task_id=r.get('legacy_task_id'),original_old_id=r.get('old_id'),pilot_id=r.get('pilot_id'),uid=r['uid'],reason=r['revision_reason'])for r in tasks])
    atomic(ROOT/'metadata/freeze.json',dict(revision='v4_116_output_conditioned_TripVVT',input_manifest_sha256=manifest,groups=dict(Counter(r['group']for r in tasks)),cohorts=dict(Counter(r['cohort']for r in tasks)),models=list(MODELS),smoke=[],created_unix=time.time(),selection_decision_sha256=sha(PILOT/'metadata/selection_decision.json'),parent_manifest_sha256=sha(PARENT/'metadata/inputs.jsonl')))
    shutil.copy2(PILOT/'metadata/selection_decision.json',ROOT/'metadata/tripvvt_selection_decision.json')
    shutil.copy2(PILOT/'metadata/selection_report.md',ROOT/'metadata/selection_report.md')
    shutil.copy2(PILOT/'metadata/pair_reviews.json',ROOT/'metadata/pair_reviews.json')
    shutil.copy2(EXT/'metadata/input_review.json',ROOT/'metadata/new16_input_review.json')
    shutil.copy2(EXT/'metadata/mask_refinement_resolution.json',ROOT/'metadata/new16_mask_refinement.json')
    shutil.copy2(EXT/'metadata/task_limitations.json',ROOT/'metadata/task_limitations.json')
    runtime=read(PARENT/'metadata/h3_runtime_contract.json');atomic(ROOT/'metadata/h3_runtime_contract.json',runtime)
    runtime_sha=sha(ROOT/'metadata/h3_runtime_contract.json');checkpoint_sha=sha(CHECKPOINT)
    proofs=[];requests=[];selections=[];input_reviews={}
    for item in tasks:
        root,sid=origins[item['id']];root=Path(root);prior=byroot[str(root)][sid]
        approved=gate_assets[str(root)][sid]
        assert approved['input_digest']==digest(prior)
        assert approved['source_pixels_receipt_sha256']==sha(root/'inputs/prepared'/sid/'pixels.json')
        assert approved['condition_receipt_sha256']==sha(root/'conditions'/sid/'condition.json')
        transplant(root,sid,item,roles=['source','text',*[f'ref_{n}'for n in range(len(item['references']))]])
        old_cond=read(root/'conditions'/sid/'condition.json');new_cond=read(ROOT/'conditions'/item['id']/'condition.json')
        assert digest(read(root/'metadata/h3_runtime_contract.json'))==digest(runtime)
        body=read(root/'metadata/h3_requests'/f'{sid}.json');atomic(ROOT/'metadata/h3_requests'/f"{item['id']}.json",body)
        for model in ['ltx_base','a849','vace']:
            p=root/'outputs'/model/f'{sid}.mp4';r=read(p.with_suffix('.json'));c=r['contract']
            effective=generation_input(item,model,new_cond if model=='vace'else None)
            assert effective==generation_input(prior,model,old_cond if model=='vace'else None)
            assert sha(p)==r['output_sha256']and c['seed']==42
            if model!='vace':assert c['steps']==40
            assert c['canvas_wh']==item['canvas_wh']and c['fps']==12 and c['num_frames']==49
            if model=='vace':
                profile=read(root/'metadata/vace_inference_plan.json')['profiles'][sid] if (root/'metadata/vace_inference_plan.json').exists() else 'unipc40'
                expected=dict(steps=40,cfg=5,flow_shift=5.,conditioning_scale=1.,scheduler='UniPCMultistepScheduler',transformer_dtype='bfloat16',vae_dtype='float32')
                if profile=='lightx2v4':
                    expected.update(steps=4,cfg=1.,scheduler='FlowMatchEulerDiscreteScheduler')
                    accel=read(root/'metadata/vace_runtime_contract.json')
                    assert c['acceleration_contract_sha256']==sha(root/'metadata/vace_runtime_contract.json')
                    assert c['condition_sha256']==sha(root/'conditions'/sid/'condition.json')
                    assert c['lora_sha256']==accel['lora_sha256'] and c['adapter_sha256']==accel['adapter_sha256'] and c['lora_strength']==1.
                    assert c['negative_prompt'] is None and len(c['sigmas'])==5 and all(abs(a-b)<1e-6 for a,b in zip(c['sigmas'],accel['sigmas']))
                    assert r['lora_load']['original_tensors']==1457 and r['lora_load']['official_lora_tensors']==1215 and len(r['lora_load']['additional_vectors'])==242 and r['lora_load']['all_loaded_values_checked']
                else:assert profile=='unipc40'
                assert all(c[k]==v for k,v in expected.items())
                assert r['original_instruction']==item['prompt']['en']and r['adapted_prompt']==new_cond['adapted_prompt']and r['official_repaint_erasure']['masked_rgb']==128
            else:
                assert all(c[k]==v for k,v in dict(cfg=3,stg=1,stg_blocks=[28],rescale=.7,ref_cfg=1.,reference_protocol='reference_positions_v7_refs_source_target').items())
                assert c['config_sha256']==sha(CONFIG)and c['native_weights_record_sha256']==sha(NATIVE/'runtime/models/native_model_revision.json')
                assert c['checkpoint']is None if model=='ltx_base'else c['checkpoint_sha256']==checkpoint_sha
                assert r['protocol_audit']['no_gt']and r['protocol_audit']['optional_reference_count']==len(item['references'])
            for suffix in ['.mp4','.json']:link(p.with_suffix(suffix),ROOT/'outputs'/model/f"{item['id']}{suffix}")
            proofs.append(dict(id=item['id'],model=model,status='passed',output_sha256=r['output_sha256'],effective_input_sha256=digest(effective),original_root=str(root),original_id=sid,original_receipt_sha256=sha(p.with_suffix('.json')),original_manifest_sha256=sha(root/'metadata/inputs.jsonl'),new_manifest_sha256=manifest,inference_protocol_verified=True))
        native=root/'outputs/h3_pdd8_native'/f'{sid}.mp4';nr=read(native.with_suffix('.json'));c=nr['contract']
        assert sha(native)==nr['output_sha256']and c['request_sha256']==digest(body)and c['runtime_contract_sha256']==runtime_sha
        assert all(c[k]==v for k,v in dict(steps=8,seed=42,tp_size=2,ulysses_degree=2,native_frames=107,native_fps=24).items())
        for model in ['h3_pdd8_native','h3_pdd8']:
            p=root/'outputs'/model/f'{sid}.mp4';r=read(p.with_suffix('.json'));assert sha(p)==r['output_sha256']
            for suffix in ['.mp4','.json']:link(p.with_suffix(suffix),ROOT/'outputs'/model/f"{item['id']}{suffix}")
        compare=read(root/'outputs/h3_pdd8'/f'{sid}.json');assert compare['native_sha256']==nr['output_sha256']
        preview=Path(compare['browser_path']);assert sha(preview)==compare['browser_sha256'];link(preview,ROOT/'review/h3_browser'/f"{item['id']}.mp4")
        proof=dict(id=item['id'],model='h3_pdd8',status='passed',original_id=sid,original_root=str(root),output_sha256=nr['output_sha256'],comparison_sha256=compare['output_sha256'],request_sha256=digest(body),runtime_contract_sha256=runtime_sha,original_receipt_sha256=sha(native.with_suffix('.json')),new_manifest_sha256=manifest,exact_request_uri_preserved=True)
        atomic(ROOT/'metadata/h3_reuse'/f"{item['id']}.json",proof);proofs.append(proof)
        old_preflight=read(root/'metadata/h3_preflight.json');req=copy.deepcopy(next(r for r in old_preflight['requests']if r['id']==sid));req['id']=item['id'];requests.append(req)
        if item['group']=='tripvvt':
            sel=copy.deepcopy(next(r for r in rows(root/'metadata/selection.jsonl')if r['id']==sid));sel['id']=item['id'];selections.append(sel)
        input_reviews[item['id']]=dict(approved=True,source_pixels_receipt_sha256=sha(ROOT/'inputs/prepared'/item['id']/'pixels.json'),condition_receipt_sha256=sha(ROOT/'conditions'/item['id']/'condition.json'),imported_gate=str(root/'metadata/generation_gate.json'),imported_gate_sha256=sha(root/'metadata/generation_gate.json'),imported_id=sid,reason='Same exact source, references, prompt and masks as directly inspected frozen input; receipt bridge above')
    jsonl(ROOT/'metadata/selection.jsonl',selections)
    assert len(proofs)==464
    atomic(ROOT/'metadata/reuse_outputs.json',dict(status='passed',complete=464,retained_v3_outputs=368,new_frozen_stage_outputs=96,proofs=proofs,input_manifest_sha256=manifest))
    atomic(ROOT/'metadata/input_review.json',dict(status='approved',cases=input_reviews,reviewer='SHA-bound import of directly inspected frozen inputs',manifest_sha256=manifest))
    atomic(ROOT/'metadata/h3_preflight.json',dict(status='passed',tasks=116,input_manifest_sha256=manifest,runtime_contract_sha256=runtime_sha,requests=requests,native_frames=107,native_fps=24,comparison_frames=49,comparison_fps=12,no_gt_or_masked_source_in_request=True))
    original_audit=read(PARENT/'metadata/native_provenance_audit.json');original_checks={r['id']:r for r in original_audit['checks']};checks=[]
    for item in tasks:
        if item['group']!='native':continue
        check=copy.deepcopy(original_checks[item['legacy_task_id']]);assert check['source_sha256']==item['source']['sha256']and check['video_id']==item['uid']
        if item['task_origin']=='official':assert check['official_record']['editing_instruction']==item['prompt']['en']
        check.update(id=item['id'],parent_id=item['legacy_task_id']);checks.append(check)
    assert len(checks)==20 and sum(r['task_origin']=='official'for r in checks)==14
    for r in original_audit['local_pinned_benchmark_files']:assert sha(r['path'])==r['sha256']
    audit=copy.deepcopy(original_audit);audit.update(manifest_sha256=manifest,native_tasks=20,official_tasks=14,checks=checks,parent_audit_sha256=sha(PARENT/'metadata/native_provenance_audit.json'),parent_checked_unix=original_audit['checked_unix'],checked_unix=time.time())
    atomic(ROOT/'metadata/native_provenance_audit.json',audit)
    processing=read(PARENT/'metadata/reference_processing.json');used={r['absolute_path']for item in tasks if item['group']=='hair'for r in item['references']}
    atomic(ROOT/'metadata/reference_processing.json',[r for r in processing if r['processed_path']in used])
    print('Assembled116 and verified464 exact-input reuse contracts',manifest,flush=True)

if __name__=='__main__':main()
