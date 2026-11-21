"""Freeze eight teacher-template hair colors and eight reference-background tasks."""
import copy,time
from collections import Counter
from common import *

def main():
    parent=ROOT.parent/'a849_external_vbench_v3_116';pilot=ROOT.parent/'a849_tripvvt_pilot32_20261115'
    previous=rows(parent/'metadata/inputs.jsonl');used={r['source']['sha256']for r in previous}
    candidates={r['id']:r for r in read(pilot/'metadata/omni_unused.json')}
    hc=[('Sp_Appearance_change_03','black','silver white',4),('Sp_Appearance_change_04','blue','silver white',4),
        ('Sp_Appearance_change_08','chestnut brown','silver white',4),('Te_Facial_expression_change_04','blonde','red',10),
        ('Te_Facial_expression_change_05','pink','black',10),('Te_Facial_expression_change_10','white','brown',11),
        ('Te_Action_category_change_06','red','black',7),('Te_Facial_expression_change_02','chestnut brown','blonde',10)]
    bg=[('Sp_Object_style_change_05',1),('Sp_Clothes_change_07',5),('Sp_Appearance_change_10',5),
        ('Sp_Environment_change_05',6),('Te_Facial_expression_change_07',10),('Te_Camera_viewpoint_transformation_02',11),
        ('Sp_Relighting_03',5),('Te_Action_content_change_02',8)]
    templates=read(parent/'metadata/hair_color_templates.json')
    choices={r['prompt']['en'].split('hair color to ')[1].rstrip('.'):r for r in templates['templates'].values()}
    items=[];source_hashes=[]
    backgrounds=[];seen=set()
    for r in previous:
        if r['group']=='background'and r['references'][0]['sha256']not in seen:
            seen.add(r['references'][0]['sha256']);backgrounds.append(r)
    assert len(backgrounds)==4
    for group,records in [('hair_color',hc),('background',bg)]:
        for i,record in enumerate(records,17):
            uid=record[0];candidate=candidates[uid];source=Path(candidate['path'])
            assert candidate['probe']['duration_last_pts']>=4
            assert sha(source)==candidate['source_sha256'] and candidate['source_sha256']not in used
            source_hashes.append(candidate['source_sha256'])
            source_review=dict(reviewer='Codex direct visual inspection; no VLM API',
                contact_sheet=f'omni_contact_{record[-1]:02}.jpg',timestamps_seconds=[0,1,2,3,4],
                notes='Single main person; target remains visible over the complete edit interval.')
            if group=='hair_color':
                _,color,observed,_=record;t=choices[color];item=copy.deepcopy(next(r for r in previous if r['group']=='hair_color'))
                item.update(subgroup=color,references=[],prompt=copy.deepcopy(t['prompt']),template_provenance=copy.deepcopy(t),
                    source_hair_color_observed=observed,axis='hair_color',mode='attribute',task_origin='self_created_training_template')
                item['vace_description']=f"Change only the person's hair color to {color}. Preserve the original hairstyle, facial identity, expression, clothing, body motion, other people, environment and camera."
                item['mask_spec']=dict(parsing_labels=[2],invert=False,dilation_pixels=4,main_person=True)
                source_review['notes']+=f' Hair is visible and originally {observed}; target {color} contrasts with source.'
            else:
                item=copy.deepcopy(backgrounds[(i-17)//2]);item['task_origin']='self_created_reference_background_edit; no GT'
                source_review['notes']+=' Preserve the foreground person; new reference scene differs from source.'
            w,h=candidate['probe']['dimensions']
            item.update(id=f'{group}_{i:02}',uid=uid,source=dict(absolute_path=str(source),sha256=candidate['source_sha256']),
                canvas_wh=[1280,736]if w>h else[736,1280],cohort='extension16',review_group=group,
                legacy_task_id=None,old_id=None,source_task_id=None,source_old_id=None,
                source_review=source_review,edit_target='the single main person',original_omni_instruction=candidate['instruction'],
                revision_reason='User requested expansion to24 with a new independent source video',
                provenance=dict(dataset='OmniEdit',revision='f6662c5371c21392b80939710ed61adec68531be',official_record=candidate['official']))
            items.append(item)
    assert len(items)==len(set(source_hashes))==16
    assert Counter(r['subgroup']for r in items if r['group']=='hair_color')==dict(black=1,blue=1,blonde=1,**{'chestnut brown':2},pink=1,red=1,white=1)
    # Repeat the exact-file and available-lineage manifest checks for every new source/reference.
    inventory=read(parent/'metadata/training_overlap.json');training=set()
    import sys
    sys.path.insert(0,str(pilot/'scripts'));from select_pool import walk
    for m in inventory['manifest_inventory']:
        assert sha(m['path'])==m['sha256']
        for r in rows(m['path']):
            for key,value in walk(r):
                if key in ['sha256','media_sha256','media_sha256_verified']:training.add(value)
    for r in items:
        assert r['source']['sha256']not in training
        assert all(ref['sha256']not in training for ref in r['references'])
    items.sort(key=lambda r:r['id']);jsonl(ROOT/'metadata/inputs.jsonl',items)
    atomic(ROOT/'metadata/training_overlap.json',dict(status='passed',hits=[],source_hits=[],missing=[],manifest_inventory=inventory['manifest_inventory'],scope=inventory['scope']))
    atomic(ROOT/'metadata/freeze.json',dict(revision='v4_extensions16',input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),models=['ltx_base','a849','vace','h3_pdd8'],smoke=[items[0]['id']],created_unix=time.time(),no_vlm_api=True))
    atomic(ROOT/'metadata/generation_targets.json',{m:[r['id']for r in items]for m in ['ltx_base','a849','vace']})
    print('Frozen16 extensions',sha(ROOT/'metadata/inputs.jsonl'),flush=True)

if __name__=='__main__':main()
