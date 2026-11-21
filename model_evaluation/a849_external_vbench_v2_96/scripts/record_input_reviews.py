"""Persist actual direct source/reference inspection; mask approval stays pending."""
from common import *

if __name__=='__main__':
    require_job()
    inputs=rows(ROOT/'metadata/inputs.jsonl');digest=sha(ROOT/'metadata/inputs.jsonl')
    records=[]
    for r in inputs:
        records.append(dict(id=r['id'],source_sha256=r['source']['sha256'],reference_sha256=[q['sha256'] for q in r['references']],source_reviewed=True,references_reviewed=True,reviewer='Codex direct visual inspection; no VLM API',evidence=['expanded_trip_candidates_00–04.jpg','garment_references_00–03.jpg'] if r['group']=='tripvvt' else ['expanded_omni_candidates_00–29.jpg','reference_background_removal.jpg' if r['group'] in ['identity','hair'] else 'v1 directly reviewed empty-scene references, unchanged' if r['group']=='background' else 'official native instructions retained'],mask_review='pending'))
    atomic(ROOT/'metadata/source_reference_visual_review.json',dict(input_manifest_sha256=digest,records=records,scope='All selected sources inspected at8 full-duration timestamps; references inspected directly. No generated96-version outputs used for selection. Mask review is a separate remaining step.'))
    frozen=read(ROOT/'metadata/freeze.json');frozen['references_reviewed']='Identity/hair: reference_background_removal.jpg; all24 garment refs: garment_references_00–03.jpg; background refs unchanged from reviewed v1; all source temporal sheets inspected.';atomic(ROOT/'metadata/freeze.json',frozen)
    atomic(ROOT/'metadata/full_launch_authorization.json',dict(input_manifest_sha256=digest,explicit_user_skip_smoke=True,status='references_verified',mask_review='pending',user_instruction='identity和hair应该把背景去除掉呀，还有就是24条太少了，扩充到96条',prior_skip_smoke_instruction='不用smoke了，我看了看感觉很不错，开始全量吧。人脸部分记住，只是把头发部分变白，不要裁掉。'))
    limitations=read(ROOT/'metadata/task_limitations.json')
    limitations['tripvvt_08']=dict(issue='Original retained instruction says only target person although source has a man left and woman right; garment reference alone does not uniquely identify the target.',fixed_source_only_mask_scope='Pre-inference mask plan assigns right foreground woman; no GT used.',handling='Retain frozen task and outputs; target-specific edit-success comparison is uncertain. Do not change mask from outputs or GT.')
    limitations['source_reuse']=dict(tasks=96,unique_source_hashes=83,handling='Different edits share13 source clips; report source clusters, do not present96 tasks as96 independent source videos.')
    atomic(ROOT/'metadata/task_limitations.json',limitations)
    old=ROOT.parent/'a849_external_vbench_v1';sid='tripvvt_06';r=read(old/'outputs/vace'/f'{sid}.json')
    note=dict(edit='partial',preservation='partial',notes='Open outer layer and inner top replace striped blouse, but both are gray and the inner top is cropped rather than the white reference layers. Source skirt also becomes gray. Walking and hand progression are broadly retained. All49 frames inspected.',output_sha256=r['output_sha256'],reviewed=True,evidence='v1/review/tripvvt_06_vace_all49.jpg')
    old_visual=read(old/'metadata/visual_observations.json');old_visual['samples'][sid]['models']['vace']=note;old_visual['inspection_complete']=True;old_visual['summary']='All72 v1 outputs directly reviewed. Superseded by96-task background-removal revision; historical identity/hair outputs excluded from new reuse.';atomic(old/'metadata/visual_observations.json',old_visual)
    visual=read(ROOT/'metadata/visual_observations.json');visual['samples'][sid]['models']['vace']=note;atomic(ROOT/'metadata/visual_observations.json',visual)
    print('96 source/reference reviews recorded; masks still pending;48 reused outputs directly reviewed',flush=True)
