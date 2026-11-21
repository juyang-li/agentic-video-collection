"""Freeze inspected32 input tasks; official caption/GT remain audit-only metadata."""
import time
from collections import Counter
from common import *
from select_pool import POOL

def main():
    review=read(ROOT/'metadata/source_review.json')
    assert review['status']=='approved' and review['scope']=='project_test_holdout'
    records=rows(ROOT/'metadata/provisional32.jsonl');notes=review['cases']
    assert set(notes)=={r['pilot_id'] for r in records} and len(records)==32
    items=[];selection=[]
    for r in records:
        sid=r['pilot_id']; note=notes[sid]
        assert note['input_acceptable'] is True and note['edit_target']
        source=POOL/r['source'];ref=POOL/r['reference'];target=POOL/r['target']
        assert sha(source)==r['source_sha256'] and sha(ref)==r['reference_sha256'] and sha(target)==r['target_sha256']
        import av
        with av.open(str(source)) as c: w,h=c.streams.video[0].width,c.streams.video[0].height
        garment={'upper_body':'upper-body garment','lower_body':'lower-body garment','dresses':'dress'}[r['category']]
        person=note['edit_target']
        prompt=f"Replace only the {garment} worn by {person} with the garment shown in the reference image. Preserve the person's facial identity, hairstyle, body shape, motion and timing, all other clothing, other people, background, lighting and camera."
        mask=dict(parsing_labels={'upper_body':[4,7],'lower_body':[5,6,7],'dresses':[4,5,6,7]}[r['category']],invert=False,dilation_pixels=12,main_person=True)
        if note.get('actor_box'):mask['actor_box']=note['actor_box']
        item=dict(id=sid,uid=f"tripvvt:{r['metadata_index']}",dataset='tripvvt',group='tripvvt',subgroup=r['category'],
            source=dict(absolute_path=str(source),sha256=r['source_sha256']),
            references=[dict(absolute_path=str(ref),sha256=r['reference_sha256'],type='garment',view='official')],
            prompt=dict(en=prompt,zh='自建换衣指令；完整官方 caption 独立保留，不用于推理。'),
            canvas_wh=[1280,736] if w>h else [736,1280],fps=12,num_frames=49,timestamps_seconds=[i/12 for i in range(49)],
            mask_spec=mask,vace_description=f"The same source person, {person}, wears the garment shown in the official reference image. Preserve facial identity, hairstyle, motion, other garments, other people, environment and camera.",
            edit_target=person,cohort='pilot32',review_group='tripvvt',legacy_task_id=None,
            task_origin='self_created_reference_conditioned_instruction_with_official_TripVVT_garment',source_review=note,
            provenance=dict(source_manifest=str(POOL/'manifest.jsonl'),project_split=r['split'],official_split=r['original_metadata']['official_split'],group_id=r['group_id'],original_sample_id=r['sample_id'],repo_id=r['repo_id'],revision=r['revision']),
            revision_reason='New pre-frozen pilot input; selection after outputs disclosed separately')
        assert 'target' not in item and 'caption' not in item
        items.append(item)
        selection.append(dict(id=sid,record=dict(source=item['source'],references=item['references'],target=dict(absolute_path=str(target),sha256=r['target_sha256']),caption=r['caption'],provenance=item['provenance'])))
    assert Counter(i['subgroup']for i in items)==dict(upper_body=10,lower_body=10,dresses=12)
    jsonl(ROOT/'metadata/inputs.jsonl',items);jsonl(ROOT/'metadata/selection.jsonl',selection)
    manifest=sha(ROOT/'metadata/inputs.jsonl')
    atomic(ROOT/'metadata/freeze.json',dict(revision='tripvvt_pilot32_project_holdout',input_manifest_sha256=manifest,
        seed=20261115,models=['a849','h3_pdd8'],smoke=[items[0]['id']],groups={'tripvvt':32},
        quotas=dict(Counter(i['subgroup']for i in items)),source_scope='project test; actual official splits in provenance',created_unix=time.time(),no_vlm_api=True))
    atomic(ROOT/'metadata/generation_targets.json',{'a849':[r['id']for r in items]})
    audit=read(ROOT/'metadata/pool_audit.json');audit.update(status='passed',scope_confirmation='User approved project-test holdout and strict A849 lineage exclusion');atomic(ROOT/'metadata/pool_audit.json',audit)
    print('Frozen32',manifest,flush=True)

if __name__=='__main__':main()
