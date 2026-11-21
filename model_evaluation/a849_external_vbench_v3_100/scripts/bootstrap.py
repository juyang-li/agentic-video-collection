"""Freeze original96 plus4 reviewed Identity and4 non-political Celeb hairstyle cases."""
import copy,shutil,time
from collections import Counter
from common import *
from contracts import generation_input,digest
from reuse import transplant,link
OLD=ROOT.parent/'a849_external_vbench_v3_96'
DIAG=ROOT.parent/'a849_external_vbench_v3_identity_fullhead'

def main():
 for folder in ['metadata','logs','review','conditions','cache','inputs/prepared','outputs/ltx_base','outputs/a849','outputs/vace','outputs/h3_pdd8','runtime']:(ROOT/folder).mkdir(parents=True,exist_ok=True)
 assert not (ROOT/'metadata/inputs.jsonl').exists(),'Do not overwrite a frozen104 manifest'
 original=rows(OLD/'metadata/inputs.jsonl');identities=rows(DIAG/'metadata/inputs.jsonl');refs=read(DIAG/'metadata/freeze.json')['reference_processing']
 by_ref={Path(r['path']).stem:r for r in refs};by_id={r['id']:r for r in original}
 items=[];origins={}
 for r in original:
  item=copy.deepcopy(r);item.update(cohort='original96',source_task_id=r['id']);items.append(item);origins[item['id']]=[str(OLD),r['id']]
 for r in identities:
  item=copy.deepcopy(r);item.update(cohort='supplement8',source_task_id=r['id'],task_origin='self_created_identity_reference_edit; user requested supplement20261114',revision_reason='User explicitly includes the4 full-head natural-hair Identity cases as supplementary cases; retain all outcomes regardless of edit success')
  items.append(item);origins[item['id']]=[str(DIAG),r['id']]
 choices=[('hair_01','00017'),('hair_09','00011'),('hair_12','00009'),('hair_08','00023')]
 for index,(source_id,ref_id) in enumerate(choices,1):
  item=copy.deepcopy(by_id[source_id]);ref=by_ref[ref_id]
  item.update(id=f'hair_celeb_{index:02d}',cohort='supplement8',subgroup='celeb_reference',old_id=source_id,source_task_id=source_id,task_origin='self_created_hairstyle_reference_edit; no GT',revision_reason='User adds4 Celeb references outside the original US president/vice-president reference set; original source, hair instruction, source-only mask and inference protocol retained',reference_selection=dict(dataset='CelebA-HQ1024',portrait_id=ref_id,excluded_original_reference_names=['harris','trump','obama','biden'],selection_basis='Direct source/reference inspection, differing shape/length rather than hair-color target; no result-based selection'))
  item['references']=[dict(absolute_path=ref['processed_path'],sha256=ref['processed_sha256'],type='hair',view='front')]
  items.append(item);origins[item['id']]=[str(OLD),source_id]
 assert len(items)==104 and len({r['id']for r in items})==104
 jsonl(ROOT/'metadata/inputs.jsonl',items)
 atomic(ROOT/'metadata/freeze.json',dict(revision='a849_v3_104_four_models_pdd8',input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),groups=dict(Counter(r['group']for r in items)),cohorts=dict(Counter(r['cohort']for r in items)),smoke=['identity_01','hair_celeb_01','hair_color_02','tripvvt_03','object_removal_02','accessory_03'],protocol=dict(fps=12,num_frames=49,time_window=[0,4],seed=42),models=['ltx_base','a849','vace','h3_pdd8'],h3='Exact README Ref2VA8-step PDD; native768-short-edge24fps; shared0–4s49 timestamp scoring projection',initial_identity_diagnostic_preserved=True,no_vlm_api=True,created_unix=time.time()))
 atomic(ROOT/'metadata/input_origins.json',origins)
 for item in items:
  parent,old_id=origins[item['id']];parent=Path(parent)
  prior=next(r for r in rows(parent/'metadata/inputs.jsonl')if r['id']==old_id)
  roles=['source','text']+([f'ref_{i}'for i in range(len(item['references']))]if item['references']==prior['references']else[])
  transplant(parent,old_id,item,roles=roles)
  gt=parent/'inputs/prepared'/old_id/'gt_review_only.mp4'
  if gt.exists():assert (ROOT/'inputs/prepared'/item['id']/'gt_review_only.mp4').is_file()
 for filename in ['selection.jsonl','training_overlap.json','task_limitations.json','reference_processing.json','vace_prompt_corrections.json','hair_color_templates.json']:
  shutil.copy2(OLD/'metadata'/filename,ROOT/'metadata'/filename)
 atomic(ROOT/'metadata/input_review.json',dict(status='pending',new_case_ids=[r['id']for r in items if r['id'].startswith('hair_celeb_')],source_and_masks='Existing directly reviewed source sequences and source-only hair masks; new full-head reference/source pairing requires direct inspection before generation',new_reference_paths=[r['references'][0]['absolute_path']for r in items if r['id'].startswith('hair_celeb_')],identity_review=str(DIAG/'metadata/identity_review.json')))
 print('Frozen104 cases; source/text/unchanged-reference encodings transplanted without modifying parents.',flush=True)
if __name__=='__main__':main()
