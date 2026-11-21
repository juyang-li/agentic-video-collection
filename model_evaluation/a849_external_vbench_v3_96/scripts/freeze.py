"""Freeze the main cohort only after all twelve identity outputs were reviewed."""
from collections import Counter,defaultdict
from common import *
from contracts import identity_decision,MODELS
from reuse import transplant
OLD=ROOT.parent/'a849_external_vbench_v2_96';DIAG=Path(read(ROOT/'metadata/identity_diagnostic_location.json')['root']) if (ROOT/'metadata/identity_diagnostic_location.json').exists() else ROOT.parent/'a849_external_vbench_v3_identity'

def main():
 review=read(DIAG/'metadata/identity_review.json');assert review['inspection_complete'] is True
 diag_items=rows(DIAG/'metadata/inputs.jsonl');assert {r['id'] for r in review['tasks']}=={r['id'] for r in diag_items}
 for task in review['tasks']:
  for model in MODELS:
   r=read(DIAG/'outputs'/model/f"{task['id']}.json");v=task['models'][model]
   assert r['status']=='complete' and v['reviewed'] is True
   assert v['output_sha256']==r['output_sha256']==sha(DIAG/'outputs'/model/f"{task['id']}.mp4")
 decision=identity_decision(review['tasks']);candidates=rows(ROOT/'metadata/candidates.jsonl')
 items=[r for r in candidates if r['group']!='hair_color' or int(r['id'].split('_')[-1])<=decision['hair_color_count']]
 if not decision['drop_identity']:items=diag_items+items
 assert len(items)==96 and len({r['id'] for r in items})==96
 counts=dict(Counter(r['group'] for r in items));assert counts.get('identity',0)==decision['identity_count']
 assert counts['native']==24 and counts['hair_color']==decision['hair_color_count']
 assert Counter(r['subgroup'] for r in items if r['group']=='native')==dict(face=4,accessory=6,object_addition=4,object_removal=4,clothing_local_attribute=6)
 assert all(r['group'] not in ['action','camera'] for r in items)
 jsonl(ROOT/'metadata/inputs.jsonl',items)
 atomic(ROOT/'metadata/freeze.json',dict(revision='a849_v3_96_natural_hair_temporal_flickering',input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),groups=counts,smoke=[],identity_decision=decision,diagnostic_manifest_sha256=sha(DIAG/'metadata/inputs.jsonl'),identity_review_sha256=sha(DIAG/'metadata/identity_review.json'),input_visual_review='direct source/ref inspection; source sheets reviewed; mask freeze required before generation',protocol={'fps':12,'num_frames':49,'time_window':[0,4],'seed':42},no_vlm_api=True,v2_preserved=True))
 previous={r['id']:r for r in rows(OLD/'metadata/inputs.jsonl')}
 selection={r['id']:r for r in rows(OLD/'metadata/selection.jsonl')}
 for item in items:
  if item['group']=='identity':root=DIAG;sid=item['id'];prior=next(r for r in diag_items if r['id']==sid)
  else:root=OLD;sid=item.get('old_id') or item.get('source_old_id');prior=previous.get(sid)
  if prior:
   roles=['source']
   if item['prompt']['en']==prior['prompt']['en']:roles.append('text')
   if item['references']==prior['references']:roles.extend(f'ref_{i}' for i in range(len(item['references'])))
   transplant(root,sid,item,roles=roles,conditions=item['mask_spec']==prior['mask_spec'])
  else:
   path=ROOT/'inputs/prepared'/item['id']/'pixels.json';r=read(path)
   assert r['source_sha256']==item['source']['sha256'];r['input_manifest_sha256']=sha(ROOT/'metadata/inputs.jsonl');atomic(path,r)
 corrections={}
 for item in items:
  cp=ROOT/'conditions'/item['id']/'condition.json'
  if cp.exists() and read(cp).get('candidate_staging'):
   c=read(cp);assert c['spec']==item['mask_spec'] and c['status']=='complete'
   assert c['input_manifest_sha256']==sha(ROOT/'metadata/candidates.jsonl')
   c.update(input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),candidate_staging=False,source_pixels_sha256=sha(ROOT/'inputs/prepared'/item['id']/'pixels.json'))
   atomic(cp,c)
  if cp.exists() and read(cp).get('status')=='complete':
   c=read(cp);derived='A continuous video. '+item['vace_description']+'\nOriginal edit specification: '+item['prompt']['en']
   if c['adapted_prompt']!=derived:
    oldc=OLD/'conditions'/item['old_id']/'condition.json'
    assert item['id']=='native_03' and c['adapted_prompt']==read(oldc)['adapted_prompt']
    corrections[item['id']]=dict(actual_adapted_prompt=c['adapted_prompt'],original_v2_condition=str(oldc),original_v2_condition_sha256=sha(oldc),reason='Preserve v2 pre-inference correction from woman/Her to neutral person/Their; official instruction unchanged.')
 atomic(ROOT/'metadata/vace_prompt_corrections.json',corrections)
 jsonl(ROOT/'metadata/selection.jsonl',[dict(selection[r['id']]) for r in items if r['group']=='tripvvt'])
 clusters=defaultdict(list)
 for r in items:clusters[r['source']['sha256']].append(r['id'])
 atomic(ROOT/'metadata/source_clusters.json',dict(task_count=96,unique_source_hashes=len(clusters),clusters=dict(clusters),note='Different edit tasks may share source; no v2/v3 progress claim.'))
 atomic(ROOT/'metadata/revision_log.json',[{k:r.get(k) for k in ['id','old_id','source_old_id','uid','task_origin','revision_reason','edit_target']} for r in items])
 refs=read(ROOT/'metadata/reference_processing.json')
 for r in refs:r.setdefault('name',Path(r['path']).stem)
 refs += [r for r in read(OLD/'metadata/reference_processing.json') if r.get('role')=='hair']
 refs=list({(r['processed_path'],r.get('role')):r for r in refs}.values())
 atomic(ROOT/'metadata/reference_processing.json',refs)
 print('Frozen',counts,decision,flush=True)
if __name__=='__main__':main()
