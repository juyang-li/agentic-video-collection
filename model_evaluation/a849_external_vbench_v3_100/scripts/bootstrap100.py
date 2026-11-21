"""Freeze final100: retained88 +8 CelebHair +4Identity; contiguous IDs sorted alphabetically."""
import copy,re,time,shutil
from collections import Counter,defaultdict
from common import *
from reuse import transplant

def main():
 parent=ROOT.parent/'a849_external_vbench_v3_104';items=rows(parent/'metadata/inputs.jsonl');by_id={i['id']:i for i in items};selection=read(ROOT/'metadata/new_hair_selection.json');assert len(selection)==4
 assert not(ROOT/'metadata/inputs.jsonl').exists(),'Final manifest is immutable'
 for name in ['metadata','logs','conditions','cache','inputs/prepared','outputs','review','runtime']:(ROOT/name).mkdir(parents=True,exist_ok=True)
 tasks=[];origins={};targets={m:[]for m in['ltx_base','a849','vace']};mapping=[]
 keep=[1,2,3,4,6,8,9,10]
 # Alternate retained Hair and Celeb Hair, preserving the user's retained order.
 sequence=[]
 for n,oldnum in enumerate(keep):
  sequence.append(('retained',f'hair_{oldnum:02d}',None))
  if n%2==0:sequence.append(('existing',f'hair_celeb_{n//2+1:02d}',None))
  else:sequence.append(('extra',selection[n//2]['source_id'],selection[n//2]))
 for index,(kind,sid,choice)in enumerate(sequence,1):
  item=copy.deepcopy(by_id[sid]);old_id=item['id'];item['id']=f'hair_{index:02d}';item['legacy_task_id']=old_id;item['review_group']='hair';item['cohort']='retained88'if kind=='retained'else'new12';item['hair_case_kind']=kind;item['source_task_id']=sid
  if kind=='extra':
   if choice.get('edit_target'):
    target=choice['edit_target'];item['edit_target']=target;item['prompt']['en']=item['prompt']['en'].replace("Change only the main person's hairstyle",'Change only the hairstyle of '+target)+' Keep the man in the background unchanged.';item['vace_description']=item['vace_description'].replace('The same main person',target[0].upper()+target[1:])+' Keep the man in the background unchanged.'
   item['references']=[dict(absolute_path=choice['processed_path'],sha256=choice['processed_sha256'],type='hair',view='front')];item['task_origin']='self_created_hairstyle_reference_edit; no GT';item['subgroup']='celeb_reference';item['reference_selection']=choice;item['revision_reason']='User20261114 adds4 more Celeb Hair; original source, prompt, source-only mask and inference protocol retained; reference changed before generation'
  item['retained_original_hair_id']=old_id if kind=='retained'else None
  origins[item['id']]=[str(parent),sid];tasks.append(item)
  if kind!='retained':targets['vace'].append(item['id'])
  if kind=='extra':
   for model in['ltx_base','a849']:targets[model].append(item['id'])
  mapping.append(dict(id=item['id'],legacy_task_id=old_id,kind=kind,source_id=sid,reference_sha256=item['references'][0]['sha256']))
 remaining=[i for i in items if i['group']!='hair']
 counters=defaultdict(int)
 for prior in sorted(remaining,key=lambda r:r['id']):
  item=copy.deepcopy(prior);prefix=re.sub(r'_\d+$','',item['id']);counters[prefix]+=1;old_id=item['id'];item['id']=f'{prefix}_{counters[prefix]:02d}';item['legacy_task_id']=old_id;item['review_group']=prefix;item['cohort']='new12'if item['group']=='identity'else'retained88';item['source_task_id']=old_id;origins[item['id']]=[str(parent),old_id];tasks.append(item)
 tasks.sort(key=lambda i:i['id']);assert len(tasks)==100 and len({i['id']for i in tasks})==100 and sum(i['group']=='native'for i in tasks)==24
 assert Counter(i['cohort']for i in tasks)==dict(retained88=88,new12=12)
 jsonl(ROOT/'metadata/inputs.jsonl',tasks);manifest=sha(ROOT/'metadata/inputs.jsonl')
 atomic(ROOT/'metadata/input_origins.json',origins);atomic(ROOT/'metadata/generation_targets.json',targets);atomic(ROOT/'metadata/hair_id_mapping.json',mapping)
 atomic(ROOT/'metadata/renumbering.json',[dict(id=i['id'],legacy_task_id=i['legacy_task_id'],parent=origins[i['id']],group=i['group'],subgroup=i.get('subgroup'))for i in tasks])
 atomic(ROOT/'metadata/freeze.json',dict(revision='a849_v3_final100_four_models_pdd8',input_manifest_sha256=manifest,groups=dict(Counter(i['group']for i in tasks)),cohorts=dict(Counter(i['cohort']for i in tasks)),smoke=['identity_01','hair_04','hair_color_02','tripvvt_03','object_removal_01','accessory_03'],models=['ltx_base','a849','vace','h3_pdd8'],protocol=dict(fps=12,num_frames=49,time_window=[0,4],seed=42),retained_original_hair_ids=[f'hair_{n:02d}'for n in keep],removed_original_hair_ids=[f'hair_{n:02d}'for n in range(1,17)if n not in keep],hair_order='retained/Celeb alternating; former4 and added4 Celeb interleaved',review_order='contiguous zero-padded IDs sorted alphabetically',no_vlm_api=True,created_unix=time.time()))
 for item in tasks:
  src,sid=origins[item['id']];prior=by_id[sid];roles=['source']+(['text']if item['prompt']['en']==prior['prompt']['en']else[])+([f'ref_{n}'for n in range(len(item['references']))]if item['references']==prior['references']else[]);transplant(Path(src),sid,item,roles=roles)
 for name in ['training_overlap.json','task_limitations.json','vace_prompt_corrections.json','hair_color_templates.json','reference_processing.json','execution_state.json']:shutil.copy2(parent/'metadata'/name,ROOT/'metadata'/name)
 refs=read(ROOT/'metadata/reference_processing.json')+selection;atomic(ROOT/'metadata/reference_processing.json',refs)
 # Remap review-only TripVVT selections to the final IDs; GT never becomes an inference input.
 selected=[];previous=rows(parent/'metadata/selection.jsonl')
 for item in tasks:
  matches=[r for r in previous if r['id']==item['legacy_task_id']]
  for r in matches:r=copy.deepcopy(r);r['id']=item['id'];selected.append(r)
 jsonl(ROOT/'metadata/selection.jsonl',selected)
 atomic(ROOT/'metadata/input_review.json',dict(status='pending',new_case_ids=targets['ltx_base'],new_reference_sources=[r['path']for r in selection],frozen_reference_and_source_audit='Direct source/reference/mask review required before generation',parent104_input_review=str(parent/'metadata/input_review.json')))
 print('Frozen final100; Hair16 alternating retained/Celeb; contiguous alphabetic IDs.',flush=True)
if __name__=='__main__':main()
