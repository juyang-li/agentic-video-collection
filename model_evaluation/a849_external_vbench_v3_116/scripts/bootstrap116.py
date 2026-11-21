"""Freeze candidate116: retained88 +16 Celeb Hair +12 Identity. No quality-based deletion."""
import copy,re,time,shutil
from collections import Counter,defaultdict
from common import *
from reuse import transplant

def main():
 parent=ROOT.parent/'a849_external_vbench_v3_104';previous=rows(parent/'metadata/inputs.jsonl');by={r['id']:r for r in previous}
 assert not(ROOT/'metadata/inputs.jsonl').exists(),'Manifest is immutable'
 hairs=read(ROOT/'metadata/hair_reference_selection.json');identities=read(ROOT/'metadata/identity_reference_selection.json');assert len(hairs)==16 and len(identities)==12
 assert len({r['person_id']for r in hairs+identities})==28
 keep=[1,2,3,4,6,8,9,10];tasks=[];origins={};targets={m:[]for m in ['ltx_base','a849','vace']}
 def add(sid,newid,choice=None,group=None):
  item=copy.deepcopy(by[sid]);item.update(id=newid,legacy_task_id=sid,source_task_id=sid,review_group=re.sub(r'_\d+$','',newid),cohort='candidate28'if choice else'retained88')
  if choice:
   group=group or'hair';item['group']=group;item['references']=[dict(absolute_path=choice['processed_path'],sha256=choice['processed_sha256'],type=group,view='front')];item['reference_selection']=choice;item['subgroup']='celeb_reference';item['task_origin']='self_created_'+group+'_reference_edit; no GT';item['revision_reason']='User requested additional distinct Celeb candidates for later selection; replace problematic old references before generation, keep all outcomes';item['candidate_reference_index']=choice['selection_index']
   if group=='hair'and choice.get('edit_target'):
    target=choice['edit_target'];item['edit_target']=target;item['prompt']['en']=item['prompt']['en'].replace("Change only the main person's hairstyle",'Change only the hairstyle of '+target)+' Keep the man in the background unchanged.';item['vace_description']=item['vace_description'].replace('The same main person',target[0].upper()+target[1:])+' Keep the man in the background unchanged.'
   for model in targets:targets[model].append(newid)
  origins[newid]=[str(parent),sid];tasks.append(item)
 for index,n in enumerate(keep):
  add(f'hair_{n:02d}',f'hair_{index*2+1:02d}');add(hairs[index]['source_id'],f'hair_{index*2+2:02d}',hairs[index])
 for index in range(8,16):add(hairs[index]['source_id'],f'hair_{index+9:02d}',hairs[index])
 # Male source01; female sources02/03/04. Repeated source clips are explicitly tracked.
 females=0
 for index,choice in enumerate(identities,1):
  if choice['attributes']['Male']:sid='identity_01'
  else:sid=f'identity_{[2,3,4][females%3]:02d}';females+=1
  add(sid,f'identity_{index:02d}',choice,'identity')
 counters=defaultdict(int)
 for prior in sorted(previous,key=lambda r:r['id']):
  if prior['group']in ['hair','identity']:continue
  prefix=re.sub(r'_\d+$','',prior['id']);counters[prefix]+=1;add(prior['id'],f'{prefix}_{counters[prefix]:02d}')
 tasks.sort(key=lambda r:r['id']);assert len(tasks)==116 and len({r['id']for r in tasks})==116
 assert Counter(r['cohort']for r in tasks)==dict(retained88=88,candidate28=28)
 assert Counter(r['group']for r in tasks)==dict(hair=24,identity=12,hair_color=16,background=16,tripvvt=24,native=24)
 for p in ['metadata','logs','conditions','cache','inputs/prepared','outputs','review','runtime']:(ROOT/p).mkdir(parents=True,exist_ok=True)
 jsonl(ROOT/'metadata/inputs.jsonl',tasks);manifest=sha(ROOT/'metadata/inputs.jsonl');atomic(ROOT/'metadata/input_origins.json',origins);atomic(ROOT/'metadata/generation_targets.json',targets)
 atomic(ROOT/'metadata/renumbering.json',[dict(id=i['id'],legacy_task_id=i['legacy_task_id'],parent=origins[i['id']],group=i['group'],subgroup=i.get('subgroup'),cohort=i['cohort'])for i in tasks])
 atomic(ROOT/'metadata/freeze.json',dict(revision='a849_v3_candidate116_four_models_pdd8',input_manifest_sha256=manifest,groups=dict(Counter(i['group']for i in tasks)),cohorts=dict(Counter(i['cohort']for i in tasks)),smoke=['identity_01','hair_02','hair_color_02','tripvvt_03','object_removal_01','accessory_03'],models=['ltx_base','a849','vace','h3_pdd8'],protocol=dict(fps=12,num_frames=49,time_window=[0,4],seed=42),retained_original_hair_ids=[f'hair_{n:02d}'for n in keep],review_order='IDs contiguous per prefix, sorted alphabetically',hair_order='Original8 retained at odd01..15; Celeb8 interleaved even02..16; extra8 at17..24',candidate_pool_not_final_selection=True,no_vlm_api=True,created_unix=time.time()))
 for item in tasks:
  src,sid=origins[item['id']];prior=by[sid];roles=['source']+(['text']if item['prompt']['en']==prior['prompt']['en']else[])+([f'ref_{n}'for n in range(len(item['references']))]if item['references']==prior['references']else[]);transplant(Path(src),sid,item,roles=roles)
 for name in ['training_overlap.json','task_limitations.json','vace_prompt_corrections.json','hair_color_templates.json','execution_state.json']:shutil.copy2(parent/'metadata'/name,ROOT/'metadata'/name)
 used={r['absolute_path']for i in tasks if i['group']in ['hair','identity']for r in i['references']};refs=read(parent/'metadata/reference_processing.json')+hairs+identities;atomic(ROOT/'metadata/reference_processing.json',[r for r in refs if r['processed_path']in used])
 selected=[]
 for item in tasks:
  for old in rows(parent/'metadata/selection.jsonl'):
   if old['id']==item['legacy_task_id']:r=copy.deepcopy(old);r['id']=item['id'];selected.append(r)
 jsonl(ROOT/'metadata/selection.jsonl',selected)
 atomic(ROOT/'metadata/input_review.json',dict(status='pending',new_case_ids=targets['ltx_base'],parent104_input_review=str(parent/'metadata/input_review.json')))
 print('Frozen116',manifest,'new28, retained88',flush=True)
if __name__=='__main__':main()
