"""CPU: assemble v3 candidates; final identity decision is required before freezing."""
import copy
from collections import Counter
from common import *
OLD=ROOT.parent/'a849_external_vbench_v2_96'
DIAG=ROOT.parent/'a849_external_vbench_v3_identity'
UE=Path('/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5/data/uedata/outfit_camera_first_frame/train/samples.jsonl')

def build():
 previous={r['id']:r for r in rows(OLD/'metadata/inputs.jsonl')};tasks=[]
 def retain(sid,subgroup=None):
  r=copy.deepcopy(previous[sid]);r['old_id']=sid;r['revision_reason']='Unchanged v2 input and generation protocol'
  if subgroup:r['subgroup']=subgroup
  tasks.append(r);return r
 for group in ['hair','background']:
  for sid,r in previous.items():
   if r['group']==group:retain(sid)
 targets={
  1:'the foreground woman on the left with long black hair and a dark belted jacket',
  3:'the foreground woman in the white cropped top and ripped gray trousers',
  4:'the woman on the right with her hair in a bun, a black top and gray shorts',
  7:'the central foreground dancer in the white top with yellow patterned sleeves and gray shorts',
  8:'the woman on the right in the cream outfit, holding the red suitcase',
  12:'the foreground woman in the colorful cropped shirt and black trousers in the gym',
  16:'the foreground woman in the black hat and black-and-tan jacket with a long white skirt',
  17:'the foreground woman in the dark patterned sleeveless top and black trousers using her phone',
  20:'the real woman on the right in the red-and-yellow top and black skirt, rather than the reflected image on the left',
  21:'the foreground woman holding a green scarf and wearing the long striped dress',
  22:'the woman in the pale blue short dress beside the man in the patterned shirt',
  23:'the foreground woman descending the stairs in the black dress and white shoes',
  24:'the foreground woman with short dark hair wearing the black patterned dress with red trim'}
 for n in range(1,25):
  r=retain(f'tripvvt_{n:02d}')
  if n in targets:
   target=targets[n];r['prompt']['en']=r['prompt']['en'].replace("the target person's",target+"'s")
   r['prompt']['zh']+=' 编辑对象为：'+target+'。'
   r['vace_description']=r['vace_description'].replace('The target person',target[0].upper()+target[1:])
   r['edit_target']=target;r['revision_reason']='Disambiguate source foreground person; retain v2 source, garment reference and source-only tracked mask'
 # Exactly the four face tasks requested, with stable old IDs.
 for sid in ['native_03','native_20','native_23','native_24']:retain(sid,'face')
 for sid in ['native_05','native_06','native_14','native_15','native_16','native_19']:retain(sid,'clothing_local_attribute')
 template=previous['native_03']
 accessory_specs=[
  ('identity_01','sunglasses','Add black sunglasses to the man by the window. Preserve his facial identity, hair, clothing, expression timing, body motion, background and camera.',[-.14,.23,1.14,.57]),
  ('identity_10','eyeglasses','Add thin silver-rimmed eyeglasses to the man in the blue shirt. Preserve his facial identity, hair, clothing, facial expression and hand gesture timing, background and camera.',[-.12,.23,1.12,.58]),
  ('identity_06','earrings','Add a pair of silver hoop earrings to the woman in the pale blazer. Keep natural hair occlusion; preserve her face, hairstyle, blazer, movement, environment and camera.',[-.20,.40,1.20,1.13]),
  ('identity_06','necklace','Add a white pearl necklace around the neck of the woman in the pale blazer. Preserve her face, hair, blazer, body movement, background and camera.',[-.15,.90,1.15,1.65]),
  ('hair_02','cap','Add a red baseball cap on the person in the gray T-shirt. Preserve their facial identity, existing hair below the cap, clothing, movement, street and camera.',[-.30,-.65,1.30,.25]),
  ('identity_13','scarf','Add a blue silk scarf around the neck of the woman seated at the laptop. Preserve her face, hair, black top, hand gestures, laptop, room and camera.',[-.25,.88,1.25,1.72]),
 ]
 for i,(sid,kind,prompt,roi) in enumerate(accessory_specs,1):
  r=copy.deepcopy(previous[sid]);r.update(id=f'accessory_{i:02d}',old_id=None,source_old_id=sid,group='native',subgroup='accessory',references=[],task_origin='self_created',revision_reason='New self-created accessory task on directly inspected source',prompt={'en':prompt,'zh':'自建配饰任务；以完整英文指令为准。'},vace_description=prompt,mask_spec={'kind':'face_relative_accessory','accessory':kind,'relative_roi':roi,'parsing_labels':[11],'invert':False,'dilation_pixels':4})
  r.pop('original_omni_instruction',None);tasks.append(r)
 for candidate in read(ROOT/'metadata/object_candidates.json'):
  uid=candidate['id'];add='_adding_' in uid;n=int(uid[-2:]);group='object_addition' if add else 'object_removal'
  s=candidate['probe'];w,h=s['dimensions']
  # Coordinates refer to the active source image, before letterbox padding.
  box=({2:[.33,.58,.51,.80],4:[.50,.15,.72,.47],6:[.50,.67,.64,.93],10:[.55,.49,.67,.80]} if add else {2:[.53,.79,.79,.97],6:[.14,.06,.50,.87],8:[.50,.15,.91,1.0],10:[.31,.17,.65,.86]})[n]
  r=copy.deepcopy(template);r.update(id=f'{group}_{n:02d}',old_id=None,uid=uid,source={'absolute_path':candidate['path'],'sha256':candidate['source_sha256']},canvas_wh=[1280,736] if w>=h else [736,1280],group='native',subgroup=group,references=[],task_origin='official',prompt={'en':candidate['instruction'],'zh':'OmniEdit 官方任务；保留英文原文。'},original_omni_instruction=candidate['instruction'],vace_description=candidate['instruction'],mask_spec={'kind':'insertion_roi' if add else 'tracked_object','source_normalized_box':box,'parsing_labels':[],'invert':False,'dilation_pixels':8},revision_reason='New official object task with source-planned insertion region or object track',source_review={'reviewer':'Codex direct visual inspection; no VLM API','sheets':['object_candidates.jpg'],'full_duration_timestamps':s['review_timestamps'],'notes':'Official target and full source time series inspected; region planned before generation.'},official_task=candidate['official'])
  tasks.append(r)
 # Actual training instruction words, avoiding catalog aliases such as gold.
 prompts={}
 for line in UE.open():
  r=json.loads(line)
  if r.get('axis')!='hair_color':continue
  color=r['target']['attributes']['hair_color_id'];p=r['prompt']
  if p['en'].startswith("Using the provided source video, apply the following edit: Please change the person's hair color to "):
   prompts.setdefault(color,dict(prompt=p,sample_id=r['sample_id'],manifest=str(UE)))
 assert set(prompts)==set(range(1,8)),prompts.keys()
 atomic(ROOT/'metadata/hair_color_templates.json',{'manifest':str(UE),'manifest_sha256':sha(UE),'templates':prompts})
 color_specs=[('hair_01',1,'blonde'),('hair_02',2,'light brown'),('hair_03',5,'blonde'),('hair_05',4,'blonde'),('hair_07',6,'dark brown'),('hair_08',3,'brown'),('hair_09',2,'light brown'),('hair_10',6,'blonde'),('hair_12',5,'brown'),('identity_14',7,'dark brown'),('hair_15',7,'brown'),('hair_16',6,'dark brown'),('identity_06',1,'blonde'),('identity_13',3,'brown'),('identity_03',2,'blonde'),('identity_04',5,'dark brown')]
 for i,(sid,color,source_color) in enumerate(color_specs,1):
  r=copy.deepcopy(previous[sid]);prompt=copy.deepcopy(prompts[color]['prompt']);target=None
  if sid=='hair_10':target='the foreground woman in the gray top and sunglasses'
  if sid=='hair_15':target='the foreground woman on the right in the black top'
  if sid=='hair_16':target='the woman on the left in the pale short-sleeved top'
  if target:
   prompt['en']=prompt['en'].replace("the person's",target+"'s");prompt['zh']+=' 编辑对象为：'+target+'。'
  words=prompt['en'].rsplit(' to ',1)[1].rstrip('.')
  r.update(id=f'hair_color_{i:02d}',old_id=None,source_old_id=sid,group='hair_color',subgroup=words,axis='hair_color',mode='attribute',references=[],prompt=prompt,vace_description=f"Change only {target or 'the person'}'s hair color to {words}. Preserve the original hairstyle, facial identity, expression, clothing, body motion, other people, environment and camera.",mask_spec=dict(parsing_labels=[2],invert=False,dilation_pixels=4,**({'actor_box':previous[sid]['mask_spec']['actor_box']} if 'actor_box' in previous[sid]['mask_spec'] else {})),task_origin='self_created_training_template',revision_reason='New hair-color-only edit using exact training template and observed contrasting source color',template_provenance=prompts[color],source_hair_color_observed=source_color,edit_target=target or 'the single main person',preservation_checks=['hairstyle','face','clothing','environment','other_people'])
  if sid=='hair_10':r['mask_spec']['actor_box']=[.05,.02,.75,.99]
  r.pop('original_omni_instruction',None);tasks.append(r)
 return tasks

def main():
 tasks=build();(ROOT/'metadata/candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in tasks))
 print('Candidates',len(tasks),dict(Counter(r['group'] for r in tasks)))
if __name__=='__main__':main()
