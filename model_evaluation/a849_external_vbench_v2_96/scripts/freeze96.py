"""Freeze96 after source-only direct inspection, before new inference."""
from collections import Counter,defaultdict
from common import *

OLD=ROOT.parent/'a849_external_vbench_v1'

def media(path,**kw):return dict(absolute_path=str(path),sha256=sha(path),**kw)

def main():
    require_job()
    inputs=rows(OLD/'metadata/inputs.jsonl');selection=rows(OLD/'metadata/selection.jsonl')
    candidates={r['id']:r for r in rows(ROOT/'metadata/omni_candidates.jsonl')}
    processed=read(ROOT/'metadata/reference_processing.json')
    refmap={(r['name'],r['role']):r for r in processed}
    names=['trump','obama','biden','harris']
    for item in inputs:
        if item['group'] not in ['identity','hair']:continue
        i=int(item['id'].split('_')[-1])-1;name=names[i] if item['group']=='identity' else ['harris','trump','obama','biden'][i]
        r=refmap[(name,item['group'])]
        item['references']=[media(r['processed_path'],type=item['group'],view='front')]
    templates={g:next(r for r in inputs if r['group']==g) for g in ['identity','hair','background']}
    def custom(group,index,uid,ref_name,notes,actor_box=None,target=None):
        item=json.loads(json.dumps(templates[group]));r=candidates[uid]
        item.update(id=f'{group}_{index:02d}',uid=uid,source=media(r['path']),original_omni_instruction=r['instruction'])
        w,h=r['probe']['dimensions'];item['canvas_wh']=[1280,736] if w>=h else [736,1280]
        item['source_review']=dict(reviewer='Codex direct visual inspection; no VLM API',sheets='expanded_omni_candidates_00–29.jpg',full_duration_timestamps=r['probe']['review_timestamps'],notes=notes)
        if group in ['identity','hair']:
            ref=refmap[(ref_name,group)];item['references']=[media(ref['processed_path'],type=group,view='front')]
        else:
            ref=next(x for x in read(OLD/'metadata/reference_processing.json') if x.get('role')=='background' and x['name']==ref_name)
            item['references']=[media(ref['path'],type='scene',view='')];place=ref_name.replace('_',' ')
            item['prompt']['en']=f'Replace only the background with the empty {place} environment in the scene reference. Preserve all source people, faces, hairstyles, clothing, poses, action timing, subject placement and camera motion. Match the people lighting to the new environment. Do not add people.'
            item['prompt']['zh']=f'仅将背景替换为参考图中的无人{dict(living_room="客厅",kitchen="厨房",forest="森林",beach="海滩")[ref_name]}。保留所有原有人物、面部、发型、服装、姿态、动作时序、人物位置和镜头运动，协调人物光照，不添加人物。'
            item['vace_description']=f'All the same source people in unchanged clothing perform their source actions in the empty reference {place}, retaining their placement, pose timing and camera movement.'
            item['subgroup']='indoor' if ref_name in ['living_room','kitchen'] else 'outdoor'
        if target:
            item['prompt']['en']=item['prompt']['en'].replace('the main person',target)
            item['prompt']['zh']+=' 编辑对象明确为'+target+'。'
            item['vace_description']=item['vace_description'].replace('The same main person',target).replace('The main person',target)
        if actor_box:item['mask_spec']['actor_box']=actor_box
        inputs.append(item);selection.append(dict(id=item['id'],official=r['official'],probe=r['probe']))

    identity=[
        ('Sp_Object_style_change_05','Seated man in yellow jacket; face and glasses visible throughout.'),
        ('Sp_Object_visual_effects_adding_04','Standing woman in pale blazer; face clear; short blonde hair.'),
        ('Te_Action_category_change_06','Seated guitarist; face visible with modest head movement; hands and instrument preserved.'),
        ('Te_Facial_expression_change_02','Single woman holding blank board; face visible above board in common window.'),
        ('Te_Facial_expression_change_07','Close-up man in black hood; facial region visible; hood preserved.'),
        ('Te_Facial_expression_change_08','Frontal man in blue shirt; occasional source hand near mouth, retain occlusion and expression timing.'),
        ('Te_Temporal_reordering_01','Close-up speaking man, hand near mouth early; preserve actual source sequence.'),
        ('Te_Temporal_reordering_02','Close-up speaking man outdoors; face clear; preserve source speech and gaze.'),
        ('Te_Motion_deletion_01','Woman at laptop; face visible, hand gestures near mouth; retain actual gesture timing.'),
        ('Sp_Relighting_09','Woman in red dress, frontal face and hair clear; preserve original lighting.'),
        ('Te_Facial_expression_change_06','Close-up man in profile; large facial region visible, partly cropped head retained as source framing.'),
        ('Sp_Relighting_03','Close-up man outdoors, initially downcast face with strong backlight; visibility varies and is retained as a challenge.'),
    ]
    for i,(uid,note) in enumerate(identity):custom('identity',i+5,uid,names[i%4],note)
    hair=[
        ('Sp_Object_style_change_10','harris','Man holding a sign; complete short hair visible in profile.'),
        ('Te_Motion_insertion_01','trump','Single blonde runner, head and ponytail visible in first4 seconds.'),
        ('Te_Motion_deletion_04','obama','Woman in navy shirt moving among trees, tied-back hair visible.'),
        ('Te_Motion_direction_change_01','biden','Woman turns while opening coat; hair seen from front, side and back; preserve actual turn.'),
        ('Sp_Environment_change_02','trump','Woman dancing in field, hair visible during turning in common window.'),
        ('Te_Temporal_delay_03','obama','Foreground woman hiking; hair clear; sunglasses and following person preserved.'),
        ('Te_Motion_frequency_change_01','harris','Woman doing high knees; blonde ponytail visible throughout common window.'),
        ('Sp_Clothes_change_03','obama','Single woman in white dress outdoors; hair visible. Original clothing prompt is provenance only and not used for custom task.'),
        ('Te_Action_content_change_07','harris','Girl skipping rope in gym; whole head visible at modest scale; preserve rope/body motion.'),
        ('Te_Motion_frequency_change_09','harris','Boy juggling football; short hair visible; action and ball remain source-defined.'),
        ('Te_Temporal_synchronization_04','trump','Foreground woman at right eating pizza; distinguish her hair from other people.'),
        ('Te_Motion_insertion_06','biden','Left woman in pale short-sleeved top; edit her hair only, preserve right woman and interaction.'),
    ]
    for i,(uid,name,note) in enumerate(hair):
        target=None;box=None
        if i==10:target='the foreground woman on the right in the black top';box=[.35,.02,.82,.98]
        if i==11:target='the woman on the left in the pale short-sleeved top';box=[0,.02,.60,.99]
        custom('hair',i+5,uid,name,note,box,target)
    backgrounds=[
        ('Te_Motion_speed_change_06','living_room','Single toddler walking through field; retain entire person and actual walking.'),
        ('Sp_Object_style_change_06','kitchen','Hatted hiker in mountains; entire foreground person visible, preserve hat.'),
        ('Sp_Environment_change_03','living_room','Woman and child in ball pit; preserve both people; toys are part of replaced environment.'),
        ('Te_Action_category_change_09','kitchen','Backpacked man by tree; clear foreground body in first4 seconds; preserve backpack.'),
        ('Te_Action_content_change_08','living_room','Parkour man beside wall; main body remains visible through common window; preserve motion.'),
        ('Te_Temporal_reordering_09','kitchen','Single person dancing on beach; visible whole body; preserve actual dance order.'),
        ('Te_Motion_trajectory_change_05','forest','Single runner on track; modest subject scale but unambiguous foreground; retain source stride.'),
        ('Te_Action_category_change_01','beach','Single runner beside river, mainly back view; retain path and body timing.'),
        ('Te_Motion_deletion_03','forest','Single man exercising indoors; large changes of pose and height, foreground persists throughout common window.'),
        ('Te_Motion_direction_change_03','beach','Single woman turning in long coat against brick wall; preserve full coat and actual turn.'),
        ('Te_Temporal_reordering_08','forest','Two women dancing side by side outdoors; preserve both and source interaction.'),
        ('Te_Action_category_change_03','beach','Single woman walking away along ridge; face not needed for environment edit; full body visible.'),
    ]
    for i,(uid,name,note) in enumerate(backgrounds):custom('background',i+5,uid,name,note)

    natives=[
        ('Sp_Appearance_change_01',[11],'The same woman has an elderly grandmother-like face while retaining her source hair, body, expression timing and setting.','将女孩面部变为像祖母的老年女性面部。'),
        ('Sp_Appearance_change_02',[11],'The same woman appears as a younger teenager with softer facial features; source hair, clothing and actions are preserved.','使女性面貌更像年轻少女，五官更柔和。'),
        ('Sp_Appearance_change_05',[11],'The same person has a smoother teenage-looking face, preserving original facial structure, glasses motion and setting.','平滑面部，使人物像青少年，保留原有面部结构。'),
        ('Sp_Appearance_change_07',[11],'The same man looks older, preserving source hair, mustache, clothing, gaze and setting.','让男性看起来更年老。'),
        ('Sp_Appearance_change_09',[11],'The same man has no beard and no facial wrinkles; source red knit cap, glasses, smoking action and surrounding room remain.','去掉男性的胡须与全部皱纹。'),
        ('Sp_Clothes_change_01',[4,5,6,7],'The same woman wears a modern elegant white floor-length evening gown instead of the black abaya, retaining source head covering, body motion and scene.','将黑色长袍换为现代优雅的白色及地晚礼服。'),
        ('Sp_Clothes_change_02',[4,5,6,7],'The same woman wears a yellow linen sundress with thin spaghetti straps; preserve source face, hair, movement and setting.','将服装换为细肩带黄色亚麻吊带裙。'),
        ('Sp_Clothes_change_04',[4,7],'The same person wears a classic blue denim jacket layered over the current gray T-shirt; source face, hair, pose and street remain.','在现有灰色T恤外加经典蓝色牛仔夹克。'),
        ('Sp_Clothes_change_05',[4,7],'The same person wears a soft cream knitted turtleneck sweater instead of the coat; source headwear, lower clothing, motion and setting remain.','将外套改为柔软的奶油色针织高领毛衣。'),
        ('Sp_Clothes_change_06',[4,7],'The same person wears a classic tan trench coat with a belted waist, preserving source red hat, face, table, action and setting.','将服装换为系腰带的经典棕褐色风衣。'),
        ('Sp_Clothes_change_07',[1,4,7,17],'The same person wears a professional navy slim-fit suit instead of graduation gown and cap; retain face, diploma and source actions.','将毕业袍和学位帽换为职业海军蓝修身西装。'),
        ('Sp_Clothes_change_09',[4,7],'The same woman wears a black down jacket, keeping her source face, hair, lower clothing, actions and city backdrop.','将女性服装换为黑色羽绒服。'),
        ('Sp_Clothes_change_10',[4,7],'The same person wears a vibrant yellow puffer jacket instead of the coat and black turtleneck, retaining source hair, face, phone action and background.','将外套与黑色高领衣换为鲜黄色羽绒夹克。'),
        ('Te_Facial_expression_change_03',[11],'The same seated woman has an angrier facial expression while retaining her source phone use, clothes, pose and room.','使女性的表情更愤怒。'),
        ('Te_Facial_expression_change_04',[11],'The same man has a peaceful happy expression while keeping his source appearance, clothing, pose and blue background.','使男性变得平和、开心。'),
        ('Te_Facial_expression_change_05',[11],'The same man looks very sad, with tears flowing down his cheeks; preserve source clothing, body gesture sequence and background.','使男性非常悲伤，泪水流过脸颊。'),
        ('Te_Facial_expression_change_09',[11],'The same woman smiles gently toward the camera while keeping source hair, black top, body pose and dark background.','让女性朝镜头温柔微笑。'),
        ('Te_Facial_expression_change_06',[11],'The same man laughs happily; retain his source hair, clothing, camera framing and surroundings.','让男性开心地大笑。'),
    ]
    for i,(uid,labels,description,zh) in enumerate(natives):
        r=candidates[uid];w,h=r['probe']['dimensions'];sid=f'native_{i+7:02d}'
        inputs.append(dict(id=sid,uid=uid,dataset='omni',group='native',subgroup='facial_expression' if uid.startswith('Te_') else None,source=media(r['path']),references=[],prompt={'en':r['instruction'],'zh':zh},canvas_wh=[1280,736] if w>=h else [736,1280],fps=12,num_frames=49,timestamps_seconds=[j/12 for j in range(49)],mask_spec=dict(parsing_labels=labels,invert=False,dilation_pixels=8),vace_description=description,original_omni_instruction=r['instruction'],source_review=dict(reviewer='Codex direct visual inspection',sheets='expanded_omni_candidates_04–07,13–14.jpg',full_duration_timestamps=r['probe']['review_timestamps'],notes='Requested person region visible in common window; official instruction retained verbatim.'),task_origin='official'))
        selection.append(dict(id=sid,official=r['official'],probe=r['probe']))
    trips=rows(ROOT/'metadata/trip_candidates.jsonl')
    for candidate in trips[6:]:
        r=candidate['record'];cat=r['references'][0]['attributes']['category'];sid=candidate['id']
        item=dict(id=sid,uid=f'tripvvt:{r["sample_id"]}',dataset='tripvvt',group='tripvvt',subgroup=cat,source=r['source'],references=r['references'],prompt=r['prompt'],canvas_wh=[736,1280],fps=12,num_frames=49,timestamps_seconds=[j/12 for j in range(49)],mask_spec=dict(parsing_labels={'upper_body':[4,7],'lower_body':[5,6],'dresses':[4,5,6,7]}[cat],invert=False,dilation_pixels=12,main_person=True),vace_description='The target person wears the garment in the official reference image. Their identity, actions, other garments, other people, environment and camera are preserved.',task_origin='existing independent test instruction and official garment reference',source_review=dict(reviewer='Codex direct visual inspection',sheets='expanded_trip_candidates_01–04.jpg',full_duration_timestamps=candidate['probe']['review_timestamps'],notes='Garment region visible during common4s; multi-person/mirror clips have source-only actor-box plan.'))
        boxes={'tripvvt_07':[.06,.02,.80,.99],'tripvvt_08':[.38,.02,.98,.99],'tripvvt_17':[.10,.02,.70,.99],'tripvvt_20':[.48,.02,.99,.99],'tripvvt_22':[.40,.02,.92,.99],'tripvvt_23':[.16,.20,.62,.99]}
        if sid in boxes:item['mask_spec']['actor_box']=boxes[sid]
        inputs.append(item);selection.append(dict(id=sid,record=r))
    inputs.sort(key=lambda r: (['identity','hair','background','native','tripvvt'].index(r['group']),r['id']))
    expected=dict(identity=16,hair=16,background=16,native=24,tripvvt=24)
    assert len(inputs)==96 and Counter(r['group'] for r in inputs)==expected
    assert len({r['id'] for r in inputs})==96
    assert Counter(r['subgroup'] for r in inputs if r['group']=='background')==dict(indoor=8,outdoor=8)
    assert Counter(r['subgroup'] for r in inputs if r['group']=='tripvvt')==dict(upper_body=8,lower_body=8,dresses=8)
    assert all(r['probe']['duration_last_pts']>=4 for r in candidates.values() if r['id'] in {i['uid'] for i in inputs})
    jsonl(ROOT/'metadata/inputs.jsonl',inputs);jsonl(ROOT/'metadata/selection.jsonl',selection)
    clusters=defaultdict(list)
    for r in inputs:clusters[r['source']['sha256']].append(r['id'])
    atomic(ROOT/'metadata/source_clusters.json',dict(task_count=96,unique_source_hashes=len(clusters),clusters=dict(clusters),note='Distinct edits may share the same source;96 tasks are not96 independent source videos. Paired trend only, no significance claim.'))
    atomic(ROOT/'metadata/freeze.json',dict(revision='background_removed_fullframe_96_v3',seed=20261112,input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),selection_sha256=sha(ROOT/'metadata/selection.jsonl'),groups=expected,smoke=[],smoke_gate='skipped_by_explicit_user_request',reviewer='Codex direct visual inspection; no VLM API',references_reviewed='reference_background_removal.jpg reviewed; garment reference sheets pending before launch',sampling='Original24 tasks retained with revised identity/hair refs; source-only human candidate inspection; Trip seeded extension8/category',custom_gt=False,unique_source_hashes=len(clusters)))
    print('Frozen96 tasks; unique source hashes',len(clusters),flush=True)

if __name__=='__main__':main()
