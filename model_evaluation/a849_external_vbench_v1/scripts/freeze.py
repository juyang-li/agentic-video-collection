"""Freeze reviewed sources, instructions and semantic masks before generation."""
from collections import Counter
from common import *

def media(path, **extra):
    return dict(absolute_path=str(path),sha256=sha(path),**extra)

def main():
    require_job()
    candidates={r['id']:r for r in rows(ROOT/'metadata/candidates.jsonl')}
    refs=read(ROOT/'metadata/reference_processing.json')
    refmap={r['name']:r for r in refs}
    inputs=[];selection=[]
    def add(sid,group,official_id,prompt=None,reference=None,labels=None,invert=False,description='',zh='',notes=''):
        r=candidates[official_id]
        inputs.append(dict(id=sid,uid=official_id,dataset='omni',group=group,subgroup=None,
            source=media(r['path']),references=reference or [],
            prompt={'en':prompt or r['instruction'],'zh':zh},
            canvas_wh=[1280,736],fps=12,num_frames=49,timestamps_seconds=[i/12 for i in range(49)],
            mask_spec={'parsing_labels':labels,'invert':invert,'dilation_pixels':8 if group!='hair' else 24},
            vace_description=description,original_omni_instruction=r['instruction'],
            source_review={'reviewer':'Codex direct visual inspection','sheets':'omni_candidates_00–05.jpg','full_duration_timestamps':r['probe']['review_timestamps'],'notes':notes or 'Single main person; requested region visible throughout the four-second edit window.'},
            task_origin='official' if prompt is None else 'custom; no GT'))
        selection.append(dict(id=sid,official=r['official'],probe=r['probe']))
    # Read in fixed order, after visual inspection; no generated outputs exist.
    identity_sources=['Sp_Appearance_change_07','Sp_Clothes_change_07','Sp_Clothes_change_10','Sp_Clothes_change_02']
    for i,(source,name) in enumerate(zip(identity_sources,['trump','obama','biden','harris'])):
        r=refmap[name]
        text='Replace only the main person\'s facial identity with the identity in the face reference image. Preserve the source hairstyle and hair color, expression, gaze, head pose, clothing, body motion, other people, background and camera. Do not copy the reference expression, clothing or background.'
        add(f'identity_{i+1:02d}','identity',source,text,[media(r['face_path'],type='identity',view='front')],[11],description='The main person has the facial identity from the reference, performing the same source actions with the source hairstyle, clothing, expression, background and camera.',zh='仅将主体面部身份替换为脸部参考图中的身份。保留原发型、发色、表情、视线、头部姿态、服装、身体动作、其他人物、背景和镜头；不复制参考图的表情、服装或背景。')
    hair_sources=['Sp_Appearance_change_01','Sp_Clothes_change_04','Sp_Appearance_change_02','Sp_Clothes_change_09']
    for i,(source,name) in enumerate(zip(hair_sources,['harris','trump','obama','biden'])):
        r=refmap[name]
        text='Change only the main person\'s hairstyle to match the single frontal hairstyle reference photograph. Preserve the source hair color, facial identity, expression, head pose, clothing, body motion, background and camera. Use the reference for hair shape and length only; do not copy its face, hair color, clothing or background.'
        add(f'hair_{i+1:02d}','hair',source,text,[media(r['hair_path'],type='hair',view='front')],[2],description='The same main person has the hairstyle shape and length in the frontal reference photograph, retaining their source hair color, face, clothing, action and environment.',zh='仅按单张正面发型照片改变主体头发的形状和长度。保留输入视频中的发色、面部身份、表情、头部姿态、服装、动作、背景和镜头；不复制参考照片的脸、发色、服装或背景。',notes='Hair visible throughout; first source turns away after the evaluation window, retained in full-duration review.')
    bg_sources=['Sp_Clothes_change_01','Sp_Clothes_change_05','Sp_Clothes_change_06','Sp_Appearance_change_05']
    for i,(source,name) in enumerate(zip(bg_sources,['living_room','kitchen','forest','beach'])):
        r=refmap[name];place=name.replace('_',' ')
        text=f'Replace the background with the empty {place} environment shown in the scene reference image. Preserve the main person, face, hairstyle, clothing, body pose, action timing, subject placement and camera motion. Match the person\'s lighting to the new environment. Do not add people from the reference.'
        add(f'background_{i+1:02d}','background',source,text,[media(r['path'],type='scene',view='')],list(range(1,18)),True,description=f'The same source person with unchanged clothing, appearance and action is in the empty {place} environment shown in the reference. Source framing and camera motion are retained.',zh=f'将背景替换为场景参考照片中的无人{dict(living_room="客厅",kitchen="厨房",forest="森林",beach="海滩")[name]}。保留主体人物、脸、发型、服装、姿态、动作时序、主体位置和镜头运动；人物光照与新环境协调，不增添其他人物。')
        inputs[-1]['subgroup']='indoor' if i<2 else 'outdoor'
    natives=[
        ('Sp_Appearance_change_03',[11],'The same elderly woman using a tablet has a youthful facial appearance. Her glasses, clothing, hairstyle, hands, tablet, background and camera are preserved.','让这名女性呈现年轻时的面貌。'),
        ('Sp_Appearance_change_04',[11],'The same woman writing at a table appears younger. Her hairstyle, clothing, action, table, laptop, flowers and camera remain the same.','让这名女性看起来更年轻。'),
        ('Sp_Appearance_change_06',[11],'The same seated woman appears significantly older, with realistic facial wrinkles and age spots. Her clothing, hairstyle, writing motion, seating position and street background remain the same.','让人物显著变老，增添真实的皱纹和老年斑。'),
        ('Sp_Appearance_change_08',[11],'The same man has piercing sapphire blue eyes and no beard. His facial identity, hairstyle, orange shirt, pose and green background remain the same.','将人物眼睛改为明亮的蓝宝石蓝色，并去除全部胡须。'),
        ('Sp_Clothes_change_08',[4,7],'The same woman reading at a table wears a striped shirt in place of the white sweater. Her face, hair, action, book, table and background are preserved.','将这名女性的白毛衣改成条纹衬衫。'),
        ('Sp_Color_change_04',[4],'The same woman outdoors wears a green T-shirt. Her identity, hairstyle, shirt shape, jeans, arm movements, background and camera are preserved.','将这名女性的T恤改为绿色。'),
    ]
    for i,(source,labels,description,zh) in enumerate(natives):
        add(f'native_{i+1:02d}','native',source,labels=labels,description=description,zh=zh)
    for i in range(6):
        sid=f'tripvvt_{i+1:02d}';r=candidates[sid]['record']
        cat=r['references'][0]['attributes']['category']
        item=dict(id=sid,uid=f'tripvvt:{r["sample_id"]}',dataset='tripvvt',group='tripvvt',subgroup=cat,
            source=r['source'],references=r['references'],prompt=r['prompt'],
            canvas_wh=[736,1280],fps=12,num_frames=49,timestamps_seconds=[i/12 for i in range(49)],
            mask_spec={'parsing_labels':{'upper_body':[4,7],'lower_body':[5,6],'dresses':[4,5,6,7]}[cat], 'invert':False,'dilation_pixels':12,'main_person':True},
            vace_description='The target person wears the garment in the official reference image. Their identity, actions, other garments, other people, environment and camera are preserved.',
            task_origin='existing independent test instruction and official garment reference',
            source_review={'reviewer':'Codex direct visual inspection','sheets':'trip_candidates_00–01.jpg','full_duration_timestamps':candidates[sid]['probe']['review_timestamps'],'notes':'Full garment visible. For multi-person clips, target is the foreground/right person; other people must remain unchanged.'})
        # Explicit actor ROI seeds SAM2 tracking, preventing target swaps.
        if i==0:item['mask_spec']['actor_box']=[.05,.02,.70,.98]
        if i==3:item['mask_spec']['actor_box']=[.40,.03,.93,.99]
        inputs.append(item);selection.append(dict(id=sid,record=r))
    assert len(inputs)==24 and Counter(r['group'] for r in inputs)==dict(identity=4,hair=4,background=4,native=6,tripvvt=6)
    assert len({r['source']['absolute_path'] for r in inputs})==24
    # Preserve official TripVVT captions and GT in review-only selection.
    jsonl(ROOT/'metadata/selection.jsonl',selection)
    jsonl(ROOT/'metadata/inputs.jsonl',inputs)
    atomic(ROOT/'metadata/freeze.json',dict(seed=20261112,input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),selection_sha256=sha(ROOT/'metadata/selection.jsonl'),groups=dict(Counter(r['group'] for r in inputs)),smoke=['identity_01','hair_01','background_01','native_06','tripvvt_02'],reviewer='Codex direct image inspection; no Gemini or other VLM API',references_reviewed='references.jpg: four face masks with neutral gray, four single frontal head photos, two empty indoor and two empty outdoor scenes',sampling='TripVVT seeded category shuffle; Omni source-only visual selection from 30 attribute candidates, 18 unique sources; no result-dependent replacement',time_window='Absolute source times 0..4s inclusive, 49/12s encoded playback; no time compression',custom_gt=False))
    print('Frozen',len(inputs),flush=True)

if __name__=='__main__':main()
