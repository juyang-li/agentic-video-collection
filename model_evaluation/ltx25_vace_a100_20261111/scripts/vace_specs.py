"""Adaptation uses only original instruction and existing legal references."""
from common import *
# Each tuple is a semantic region, tracked independently and then unioned.
# Explicit insertion boxes are source-normalized planned ROIs, to be visually checked.
COINVE={
 'coinve_01':{'queries':[],'manual_objects':[[.36,.36,.65,1.0],[.32,0,.73,.28]],'description':'A woman in the same kitchen, without the red baking apron. The range hood is modern and black. Her identity, actions and the rest of the kitchen remain the same.'},
 'coinve_02':{'queries':['basketball jersey','neck'],'description':'The same man wears a red Chicago Bulls basketball jersey and a thick silver chain necklace. His face, actions, scene and camera remain the same.'},
 'coinve_03':{'queries':['infotainment touchscreen','dashboard'],'description':'The same car interior has black over-ear headphones resting on top of its central infotainment touchscreen. The dashboard panel above the center console has a matte grey carbon fiber texture. All other details and camera movement remain the same.'},
 'coinve_04':{'queries':['red lanyard'],'background':True,'description':'The same man, without a red lanyard, appears against a blurred outdoor park with green trees and sunlight. His identity, clothing, action and the camera remain the same.'},
 'coinve_05':{'queries':['neck'],'background':True,'description':'The same woman wearing a purple cardigan has a red scarf around her neck, with a blurred green garden behind her. Her face, actions and the camera remain the same.'},
 'coinve_06':{'queries':[],'background':True,'protect':'purple character . table','boxes':[[.03,.45,.45,.9]],'description':'The same purple character and wooden table, with a large red apple sitting on the table to the left of the character. The background is a bright blue sky with white clouds. The character, its actions and the camera remain the same.'},
 'coinve_07':{'queries':['goggles'],'background':True,'description':'The same person, without the blue goggles and neon yellow strap, against a blurred outdoor forest with green trees and dappled sunlight. The person, clothing, action and camera remain the same.'},
 'coinve_08':{'queries':['yellow shirt','blue neon sign'],'boxes':[[0,.15,.35,.95]],'description':'The same man wears his black jacket without the yellow shirt underneath. A large potted green plant stands by the dark background wall on the left. A warm-toned vintage clock face occupies the former blue neon sign position. His identity, action and camera remain the same.'},
 'coinve_09':{'queries':['blouse','neck'],'background':True,'description':'The same female news anchor wears a solid navy blue blazer and a silver pendant necklace. The background is a blurred city street at night. Her identity, actions and camera remain the same.'},
 'coinve_10':{'queries':['white shirt on the woman on the left','baby head'],'background':True,'description':'The same mother on the left wears a red floral dress. The baby has a large yellow bow on top of its head. Behind them is a blurred sunlit living room with furniture. Their faces, actions and the camera remain the same.'},
}

def spec(item):
    g=item['group'];original=item['prompt']['en']
    base={'original_instruction':original,'prompt_provenance':'source context named in original instruction plus existing references; no GT video or official target caption read','queries':[],'dilation_pixels':12,'background':False,'boxes':[],'protect':'person'}
    target={
      'identity':'The person has the facial identity shown in the identity reference, with consistent skin tone on all exposed skin. The source hairstyle, hair color, clothing, expression, action, environment and camera are preserved.',
      'whole_outfit':'The same person wears the complete outfit shown in the front and back reference images. The face, hair, motion, environment and camera match the source video.',
      'hair':'The same person has the hairstyle shown in the front and back references, while retaining the source face, hair color, clothing, action, scene and camera.',
      'site':'The same person, clothing and action appear in the environment in the scene reference image. Lighting on the person is consistent with the new environment. The source framing and camera movement are preserved.',
      'action':'The same person with the source appearance and clothing performs the action shown by the conditioning pose sequence. The source scene and camera are preserved.',
      'camera':'The person and scene retain their source appearance and action. The starting viewing direction, subject placement and framing match the camera reference image. The camera motion is described by the original camera instruction below; this is its first four seconds.',
      'tripvvt':'The same person wears the garment in the official garment reference image. Their identity, motion, other clothing, other people, background and camera are preserved.',
    }
    if g=='coinve':base.update(COINVE[item['id']]);description=base.pop('description')
    elif g=='single_piece':
        garment={'shirts':'top','full_body':'one-piece garment','skirts':'skirt','pants':'trousers'}[item['subgroup']]
        base['queries']=[{'shirts':'shirt . top','full_body':'dress . jumpsuit','skirts':'skirt','pants':'pants . trousers'}[item['subgroup']]]
        base.update(queries=[],parsing_labels={'shirts':[4,7,14,15],'full_body':[4,5,6,7,12,13,14,15],'skirts':[5,6,12,13],'pants':[5,6,12,13]}[item['subgroup']],mask_backend='segformer_garment_then_sam2')
        description=f'The same person wears the {garment} in the front and back reference images. Only this garment changes. The remaining clothing, person, movement, scene and camera are preserved.'
    elif g=='hair_color':
        text=original.split('apply the following edit:')[-1].strip()
        color=text.split('color to ')[-1].rstrip('.')
        description=f'The same person has {color} hair. Their hairstyle, face, clothing, movement, scene and camera match the source video.'
        base['queries']=['hair']
    else:description=target[g]
    if g=='identity':base.update(queries=[],parsing_labels=[11,12,13,14,15],mask_backend='segformer_skin_then_sam2',dilation_pixels=6)
    if g=='whole_outfit':base.update(queries=[],parsing_labels=[4,5,6,7,8,9,10,12,13,14,15],mask_backend='segformer_clothes_limbs_shoes_then_sam2')
    if g in ('hair','hair_color'):base.update(queries=[],parsing_labels=[2],mask_backend='segformer_hair_skin_protected_then_sam2',protect_source_face_neck=True,dilation_pixels=32 if g=='hair' else 6)
    if g=='site':base.update(background=True,queries=[],parsing_labels=list(range(1,18)),parsing_invert=True,mask_backend='segformer_person_inverse_then_sam2')
    if g=='tripvvt':base.update(queries=[],parsing_labels={'upper_body':[4,14,15],'lower_body':[5,6,12,13],'dresses':[4,5,6,7,12,13,14,15]}[item['subgroup']],mask_backend='segformer_garment_then_sam2',main_person=True)
    if item['id']=='tripvvt_01':base['parsing_labels']=[4]
    if g=='camera':base['full_frame']=True
    if g=='action':base['pose']=True
    if g=='hair':
        color_catalog=Path('/scratch/prj0000000275/jcy/xks/DataMake/00_engine/outputs/dataset_package_v1/dataset_v1/catalogs/hair_color.json')
        color_id=item['source']['attributes']['hair_color_id']
        color=next(v['code'] for v in read(color_catalog)['items'] if v['id']==color_id).replace('_',' ')
        description+=f' The source hair color is {color}; the edited hair remains {color}. Do not copy the reference hair color.'
        base['source_color_provenance']={'source_hair_color_id':color_id,'color':color,'catalog':str(color_catalog),'catalog_sha256':sha(color_catalog),'role':'source-visible preservation attribute only; no GT or target attribute read'}
    base['adapted_prompt']='A continuous video. '+description+'\nOriginal edit specification: '+original
    return base

if __name__=='__main__':
    jsonl(ROOT/'metadata/vace_specs_v8.jsonl',[{'id':i['id'],**spec(i)} for i in rows(ROOT/'metadata/inputs.jsonl')])
