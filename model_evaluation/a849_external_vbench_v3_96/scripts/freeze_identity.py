"""CPU: freeze four directly reviewed identity retests separately from the cohort."""
import copy,sys,shutil
from common import *
from reuse import transplant

def main():
 old=ROOT.parent/'a849_external_vbench_v2_96';diag=ROOT.parent/'a849_external_vbench_v3_identity'
 previous={r['id']:r for r in rows(old/'metadata/inputs.jsonl')}
 refs=read(ROOT/'metadata/reference_processing.json');records=[]
 notes=['Single near-frontal man, face large and unobstructed.', 'Woman in pale blazer; short source blonde hair retained.', 'Woman at laptop, hand occasionally near mouth; preserve gestures.', 'Woman in red dress; long dark hair and environment retained.']
 for i,(sid,ref,note) in enumerate(zip(['identity_01','identity_06','identity_13','identity_14'],refs,notes,strict=True),1):
  r=copy.deepcopy(previous[sid]);r.update(id=f'identity_{i:02d}',old_id=sid,revision_reason='1024 CelebA-HQ near-frontal identity reference; natural hair; foreground unchanged; background RGB128')
  r['references']=[dict(absolute_path=ref['processed_path'],sha256=ref['processed_sha256'],type='identity',view='front')]
  r['source_review']=dict(reviewer='Codex direct visual inspection, no VLM API',notes=note,sheets=['identity_source_candidates.jpg','identity_input_temporal.jpg'],review_timestamps=[0,1,2,3,4])
  r['task_origin']='self_created_reference_edit';records.append(r)
  ref['visual_review']='Codex inspected original/segmented reference sheet; face and natural hair retained; neutral background; 2026-11-14'
 atomic(ROOT/'metadata/reference_processing.json',refs)
 jsonl(diag/'metadata/inputs.jsonl',records)
 atomic(diag/'metadata/freeze.json',dict(revision='v3_identity_natural_hair',input_manifest_sha256=sha(diag/'metadata/inputs.jsonl'),smoke=[],groups={'identity':4},protocol='same v2 inference settings; 49 frames / 12fps / 0..4s',decision_rule='Drop identity from main only if at least 3/4 tasks lack clear identity transfer in ALL THREE models; missing outputs do not count as failures.'))
 for r in records:transplant(old,r['old_id'],r,diag)
 atomic(diag/'metadata/input_review.json',dict(status='pending_temporal_mask_review',references=refs,notes=notes))
 from PIL import Image,ImageOps,ImageDraw
 sheet=Image.new('RGB',(1500,4*340),'#20242a');d=ImageDraw.Draw(sheet)
 for j,r in enumerate(records):
  for k,f in enumerate([0,12,24,36,48]):
   frame=Image.open(diag/'inputs/prepared'/r['id']/f'{f:03d}.png');mask=Image.open(read(diag/'conditions'/r['id']/'condition.json')['mask_frames'][f]).convert('L')
   for row,img in enumerate([frame,mask]):
    x=k*300;y=j*340+row*160;sheet.paste(ImageOps.contain(img,(295,135)),(x,y+23));d.text((x+2,y+2),f"{r['id']} {f/12:g}s {'source' if row==0 else 'face mask'}",fill='white')
 sheet.save(ROOT/'review/identity_input_temporal.jpg',quality=95)
 print(diag)
if __name__=='__main__':main()
