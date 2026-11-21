"""CPU pre-generation freeze of every source, instruction, reference and actual mask."""
import argparse
from common import *
from contracts import digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--approve',action='store_true');a=p.parse_args()
 import numpy as np
 from PIL import Image
 items=rows(ROOT/'metadata/inputs.jsonl');manifest=sha(ROOT/'metadata/inputs.jsonl')
 assert read(ROOT/'metadata/freeze.json')['input_manifest_sha256']==manifest
 overlap=read(ROOT/'metadata/training_overlap.json');assert overlap['status']=='complete' and not overlap['source_hits']
 review=read(ROOT/'metadata/input_visual_review.json');assert review['inspection_complete'] is True and review['manifest_sha256']==manifest
 assert set(review['samples'])=={r['id'] for r in items}
 inventory=[]
 for item in items:
  sid=item['id'];assert review['samples'][sid]['source_and_reference_reviewed'] and review['samples'][sid]['mask_reviewed']
  for r in [item['source'],*item['references']]:assert sha(r['absolute_path'])==r['sha256']
  if item['group']=='hair_color':
   assert item['axis']=='hair_color' and item['mode']=='attribute' and item['references']==[]
   assert item['prompt']['en'].startswith('Using the provided source video, apply the following edit: Please change ')
  prepared=ROOT/'inputs/prepared'/sid;c=read(ROOT/'conditions'/sid/'condition.json');px=read(prepared/'pixels.json')
  assert c['status']=='complete' and c['input_manifest_sha256']==px['input_manifest_sha256']==manifest
  assert px['source_sha256']==item['source']['sha256']
  assert [sha(prepared/f'{i:03d}.png') for i in range(49)]==px['frame_sha256']
  assert len(c['mask_frames'])==49 and [sha(p) for p in c['mask_frames']]==c['mask_sha256']
  derived='A continuous video. '+item['vace_description']+'\nOriginal edit specification: '+item['prompt']['en']
  overrides=read(ROOT/'metadata/vace_prompt_corrections.json')
  if sid in overrides:
   override=overrides[sid]
   assert sid=='native_03' and c['adapted_prompt']==override['actual_adapted_prompt']
   assert sha(override['original_v2_condition'])==override['original_v2_condition_sha256']
  else:assert c['adapted_prompt']==derived
  inventory.append(dict(id=sid,input_digest=digest(item),source_pixels_receipt_sha256=sha(prepared/'pixels.json'),condition_receipt_sha256=sha(ROOT/'conditions'/sid/'condition.json'),mask_sha256=c['mask_sha256'],reference_sha256=[r['sha256'] for r in item['references']]))
 for ref in read(ROOT/'metadata/reference_processing.json'):
  original=np.asarray(Image.open(ref['path']).convert('RGB'));actual=np.asarray(Image.open(ref['processed_path']).convert('RGB'));mask=np.asarray(Image.open(ref['foreground_mask']))>127
  assert original.shape==actual.shape and np.all(actual[~mask]==128) and np.array_equal(actual[mask],original[mask])
  assert sha(ref['processed_path'])==ref['processed_sha256']
 record=dict(status='approved' if a.approve else 'verified',manifest_sha256=manifest,assets=inventory,direct_visual_review_sha256=sha(ROOT/'metadata/input_visual_review.json'),training_overlap_sha256=sha(ROOT/'metadata/training_overlap.json'),policy='All source/prompt/ref/mask inputs frozen; no VLM API; no output-based task selection except prespecified identity diagnostic rule.')
 path=ROOT/'metadata/generation_gate.json'
 if path.exists():assert read(path)==record,'Frozen asset inventory changed'
 else:atomic(path,record)
 print(record['status'],len(items),'inputs',flush=True)
if __name__=='__main__':main()
