"""Reuse immutable blobs with new provenance receipts; never edit parent artifacts."""
import shutil
from common import *
def link(src,dst):
 src=Path(src).resolve();dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists(): assert dst.resolve()==src
 else: dst.symlink_to(src)
def transplant(old,old_id,item,root=ROOT,roles=('source','text'),conditions=True):
 new_id=item['id'];manifest=sha(root/'metadata/inputs.jsonl')
 previous=next(r for r in rows(old/'metadata/inputs.jsonl') if r['id']==old_id)
 assert item['source']==previous['source'] and item['canvas_wh']==previous['canvas_wh']
 assert item['timestamps_seconds']==previous['timestamps_seconds']
 if 'text' in roles: assert item['prompt']['en']==previous['prompt']['en']
 src=old/'inputs/prepared'/old_id;dst=root/'inputs/prepared'/new_id;dst.mkdir(parents=True,exist_ok=True)
 for x in src.iterdir():
  if x.name!='pixels.json':link(x,dst/x.name)
 r=read(src/'pixels.json');r.update(input_manifest_sha256=manifest,reused_from_receipt=str(src/'pixels.json'),reused_from_sha256=sha(src/'pixels.json'))
 atomic(dst/'pixels.json',r)
 for role in roles:
  src=old/'cache'/old_id/f'{role}.pt';dst=root/'cache'/new_id/f'{role}.pt'
  if not src.exists():continue
  if role.startswith('ref_'): assert item['references']==previous['references']
  link(src,dst);r=read(src.with_suffix('.json'));assert r['sha256']==sha(src)
  r['reused_from_receipt']=str(src.with_suffix('.json'));r['reused_from_sha256']=sha(src.with_suffix('.json'))
  r['identity']['manifest_sha256']=manifest;atomic(dst.with_suffix('.json'),r)
 if conditions:
  assert item['mask_spec']==previous['mask_spec']
  src=old/'conditions'/old_id;dst=root/'conditions'/new_id;dst.mkdir(parents=True,exist_ok=True)
  r=read(src/'condition.json');assert r['status']=='complete'
  for path,h in zip(r['mask_frames'],r['mask_sha256'],strict=True):assert sha(path)==h
  r.update(id=new_id,input_manifest_sha256=manifest,source_pixels_sha256=sha(root/'inputs/prepared'/new_id/'pixels.json'),adapted_prompt='A continuous video. '+item['vace_description']+'\nOriginal edit specification: '+item['prompt']['en'],reused_from_receipt=str(src/'condition.json'),reused_from_sha256=sha(src/'condition.json'),human_protocol_review='pending')
  atomic(dst/'condition.json',r)
