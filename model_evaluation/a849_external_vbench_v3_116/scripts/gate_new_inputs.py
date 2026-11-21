"""CPU freeze gate: sources, references, masks, auxiliary prompts; never modifies media."""
from common import *
from contracts import digest

def main():
 items=rows(ROOT/'metadata/inputs.jsonl');review=read(ROOT/'metadata/input_review.json');assert review['status']=='approved' and review['directly_reviewed_cases']==[r['id']for r in items if r['cohort']=='candidate28']
 assert len(items)==116 and len({r['id']for r in items})==116
 assets=[]
 for item in items:
  sid=item['id'];px=ROOT/'inputs/prepared'/sid/'pixels.json';cpath=ROOT/'conditions'/sid/'condition.json';p=read(px);c=read(cpath)
  assert p['input_manifest_sha256']==c['input_manifest_sha256']==sha(ROOT/'metadata/inputs.jsonl') and c['status']=='complete'
  assert sha(item['source']['absolute_path'])==item['source']['sha256']
  assert len(c['mask_frames'])==49 and [sha(x)for x in c['mask_frames']]==c['mask_sha256']
  assert [sha(ROOT/'inputs/prepared'/sid/f'{n:03d}.png')for n in range(49)]==p['frame_sha256']
  assert all(sha(r['absolute_path'])==r['sha256']for r in item['references'])
  assets.append(dict(id=sid,input_digest=digest(item),source_pixels_receipt_sha256=sha(px),condition_receipt_sha256=sha(cpath)))
 atomic(ROOT/'metadata/generation_gate.json',dict(status='approved',manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),direct_new_input_review_sha256=sha(ROOT/'metadata/input_review.json'),assets=assets))
 print('116-case source/reference/mask gate approved',flush=True)
if __name__=='__main__':main()
