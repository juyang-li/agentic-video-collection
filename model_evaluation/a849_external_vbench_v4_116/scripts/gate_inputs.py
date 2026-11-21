"""Hash-bind direct source/reference/mask inspection before any model generation."""
from common import *
from contracts import digest

def main():
    items=rows(ROOT/'metadata/inputs.jsonl');review=read(ROOT/'metadata/input_review.json')
    assert review['status']=='approved' and set(review['cases'])=={r['id']for r in items}
    assets=[];manifest=sha(ROOT/'metadata/inputs.jsonl')
    for item in items:
        sid=item['id'];p=ROOT/'inputs/prepared'/sid/'pixels.json';cpath=ROOT/'conditions'/sid/'condition.json';c=read(cpath)
        assert read(p)['input_manifest_sha256']==c['input_manifest_sha256']==manifest and c['status']=='complete'
        assert review['cases'][sid]['source_pixels_receipt_sha256']==sha(p)
        assert review['cases'][sid]['condition_receipt_sha256']==sha(cpath)
        assert sha(item['source']['absolute_path'])==item['source']['sha256']
        assert [sha(ROOT/'inputs/prepared'/sid/f'{i:03d}.png')for i in range(49)]==read(p)['frame_sha256']
        assert [sha(path)for path in c['mask_frames']]==c['mask_sha256']
        assert all(sha(r['absolute_path'])==r['sha256']for r in item['references'])
        assets.append(dict(id=sid,input_digest=digest(item),source_pixels_receipt_sha256=sha(p),condition_receipt_sha256=sha(cpath)))
    atomic(ROOT/'metadata/generation_gate.json',dict(status='approved',manifest_sha256=manifest,direct_new_input_review_sha256=sha(ROOT/'metadata/input_review.json'),assets=assets))
    print('Generation gate approved',len(items),flush=True)

if __name__=='__main__':main()
