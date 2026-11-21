"""CPU pixel audit of all Hair/Identity references; preserve originals and gray128 backgrounds."""
import time
from common import *

def main():
 import numpy as np
 from PIL import Image
 used={Path(r['absolute_path']).resolve():r['sha256']for item in rows(ROOT/'metadata/inputs.jsonl')if item['group']in['hair','identity']for r in item['references']};processing={Path(r['processed_path']).resolve():r for r in read(ROOT/'metadata/reference_processing.json')};results=[]
 for path,h in used.items():
  r=processing[path];original=Path(r['path']);maskpath=Path(r['foreground_mask']);assert sha(path)==h==r['processed_sha256'];assert sha(original)==r['sha256'];assert sha(maskpath)==r['foreground_mask_sha256']
  a=np.asarray(Image.open(original).convert('RGB'));b=np.asarray(Image.open(path).convert('RGB'));mask=np.asarray(Image.open(maskpath).convert('L'));assert a.shape==b.shape and mask.shape==a.shape[:2]and set(np.unique(mask))<={0,255};foreground=mask==255;assert foreground.any()and(~foreground).any()
  assert np.array_equal(a[foreground],b[foreground]);assert np.all(b[~foreground]==128)
  results.append(dict(path=str(path),processed_sha256=h,original_sha256=r['sha256'],foreground_mask_sha256=r['foreground_mask_sha256'],canvas_wh=[a.shape[1],a.shape[0]],foreground_bitwise_original=True,background_rgb=[128,128,128],foreground_pixels=int(foreground.sum()),no_crop_or_resize=True))
 assert len(results)==len(used)
 atomic(ROOT/'metadata/reference_pixel_audit.json',dict(status='passed',unique_references=len(results),manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),references=results,completed_unix=time.time(),note='Pixel audit does not establish segmentation completeness; use direct full-head input review and recorded limitations.'))
 print('All current Hair/Identity references: original foreground pixels and RGB128 background verified.',flush=True)
if __name__=='__main__':main()
