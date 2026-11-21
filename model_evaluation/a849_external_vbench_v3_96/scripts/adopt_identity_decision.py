"""Adopt the corrected, directly reviewed identity rule without mutating live inputs.

If the rule changes the provisional composition, stop before writing anything;
that composition needs its own input-equivalence migration after generation.
"""
import copy,time
from collections import Counter
from common import *
from contracts import identity_decision,MODELS,digest
from score import DIAG


def main():
 review=read(DIAG/'metadata/identity_review.json')
 assert review['inspection_complete'] is True
 diagnostic=rows(DIAG/'metadata/inputs.jsonl')
 assert {r['id'] for r in review['tasks']}=={r['id'] for r in diagnostic}
 for task in review['tasks']:
  for model in MODELS:
   observed=task['models'][model];path=DIAG/'outputs'/model/f"{task['id']}.mp4";receipt=read(path.with_suffix('.json'))
   assert observed['reviewed'] is True and receipt['status']=='complete'
   assert observed['output_sha256']==receipt['output_sha256']==sha(path)
 decision=identity_decision(review['tasks'])
 expected=[r for r in rows(ROOT/'metadata/candidates.jsonl') if r['group']!='hair_color' or int(r['id'].split('_')[-1])<=decision['hair_color_count']]
 if not decision['drop_identity']:expected=diagnostic+expected
 current=rows(ROOT/'metadata/inputs.jsonl')
 assert len(expected)==96
 if [digest(r) for r in current]!=[digest(r) for r in expected]:
  raise RuntimeError('Corrected identity decision changes composition. Preserve live inputs and migrate the final96 after generation; do not mark the provisional cohort validated.')
 frozen=read(ROOT/'metadata/freeze.json');assert frozen['input_manifest_sha256']==sha(ROOT/'metadata/inputs.jsonl')
 assert frozen['groups']==dict(Counter(r['group'] for r in expected))
 corrected_freeze=read(DIAG/'metadata/freeze.json');assert corrected_freeze['input_manifest_sha256']==sha(DIAG/'metadata/inputs.jsonl')
 input_review=read(DIAG/'metadata/input_review.json');assert input_review['status']=='approved'
 refs=copy.deepcopy(corrected_freeze['reference_processing'])
 import numpy as np
 from PIL import Image
 for ref in refs:
  raw=np.asarray(Image.open(ref['path']).convert('RGB'));actual=np.asarray(Image.open(ref['processed_path']).convert('RGB'));mask=np.asarray(Image.open(ref['foreground_mask']))>127
  assert raw.shape==actual.shape==(1024,1024,3) and not mask[0].any()
  assert np.array_equal(raw[mask],actual[mask]) and np.all(actual[~mask]==128)
  assert sha(ref['processed_path'])==ref['processed_sha256']
  ref.update(role='identity',name=Path(ref['path']).stem,visual_review='Approved by direct original and segmented full-head reference inspection; see corrected diagnostic input_review.json')
 old_refs=ROOT/'metadata/reference_processing.json';archive=ROOT/'metadata/initial_framing_reference_processing.json'
 if not archive.exists():atomic(archive,read(old_refs))
 refs += [r for r in read(old_refs) if r.get('role')=='hair']
 atomic(old_refs,refs)
 frozen.update(identity_decision=decision,identity_decision_status='validated_fullhead_retest',diagnostic_manifest_sha256=sha(DIAG/'metadata/inputs.jsonl'),identity_review_sha256=sha(DIAG/'metadata/identity_review.json'),identity_diagnostic_root=str(DIAG))
 atomic(ROOT/'metadata/freeze.json',frozen)
 atomic(ROOT/'metadata/identity_diagnostic.json',dict(review,diagnostic_root=str(DIAG)))
 correction=read(ROOT/'metadata/identity_protocol_correction.json')
 correction.update(status='validated',validated_unix=time.time(),decision=decision,formal_diagnostic=str(DIAG),formal_review_sha256=sha(DIAG/'metadata/identity_review.json'),main_manifest_unchanged=True,main_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),initial_diagnostic='Preserved as historical framing diagnostic; cropped-reference tasks02–04 excluded from formal identity decision',ordering_deviation='The initial framing error was discovered during main generation; unchanged non-identity tasks continued, and the formal full-head retest controls the final cohort decision.')
 atomic(ROOT/'metadata/identity_protocol_correction.json',correction)
 print('Adopted corrected identity decision',decision,flush=True)
if __name__=='__main__':main()
