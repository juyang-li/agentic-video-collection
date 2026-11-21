"""Freeze corrected full-head references; keep the original diagnostic untouched."""
import copy,shutil,os,signal
from common import *
from reuse import transplant,link
OLD=ROOT.parent/'a849_external_vbench_v3_identity'
DEST=ROOT.parent/'a849_external_vbench_v3_identity_fullhead'

def main():
 for folder in ['scripts','metadata','logs','inputs/prepared','outputs/ltx_base','outputs/a849','outputs/vace','cache','conditions','review']:(DEST/folder).mkdir(parents=True,exist_ok=True)
 for name in ['common.py','native_worker.py','native_cache.py','single_front_adapter.py','media.py']:
  shutil.copy2(OLD/'scripts'/name,DEST/'scripts'/name)
 shutil.copy2(ROOT/'scripts/vace_worker.py',DEST/'scripts/vace_worker.py')
 shutil.copy2(ROOT/'scripts/reuse.py',DEST/'scripts/reuse.py')
 shutil.copy2(ROOT/'.gitignore',DEST/'.gitignore')
 refs=[next(r for r in read(ROOT/'metadata/reference_processing.json') if Path(r['path']).stem=='00011')]+read(ROOT/'metadata/fullhead_reference_processing.json')
 assert [Path(r['path']).stem for r in refs]==['00011','00009','00017','00023']
 import numpy as np
 from PIL import Image
 for ref in refs:
  raw=np.asarray(Image.open(ref['path']).convert('RGB'));actual=np.asarray(Image.open(ref['processed_path']).convert('RGB'));mask=np.asarray(Image.open(ref['foreground_mask']))>127
  assert raw.shape==actual.shape==(1024,1024,3) and not mask[0].any()
  assert np.array_equal(raw[mask],actual[mask]) and np.all(actual[~mask]==128)
  ref['visual_review']='pending corrected full-head reference sheet inspection'
 records=rows(OLD/'metadata/inputs.jsonl')
 for item,ref in zip(records,refs,strict=True):
  item['references']=[dict(absolute_path=ref['processed_path'],sha256=ref['processed_sha256'],type='identity',view='front')]
  item['revision_reason']='Corrected complete-head1024 reference, natural hair and original foreground pixels retained; original cropped-reference diagnostic archived'
 jsonl(DEST/'metadata/inputs.jsonl',records)
 atomic(DEST/'metadata/freeze.json',dict(revision='v3_identity_fullhead_corrected',input_manifest_sha256=sha(DEST/'metadata/inputs.jsonl'),smoke=[],groups={'identity':4},decision_rule='At least3 tasks must lack clear reference facial identity transfer in ALL3 models to drop Identity from main.',reference_processing=refs,original_diagnostic=str(OLD),protocol='Same three-model parameters, source sampling, source-only face mask, seed42,49frames/12fps'))
 for item in records:
  roles=['source','text']+(['ref_0'] if item['id']=='identity_01' else [])
  transplant(OLD,item['id'],item,DEST,roles=roles)
 reused=[]
 for model in ['ltx_base','a849','vace']:
  src=OLD/'outputs'/model/'identity_01.mp4';receipt=src.with_suffix('.json');r=read(receipt);assert r['status']=='complete' and sha(src)==r['output_sha256']
  link(src,DEST/'outputs'/model/src.name);link(receipt,DEST/'outputs'/model/receipt.name)
  reused.append(dict(id='identity_01',model=model,original_receipt=str(receipt),original_receipt_sha256=sha(receipt),output_sha256=r['output_sha256'],reason='Original reference00011 already meets full-head criterion; exact source/prompt/ref/mask and inference protocol unchanged.'))
 atomic(DEST/'metadata/reuse_outputs.json',dict(reused=reused,new_manifest_sha256=sha(DEST/'metadata/inputs.jsonl')))
 atomic(DEST/'metadata/input_review.json',dict(status='pending',reference_sheet=str(ROOT/'review/fullhead_reference_processing.jpg'),reference_records=refs,source_and_mask='Exact same four reviewed source windows/face masks as original diagnostic; raw frame hashes preserved.'))
 atomic(ROOT/'metadata/identity_diagnostic_location.json',dict(root=str(DEST),supersedes=str(OLD),reason='Original tasks02–04 references had cropped crowns; original outputs retained as framing diagnostics.'))
 print('Corrected diagnostic frozen;3 exact outputs reused,9 require regeneration',flush=True)
if __name__=='__main__':main()
