"""Data/documents ZIP only; validation must match the frozen manifest."""
from build import *
import zipfile

def main():
 p=ROOT/'package';v=read(p/'validation.json');assert v['status']=='passed' and v['input_manifest_sha256']==sha(p/'inputs.jsonl')
 allowed={'README.md','inputs.jsonl','ground_truth.jsonl','media_inventory.jsonl','sampling.json','validation.json','model_interfaces.json','result_schema.json','index.html','review_manifest.json'}
 assert {f.name for f in p.iterdir()}==allowed|{'media','review_media'}
 dst=ROOT/'frozen300_20261113.zip';tmp=dst.with_suffix('.part.zip')
 review=read(p/'review_manifest.json')
 paths=set(allowed)|{m['path'] for m in rows(p/'media_inventory.jsonl')}
 paths.update(x[k] for x in review['clips'] for k in ['path','poster'])
 files=sorted(p/rel for rel in paths)
 assert len(rows(p/'inputs.jsonl'))==300 and len(rows(p/'ground_truth.jsonl'))==300
 with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
  for f in files:
   assert not f.is_symlink()
   z.write(f,'frozen300_20261113/'+f.relative_to(p).as_posix())
 with zipfile.ZipFile(tmp) as z:
  assert len(z.infolist())==len(files) and z.testzip() is None
 tmp.replace(dst)
 write(ROOT/'metadata/package_receipt.json',{'status':'complete','zip':str(dst),'bytes':dst.stat().st_size,'files':len(files),'input_manifest_sha256':v['input_manifest_sha256'],'local_zip_crc_check':'passed','contains_weights_environment_inference_scripts':False})
 print('PACKAGE_COMPLETE',dst,dst.stat().st_size,flush=True)
if __name__=='__main__':main()
