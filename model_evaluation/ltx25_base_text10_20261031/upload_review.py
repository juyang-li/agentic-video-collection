from pathlib import Path
import json,hashlib,importlib.util
ROOT=Path(__file__).resolve().parent
receipt=json.loads((ROOT/'review/package_receipt.json').read_text())
archive=Path(receipt['package']);assert archive.is_file()
with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
assert archive.stat().st_size==receipt['bytes']
side=archive.with_suffix('.sha256.json')
if not side.exists():
    d={**receipt,'file':str(archive),'remote_path':'/apps/bypy/lance_review/'+archive.name,'status':'ready_for_upload'}
    side.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('shared_bench_delivery',ROOT.parent/'ltx25_singleturn_v1/package_bench.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.ROOT=ROOT;module.PACKAGE=archive.with_suffix('')
module.upload(receipt_name='baidu_delivery_receipt.json')
