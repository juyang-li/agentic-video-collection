"""Reuse the established uploader; verify listing only, without remote readback."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('existing_delivery',ROOT.parent/'recent_motion_review_20261030/deliver.py')
delivery=importlib.util.module_from_spec(spec);spec.loader.exec_module(delivery)
delivery.UPLOADER=ROOT/'upload_parallel.py'
ZIP=ROOT/'review/datamake_v1_balanced80_scroll_20261030.zip'

def main():
    side=ZIP.with_suffix('.sha256.json');d=json.loads(side.read_text());token=delivery.load_token(delivery.CONFIG)
    result=delivery.upload(ZIP,d['remote_path'])
    rows=[x for x in delivery.listing(token) if x.get('path')==d['remote_path']]
    assert len(rows)==1 and rows[0]['size']==d['bytes'] and rows[0]['fs_id']==result['remote_result']['fs_id']
    d.update(status='uploaded',remote_listing_verified=True,remote_fs_id=rows[0]['fs_id'],
             remote_full_download_verification=False,verified_at_utc=datetime.now(timezone.utc).isoformat(),
             verification_scope='upload completion and remote listing size/fs_id; no remote whole-file readback')
    side.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    remote_side=str(Path(d['remote_path']).with_suffix('.sha256.json'));delivery.upload(side,remote_side)
    assert any(x.get('path')==remote_side and x.get('size')==side.stat().st_size for x in delivery.listing(token))
    receipt={**d,'sidecar_remote_path':remote_side,'rating_controls':0,'groups':8,'per_group':10}
    (ROOT/'metadata/delivery_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
