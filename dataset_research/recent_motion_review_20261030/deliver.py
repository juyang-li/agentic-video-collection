"""Upload the scroll package; intentionally no remote whole-file readback."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
import requests

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'lance_general_edit_finetune/1_specialist_filtering/scripts'))
from upload_baidu_xpan import checked_json,load_token
ZIP=ROOT/'review/recent_motion_datasets_scroll50_20261030.zip'
CONFIG=ROOT.parent/'ref_edit_review_20261030/metadata/.baidu_session.json'
UPLOADER=PROJECT/'lance_general_edit_finetune/1_specialist_filtering/scripts/upload_baidu_xpan.py'


def listing(token):
    out=[]
    for start in range(0,10000,1000):
        try:r=requests.get('https://pan.baidu.com/rest/2.0/xpan/file',params={'method':'list','access_token':token,'dir':'/apps/bypy/lance_review','start':start,'limit':1000},timeout=(20,60))
        except requests.RequestException as e:raise RuntimeError('Baidu listing: '+type(e).__name__) from None
        rows=checked_json(r,'Baidu listing').get('list',[]);out.extend(rows)
        if len(rows)<1000:return out
    raise RuntimeError('pagination bound exceeded')


def upload(path,remote):
    state=path.with_suffix('.baidu_upload.json')
    if state.exists():
        d=json.loads(state.read_text())
        if d.get('status')=='complete' and d.get('size')==path.stat().st_size and d.get('mtime_ns')==path.stat().st_mtime_ns:return d
    subprocess.run([sys.executable,str(UPLOADER),'--local',str(path.resolve()),'--remote',remote,'--state',str(state),'--bypy-config',str(CONFIG)],check=True)
    return json.loads(state.read_text())


def main():
    side=ZIP.with_suffix('.sha256.json');d=json.loads(side.read_text());token=load_token(CONFIG)
    if any(x.get('path')==d['remote_path'] for x in listing(token)) and not ZIP.with_suffix('.baidu_upload.json').exists():
        d['remote_path']=str(Path(d['remote_path']).with_name(ZIP.stem+'_'+d['sha256'][:8]+'.zip'))
    result=upload(ZIP,d['remote_path'])
    rows=[x for x in listing(token) if x.get('path')==d['remote_path']]
    assert len(rows)==1 and rows[0]['size']==d['bytes'] and rows[0]['fs_id']==result['remote_result']['fs_id']
    d.update(status='uploaded',remote_listing_verified=True,remote_fs_id=rows[0]['fs_id'],remote_full_download_verification=False,
             verification_scope='upload completion + remote listing size/fs_id; no remote whole-file download at user request',verified_at_utc=datetime.now(timezone.utc).isoformat())
    side.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    remote_side=str(Path(d['remote_path']).with_suffix('.sha256.json'));upload(side,remote_side)
    assert any(x.get('path')==remote_side and x.get('size')==side.stat().st_size for x in listing(token))
    receipt={**d,'sidecar_remote_path':remote_side,'rating_controls':0,'group_counts':{'X-Dance':12,'RealisDance-Val':24,'MoveBench-human':11,'Wan-official-examples':3}}
    (ROOT/'metadata/delivery_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
