#!/usr/bin/env python3
"""Upload the validated ZIP and verify the exact remote content when downloadable."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import requests

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'lance_general_edit_finetune/1_specialist_filtering/scripts'))
from upload_baidu_xpan import checked_json,load_token

NAME='ref_edit_clothing25_motion25_review_20261030'
ARCHIVE=ROOT/'review'/f'{NAME}.zip'
CONFIG=ROOT/'metadata/.baidu_session.json'
UPLOADER=PROJECT/'lance_general_edit_finetune/1_specialist_filtering/scripts/upload_baidu_xpan.py'


def save(path,d):path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def upload(local,remote):
    state=local.with_suffix('.baidu_upload.json')
    if state.exists():
        old=json.loads(state.read_text())
        if old.get('status')=='complete' and old.get('mtime_ns')==local.stat().st_mtime_ns and old.get('size')==local.stat().st_size:
            return old
    subprocess.run([sys.executable,str(UPLOADER),'--local',str(local.resolve()),'--remote',remote,
                    '--state',str(state),'--bypy-config',str(CONFIG)],check=True)
    return json.loads(state.read_text())


def listing(token):
    out=[]
    for offset in range(0,10000,1000):
        r=requests.get('https://pan.baidu.com/rest/2.0/xpan/file',params={'method':'list','access_token':token,
            'dir':'/apps/bypy/lance_review','start':offset,'limit':1000},timeout=(20,60))
        rows=checked_json(r,'verify remote listing').get('list',[]);out.extend(rows)
        if len(rows)<1000:return out
    raise RuntimeError('remote listing pagination exceeded bound')


def downloaded_sha(token,fsid,expected_bytes):
    r=requests.get('https://pan.baidu.com/rest/2.0/xpan/multimedia',params={
        'method':'filemetas','access_token':token,'fsids':json.dumps([fsid]),'dlink':1},timeout=(20,60))
    rows=checked_json(r,'read download metadata').get('list',[])
    if len(rows)!=1 or not rows[0].get('dlink'):raise RuntimeError('no authorized remote download link')
    try:
        response=requests.get(rows[0]['dlink'],params={'access_token':token},
                              headers={'User-Agent':'pan.baidu.com'},stream=True,timeout=(20,90))
        if response.status_code!=200:raise RuntimeError(f'remote download HTTP {response.status_code}')
        h=hashlib.sha256();size=0
        with response:
            for block in response.iter_content(1024*1024):h.update(block);size+=len(block)
        if size!=expected_bytes:raise RuntimeError('remote download byte count mismatch')
        return h.hexdigest()
    except requests.RequestException as e:
        raise RuntimeError('remote download network failure: '+type(e).__name__) from None


def main():
    sidecar=ARCHIVE.with_suffix('.sha256.json');receipt=json.loads(sidecar.read_text())
    assert ARCHIVE.stat().st_size==receipt['bytes']
    with ARCHIVE.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
    token=load_token(CONFIG);remote=receipt['remote_path']
    occupied=[x for x in listing(token) if x.get('path')==remote]
    if occupied and not ARCHIVE.with_suffix('.baidu_upload.json').exists():
        remote=str(Path(remote).with_name(ARCHIVE.stem+'_'+receipt['sha256'][:8]+'.zip'))
        assert not any(x.get('path')==remote for x in listing(token)),'unexpected existing remote release'
        receipt['remote_path']=remote;save(sidecar,receipt)
    up=upload(ARCHIVE,remote)
    found=[x for x in listing(token) if x.get('path')==remote]
    assert len(found)==1 and found[0]['size']==receipt['bytes'] and found[0]['fs_id']==up['remote_result']['fs_id']
    receipt.update(upload_status='complete',remote_listing_verified=True,remote_bytes=found[0]['size'],
                   remote_fs_id=found[0]['fs_id'],verified_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        digest=downloaded_sha(token,found[0]['fs_id'],receipt['bytes'])
        assert digest==receipt['sha256'],'remote ZIP SHA256 mismatch'
        receipt.update(remote_download_sha256_verified=True,remote_download_sha256=digest,
                       sha256_scope='local ZIP and complete independent remote download')
        print('Full remote ZIP SHA256 verified',flush=True)
    except RuntimeError as e:
        receipt.update(remote_download_sha256_verified=False,remote_download_verification_note=str(e),
                       sha256_scope='local ZIP; all upload block MD5s and fresh remote listing verified')
        print('Remote download check unavailable; block MD5 and listing verified: '+str(e),flush=True)
    receipt['sidecar_remote_path']=str(Path(remote).with_suffix('.sha256.json'))
    save(sidecar,receipt)
    sideup=upload(sidecar,receipt['sidecar_remote_path'])
    final=listing(token)
    side=[x for x in final if x.get('path')==receipt['sidecar_remote_path']]
    assert len(side)==1 and side[0]['size']==sidecar.stat().st_size and side[0]['fs_id']==sideup['remote_result']['fs_id']
    delivery={'status':'complete','zip_remote_path':remote,'zip_bytes':receipt['bytes'],'zip_sha256':receipt['sha256'],
        'zip_remote_fs_id':found[0]['fs_id'],'sidecar_remote_path':receipt['sidecar_remote_path'],
        'sidecar_remote_fs_id':side[0]['fs_id'],'sidecar_remote_bytes':side[0]['size'],'cards':50,
        'garment_triplets':25,'motion_source_reference_pairs_without_gt':25,
        'remote_listing_verified':True,'all_upload_block_md5s_verified':True,
        'remote_download_sha256_verified':receipt['remote_download_sha256_verified'],
        'verified_at_utc':datetime.now(timezone.utc).isoformat()}
    save(ROOT/'metadata/delivery_receipt.json',delivery);print(json.dumps(delivery,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
