#!/usr/bin/env python3
"""Refresh the existing Baidu authorization without changing home configuration."""
import json
import os
from pathlib import Path
import requests
from bypy import const

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'metadata/.baidu_session.json'


def main():
    old = json.loads((Path.home()/'.bypy/bypy.json').read_text())
    try:
        r=requests.post(const.TokenUrl, data={'grant_type':'refresh_token',
            'refresh_token':old['refresh_token'],'client_id':const.ApiKey,
            'client_secret':const.SecretKey},timeout=(20,45))
        d=r.json()
    except requests.RequestException as e:
        raise RuntimeError('Baidu refresh network failure: '+type(e).__name__) from None
    if not d.get('access_token'):
        print(json.dumps({'status':'refresh_failed','http_status':r.status_code,
                          'error':d.get('error'),'description':d.get('error_description')}))
        return
    DEST.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(DEST,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as f:json.dump(d,f)
    os.chmod(DEST,0o600)
    check=requests.get('https://pan.baidu.com/rest/2.0/xpan/file',params={
        'method':'list','access_token':d['access_token'],'dir':'/apps/bypy/lance_review','limit':1},timeout=(20,45)).json()
    print(json.dumps({'status':'refreshed','directory_errno':check.get('errno'),
                      'config_location':'private task-local session; excluded from review package'}))


if __name__=='__main__':main()
