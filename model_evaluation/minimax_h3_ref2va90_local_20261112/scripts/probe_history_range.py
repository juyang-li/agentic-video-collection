from common import *
import requests
require_job();up=read(OLD/'review/a100_results_scroll_20261111.baidu_upload.json');token=read(ROOT.parents[1]/'dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json')['access_token']
s=requests.Session();s.trust_env=False;s.headers['User-Agent']='pan.baidu.com'
r=s.get('https://pan.baidu.com/rest/2.0/xpan/multimedia',params={'method':'filemetas','access_token':token,'fsids':json.dumps([up['remote_result']['fs_id']]),'dlink':1},timeout=60);r.raise_for_status();data=r.json();assert data.get('errno')==0
with s.get(data['list'][0]['dlink'],params={'access_token':token},headers={'Range':'bytes=0-15'},timeout=(20,60),stream=True) as r:
 result={'status':r.status_code,'content_range':r.headers.get('Content-Range'),'range_supported':r.status_code==206}
 if r.status_code==206:result['zip_magic_valid']=r.content[:4]==b'PK\x03\x04'
 atomic(ROOT/'metadata/history_range_support.json',result);print(json.dumps(result),flush=True)
