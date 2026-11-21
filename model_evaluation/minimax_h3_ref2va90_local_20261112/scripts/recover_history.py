"""Recover historical comparison media; never a post-upload remote audit."""
from common import *
import sys,time,zipfile,requests
def load_token(path):
 token=read(path).get('access_token')
 if not token:raise RuntimeError('Missing configured Baidu access token')
 return token

def main():
 require_job();started=time.time();upload=read(OLD/'review/a100_results_scroll_20261111.baidu_upload.json');assert upload['status']=='complete'
 archive=ROOT/'review/history_source.zip'
 token=load_token(ROOT.parents[1]/'dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json')
 session=requests.Session();session.trust_env=False;session.headers['User-Agent']='pan.baidu.com'
 if not archive.exists():
  try:
   response=session.get('https://pan.baidu.com/rest/2.0/xpan/multimedia',params={'method':'filemetas','access_token':token,'fsids':json.dumps([upload['remote_result']['fs_id']]),'dlink':1},timeout=(20,60))
   response.raise_for_status();result=response.json();assert result.get('errno')==0,'filemetas rejected'
   url=result['list'][0]['dlink']
   tmp=archive.with_suffix('.part.zip')
   with session.get(url,params={'access_token':token},stream=True,timeout=(20,120)) as response:
    response.raise_for_status()
    with tmp.open('wb') as f:
     for b in response.iter_content(8*1024**2):
      if b:f.write(b)
   with zipfile.ZipFile(tmp) as z:assert any(n.endswith('/index.html') for n in z.namelist())
   tmp.replace(archive)
  except requests.RequestException as e:raise RuntimeError('Historical retrieval network failure: '+type(e).__name__) from None
 inventory=read(OLD/'metadata/deliveries/a100_results_scroll_20261111/manifest.json')
 wanted=[c for c in inventory if c['group']!='action'];assert len(wanted)==90
 files=[]
 with zipfile.ZipFile(archive) as z:
  names=z.namelist()
  for card in wanted:
   for media in card['media']:
    role=media['role']
    if role not in ('SOURCE','GT','ltx_base','a849') and not role.startswith('REFERENCE'):continue
    rel=Path(media['path']);assert not rel.is_absolute() and '..' not in rel.parts
    name=next(n for n in names if n.endswith('/'+str(rel)))
    dst=ROOT/'review/history'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():
     with z.open(name) as src,dst.open('wb') as target:
      import shutil;shutil.copyfileobj(src,target)
    if role in ('ltx_base','a849'):assert sha(dst)==card['generations'][role]['output_sha256'],str(dst)
    files.append({'id':card['id'],'role':role,'path':str(dst)})
 atomic(ROOT/'metadata/history_recovery.json',{'status':'complete','purpose':'recover model comparison inputs; not upload verification','remote':upload['remote'],'samples':90,'files':files,'seconds':time.time()-started})
 print('HISTORY_RECOVERY_COMPLETE',len(files),flush=True)
if __name__=='__main__':main()
