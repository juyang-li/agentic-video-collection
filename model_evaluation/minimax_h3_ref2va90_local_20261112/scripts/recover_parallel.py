"""Range-resume a single historical ZIP, retaining already received bytes."""
from common import *
import requests,time,math,subprocess,sys,zipfile,shutil
from concurrent.futures import ThreadPoolExecutor
require_job();upload=read(OLD/'review/a100_results_scroll_20261111.baidu_upload.json')
archive=ROOT/'review/history_source.zip';chunkdir=ROOT/'review/history_chunks';chunkdir.mkdir(exist_ok=True)
if not archive.exists():
 token=read(ROOT.parents[1]/'dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json')['access_token']
 s=requests.Session();s.trust_env=False;s.headers['User-Agent']='pan.baidu.com'
 r=s.get('https://pan.baidu.com/rest/2.0/xpan/multimedia',params={'method':'filemetas','access_token':token,'fsids':json.dumps([upload['remote_result']['fs_id']]),'dlink':1},timeout=60);r.raise_for_status();result=r.json();assert result.get('errno')==0;url=result['list'][0]['dlink']
 manifest=chunkdir/'ranges.json'
 if not manifest.exists():
  partial=archive.with_suffix('.part.zip');prefix=chunkdir/'prefix.bin'
  if partial.exists():partial.replace(prefix)
  n=prefix.stat().st_size if prefix.exists() else 0;total=upload['size'];width=math.ceil((total-n)/4)
  ranges=[{'index':i,'start':a,'end':min(total-1,a+width-1)} for i,a in enumerate(range(n,total,width))]
  atomic(manifest,{'total':total,'prefix_bytes':n,'ranges':ranges,'purpose':'restore historical media, not upload validation'})
 spec=read(manifest)
 def download(part):
  path=chunkdir/f'{part["index"]:02d}.bin';expected=part['end']-part['start']+1
  for attempt in range(5):
   have=path.stat().st_size if path.exists() else 0;assert have<=expected
   if have==expected:return
   start=part['start']+have;session=requests.Session();session.trust_env=False;session.headers['User-Agent']='pan.baidu.com'
   try:
    with session.get(url,params={'access_token':token},headers={'Range':f'bytes={start}-{part["end"]}'},stream=True,timeout=(20,120)) as response:
     response.raise_for_status();assert response.status_code==206 and response.headers.get('Content-Range','').startswith(f'bytes {start}-')
     with path.open('ab') as f:
      for block in response.iter_content(1024**2):
       if block:f.write(block)
    assert path.stat().st_size==expected
    print('range complete',part['index'],expected,flush=True);return
   except requests.RequestException as e:
    if attempt==4:raise RuntimeError('Historical range network error: '+type(e).__name__) from None
    time.sleep(5)
  raise RuntimeError('Range retry budget exhausted')
 with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(download,spec['ranges']))
 tmp=archive.with_suffix('.assembling.zip')
 with tmp.open('wb') as out:
  prefix=chunkdir/'prefix.bin'
  if prefix.exists():
   with prefix.open('rb') as f:shutil.copyfileobj(f,out)
  for part in spec['ranges']:
   with (chunkdir/f'{part["index"]:02d}.bin').open('rb') as f:shutil.copyfileobj(f,out)
 assert tmp.stat().st_size==spec['total']
 with zipfile.ZipFile(tmp) as z:assert any(n.endswith('/index.html') for n in z.namelist())
 tmp.replace(archive)
subprocess.run([sys.executable,str(ROOT/'scripts/recover_history.py')],check=True)
