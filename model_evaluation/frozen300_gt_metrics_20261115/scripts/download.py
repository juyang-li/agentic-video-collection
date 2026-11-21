from common import *
import requests,time,concurrent.futures as cf
from huggingface_hub import get_token

def main():
 token=get_token();base=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/';dest=ROOT/'remote'
 def get(name):
  p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
  expected=hashes.get(name)
  if p.exists() and (not expected or sha(p)==expected):return
  for attempt in range(4):
   try:
    with requests.Session() as s:
     s.trust_env=False;tmp=p.with_suffix(p.suffix+'.part');offset=tmp.stat().st_size if tmp.exists() else 0
     h={'Authorization':'Bearer '+token}
     if offset:h['Range']=f'bytes={offset}-'
     with s.get(base+name,headers=h,stream=True,timeout=(15,90)) as r:
      r.raise_for_status();append=offset and r.status_code==206
      with tmp.open('ab' if append else 'wb') as f:
       for b in r.iter_content(1024**2):f.write(b)
     if expected and sha(tmp)!=expected:tmp.unlink();raise RuntimeError('SHA mismatch '+name)
     tmp.replace(p);return
   except Exception as e:
    print('retry',name,attempt,type(e).__name__,flush=True)
    if attempt==3:raise RuntimeError('Download failed '+name) from None
    time.sleep(3)
 hashes={}
 for name in ['README.md','summary.json','SHA256SUMS_outputs.txt','results/h3/results.jsonl','results/vace/results.jsonl','code/common/frozen300.py']:get(name)
 for l in (dest/'SHA256SUMS_outputs.txt').read_text().splitlines():
  if l and not l.startswith('#'):
   a=l.split();hashes[a[1]]=a[0]
 assert len(hashes)==600
 names=list(hashes)+[f'results/{m}/{r["id"]}.json' for m in ['h3','vace'] for r in rows(PKG/'inputs.jsonl')]
 with cf.ThreadPoolExecutor(max_workers=4) as pool:
  for i,_ in enumerate(pool.map(get,names),1):
   if i%25==0:print('downloaded',i,'/',len(names),flush=True)
 atomic(ROOT/'metadata/download.json',{'status':'complete','revision':REV,'files':hashes})
if __name__=='__main__':main()
