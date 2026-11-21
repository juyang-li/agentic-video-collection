from common import *
import time
from concurrent.futures import ThreadPoolExecutor
require_job();m=read(ROOT/'metadata/model_download_manifest.json');started=time.time()
def check(f):
 p=ROOT/'runtime/models/H3'/f['rfilename'];assert p.stat().st_size==f['size']
 digest=sha(p)
 if f.get('lfs'):assert digest==f['lfs']['sha256'],str(p)
 else:
  blob=hashlib.sha1(f'blob {p.stat().st_size}\0'.encode()+p.read_bytes()).hexdigest();assert blob==f['blobId'],str(p)
 return {'path':f['rfilename'],'bytes':f['size'],'sha256':digest}
with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(check,m['files']))
atomic(ROOT/'metadata/weights_verified.json',{'status':'passed','revision':m['revision'],'files':records,'seconds':time.time()-started})
print('WEIGHTS_VERIFIED',len(records),flush=True)
