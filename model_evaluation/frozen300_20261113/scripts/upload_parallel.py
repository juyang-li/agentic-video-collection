"""Same approved Xpan uploader protocol, four concurrent block transfers, ZIP only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmark_full10s_20261112/scripts'))
from upload_baidu_xpan import *
import concurrent.futures as cf
import threading

def main():
 p=argparse.ArgumentParser();p.add_argument('--local',type=Path,required=True);p.add_argument('--remote',required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--bypy-config',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 assert a.local.suffix=='.zip' and a.remote.startswith('/apps/bypy/lance_review/') and a.remote.endswith('.zip')
 assert 1<=a.workers<=4
 local=a.local.resolve();size=local.stat().st_size;mtime=local.stat().st_mtime_ns;blocksize=4*1024**2
 hashes=block_hashes(local,blocksize);token=load_token(a.bypy_config)
 session=requests.Session();session.trust_env=False;session.headers['User-Agent']='lance-review-uploader/1.0'
 state=json.loads(a.state.read_text()) if a.state.exists() else {}
 matches=(state.get('local')==str(local) and state.get('remote')==a.remote and state.get('size')==size and state.get('mtime_ns')==mtime and state.get('block_md5s')==hashes and state.get('block_size')==blocksize and state.get('uploadid'))
 if not matches:
  result=retry_call('precreate',5,lambda:precreate(session,token,a.remote,size,hashes))
  state={'local':str(local),'remote':a.remote,'size':size,'mtime_ns':mtime,'block_size':blocksize,'block_md5s':hashes,'uploadid':result['uploadid'],'needed':[int(x) for x in result.get('block_list',range(len(hashes)))],'completed':[],'status':'uploading'};write_state(a.state,state)
 if state.get('status')=='complete':print('UPLOAD_ALREADY_COMPLETE',flush=True);return
 uploadid=state['uploadid'];completed=set(state['completed']);needed=state['needed'];thread=threading.local()
 def transfer(i):
  if not hasattr(thread,'session'):
   thread.session=requests.Session();thread.session.trust_env=False;thread.session.headers['User-Agent']='lance-review-uploader/1.0'
  with local.open('rb') as f:f.seek(i*blocksize);block=f.read(blocksize)
  def send():
   result=upload_block(thread.session,token,a.remote,uploadid,i,block)
   if result.get('md5') and result['md5']!=hashes[i]:raise RuntimeError('upload block digest mismatch')
   return result
  retry_call('upload block '+str(i),5,send);return i
 with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures=[pool.submit(transfer,i) for i in needed if i not in completed]
  for f in cf.as_completed(futures):
   completed.add(f.result());state['completed']=sorted(completed);write_state(a.state,state)
   if len(completed)%10==0 or len(completed)==len(needed):print(f'blocks={len(completed)}/{len(needed)} MiB={min(size,len(completed)*blocksize)/1024**2:.1f}',flush=True)
 result=retry_call('create',5,lambda:create_file(session,token,a.remote,size,uploadid,hashes))
 state.update(status='complete',remote_result={k:result.get(k) for k in ['fs_id','path','server_filename','size']});write_state(a.state,state)
 print('UPLOAD_COMPLETE',json.dumps(state['remote_result'],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
