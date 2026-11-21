"""Repair the 2048-part limit using 8MiB parts, bounded retries and persisted errors.
The exact original ZIP and approved remote path are retained. No inference rerun.
"""
from common import *
import argparse,concurrent.futures as cf,fcntl,hashlib,time,requests,threading
TOKEN_PATH=Path('/scratch/prj0000000275/jcy/ljy/video_collection/dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json')
STATE=ROOT/'metadata/baidu_upload_8mib.json'
EXECUTION=ROOT/'metadata/upload_execution.json'
BLOCK=8*1024**2

def session():
    s=requests.Session();s.trust_env=False;s.headers['User-Agent']='ltx-review-uploader/2.0';return s

def call(s,method,url,**kwargs):
    try:
        r=getattr(s,method)(url,timeout=(10,45),**kwargs)
        try:data=r.json()
        except ValueError:raise RuntimeError(f'HTTP {r.status_code}: non-JSON response') from None
        errno=int(data.get('errno',data.get('error_code',0)) or 0)
        if r.status_code!=200 or errno:
            message=str(data.get('error_msg',data.get('errmsg','')))[:200]
            raise RuntimeError(f'HTTP {r.status_code}, errno={errno}, message={message}')
        return data
    except requests.RequestException as e:raise RuntimeError(type(e).__name__) from None

def retry(label,fn,attempts=3):
    for attempt in range(1,attempts+1):
        try:return fn()
        except RuntimeError as e:
            print(f'{label} attempt={attempt}/{attempts} error={e}',flush=True)
            if 'Invalid param' in str(e) or attempt==attempts:raise
            time.sleep(min(2**attempt,8))

def main(workers):
    assert not os.environ.get('SLURM_JOB_ID')
    lock=(ROOT/'logs/upload.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=read(ROOT/'metadata/package_status.json');assert receipt['status']=='complete' and receipt['successful_outputs']==300
    path=Path(receipt['zip_path']);size=path.stat().st_size
    atomic(EXECUTION,{'status':'hashing','pid':os.getpid(),'remote':REMOTE,'block_size':BLOCK,'started_unix':time.time()})
    hashes=[];digest=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(BLOCK):hashes.append(hashlib.md5(b).hexdigest());digest.update(b)
    assert digest.hexdigest()==receipt['zip_sha256'] and len(hashes)<=2048
    token=read(TOKEN_PATH)['access_token'];s=session()
    state=read(STATE) if STATE.exists() else {}
    if not (state.get('block_md5s')==hashes and state.get('remote')==REMOTE and state.get('size')==size and state.get('uploadid')):
        data=retry('precreate',lambda:call(s,'post','https://pan.baidu.com/rest/2.0/xpan/file',params={'method':'precreate','access_token':token},data={'path':REMOTE,'size':size,'isdir':0,'autoinit':1,'rtype':3,'block_list':json.dumps(hashes)}))
        state={'local':str(path),'remote':REMOTE,'size':size,'mtime_ns':path.stat().st_mtime_ns,'block_size':BLOCK,'block_md5s':hashes,'uploadid':data['uploadid'],'needed':[int(i) for i in data.get('block_list',range(len(hashes)))],'completed':[],'status':'uploading'}
        atomic(STATE,state)
    if state['status']=='complete':print('ALREADY_COMPLETE',flush=True);return
    host='c.pcs.baidu.com'
    # Use the server recommended for this upload session when available.
    try:
        loc=call(s,'get','https://d.pcs.baidu.com/rest/2.0/pcs/file',params={'method':'locateupload','access_token':token,'path':REMOTE,'uploadid':state['uploadid']})
        candidate=loc.get('host','')
        if candidate.endswith('.pcs.baidu.com') and '/' not in candidate:host=candidate
    except RuntimeError as e:print('locateupload fallback:',e,flush=True)
    thread=threading.local()
    def transfer(index):
        t=time.monotonic()
        with path.open('rb') as f:f.seek(index*BLOCK);block=f.read(BLOCK)
        if not hasattr(thread,'session'):thread.session=session()
        data=retry(f'block={index}',lambda:call(thread.session,'post',f'https://{host}/rest/2.0/pcs/superfile2',params={'method':'upload','access_token':token,'type':'tmpfile','path':REMOTE,'uploadid':state['uploadid'],'partseq':index},files={'file':('block',block,'application/octet-stream')}))
        assert data.get('md5')==hashes[index],f'MD5 mismatch block {index}'
        return index,round(time.monotonic()-t,2)
    completed=set(state['completed']);pending=[i for i in state['needed'] if i not in completed]
    atomic(EXECUTION,{'status':'uploading','pid':os.getpid(),'remote':REMOTE,'block_size':BLOCK,'total_blocks':len(hashes),'host':host})
    atomic(ROOT/'metadata/delivery_status.json',{'status':'uploading','remote':REMOTE,'completed':300,'recovery':'8MiB blocks; fixed invalid part_id'})
    def save(result):
        index,seconds=result;completed.add(index);state.update(status='uploading',completed=sorted(completed),updated_unix=time.time(),last_block_seconds=seconds);atomic(STATE,state)
        print(f'blocks={len(completed)}/{len(state["needed"])} index={index} seconds={seconds}',flush=True)
    # Validate one full-sized part before starting bulk transfers (account-dependent part-size limit).
    if pending:save(transfer(pending.pop(0)))
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        iterator=iter(pending);active={}
        for _ in range(workers):
            i=next(iterator,None)
            if i is not None:active[pool.submit(transfer,i)]=i
        while active:
            done,_=cf.wait(active,return_when=cf.FIRST_COMPLETED)
            for future in done:
                del active[future]
                try:save(future.result())
                except Exception as e:
                    state.update(status='failed',last_error=str(e),updated_unix=time.time());atomic(STATE,state)
                    for f in active:f.cancel()
                    raise
                i=next(iterator,None)
                if i is not None:active[pool.submit(transfer,i)]=i
    assert set(state['needed'])<=completed
    result=retry('create',lambda:call(s,'post','https://pan.baidu.com/rest/2.0/xpan/file',params={'method':'create','access_token':token},data={'path':REMOTE,'size':size,'isdir':0,'rtype':3,'uploadid':state['uploadid'],'block_list':json.dumps(hashes)}))
    assert int(result.get('size',-1))==size and result.get('path')==REMOTE
    state.update(status='complete',remote_result={k:result.get(k) for k in ['fs_id','path','server_filename','size']},completed_unix=time.time());atomic(STATE,state)
    atomic(ROOT/'metadata/baidu_upload.json',state)
    atomic(EXECUTION,{'status':'complete','remote':REMOTE,'only_zip_uploaded':True,'completed_unix':time.time(),'extra_remote_verification':False,'recovery_state':str(STATE)})
    atomic(ROOT/'metadata/delivery_status.json',{'status':'complete','completed':300,'remote':REMOTE,'only_zip_uploaded':True,'completed_unix':time.time()})
    atomic(ROOT/'metadata/upload_recovery.json',{'status':'complete','cause':'4MiB parts exceeded allowed part_id range','block_size':BLOCK,'blocks':len(hashes),'same_original_zip':True,'remote':REMOTE})
    print('UPLOAD_COMPLETE',json.dumps(state['remote_result'],ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3,choices=range(1,5));a=p.parse_args()
    try:main(a.workers)
    except Exception as e:
        atomic(EXECUTION,{'status':'failed','remote':REMOTE,'error':str(e),'failed_unix':time.time()})
        print('RECOVERY_FAILED',str(e),flush=True);raise SystemExit(1)
