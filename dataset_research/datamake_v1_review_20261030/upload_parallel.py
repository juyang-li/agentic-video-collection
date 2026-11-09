"""Four-way resume of the existing, explicitly authorized Baidu upload."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,json,sys,threading,time
import requests

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'lance_general_edit_finetune/1_specialist_filtering/scripts'))
import upload_baidu_xpan as base

def main():
    parser=argparse.ArgumentParser()
    for name in ['local','remote','state','bypy-config']:parser.add_argument('--'+name,required=True)
    a=parser.parse_args();local=Path(a.local);statepath=Path(a.state)
    # Creation of a new session stays with the existing uploader. This only resumes it.
    if not statepath.exists():
        import subprocess
        subprocess.run([sys.executable,str(Path(base.__file__)),*sys.argv[1:]],check=True);return
    state=json.loads(statepath.read_text());size=local.stat().st_size
    assert state['local']==str(local.resolve()) and state['remote']==a.remote
    assert state['size']==size and state['mtime_ns']==local.stat().st_mtime_ns
    assert a.remote=='/apps/bypy/lance_review/datamake_v1_balanced80_scroll_20261030.zip'
    if state.get('status')=='complete':return
    token=base.load_token(Path(a.bypy_config));uploadid=state['uploadid']
    completed=set(state['completed']);needed=state['needed'];thread=threading.local()
    def worker(index):
        if not hasattr(thread,'session'):
            thread.session=requests.Session();thread.session.headers['User-Agent']='lance-review-uploader/1.0'
        with local.open('rb') as f:f.seek(index*state['block_size']);block=f.read(state['block_size'])
        for attempt in range(1,6):
            try:
                result=base.upload_block(thread.session,token,a.remote,uploadid,index,block)
                returned=str(result.get('md5',''))
                if returned and returned!=state['block_md5s'][index]:raise RuntimeError('block md5 mismatch')
                return index
            except Exception:
                if attempt==5:raise
                time.sleep(min(2**attempt,10))
    print('resuming',len(completed),'/',len(needed),'blocks with 4 workers',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(worker,i) for i in needed if i not in completed]
        for job in as_completed(jobs):
            completed.add(job.result());state['completed']=sorted(completed);base.write_state(statepath,state)
            if len(completed)%25==0:print('blocks',len(completed),'/',len(needed),flush=True)
    session=requests.Session();session.headers['User-Agent']='lance-review-uploader/1.0'
    result=base.retry_call('create',5,lambda:base.create_file(session,token,a.remote,size,uploadid,state['block_md5s']))
    assert int(result.get('size',size))==size
    state['status']='complete';state['remote_result']={k:result.get(k) for k in ['fs_id','path','server_filename','size']}
    base.write_state(statepath,state);print('upload complete',flush=True)

if __name__=='__main__':main()
