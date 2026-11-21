"""Resume approved full evaluation, validate, package, and upload ZIP only."""
import os,subprocess,sys,time
from engine import ROOT,EVAL,read,save

def main():
    assert read(ROOT/'metadata/checkpoint100_review.json')['status']=='approved_to_continue'
    log=ROOT/'logs';log.mkdir(exist_ok=True)
    for script,args in [('engine.py',[]),('report.py',[]),('audit.py',[]),('review.py',[])]:
        with (log/('full_'+script+'.log')).open('a') as stream:
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),*args],stdout=stream,stderr=subprocess.STDOUT,check=True)
    assert read(ROOT/'metadata/completion_audit.json')['status']=='passed'
    archive=ROOT/'ds_full_eval_20261115_review40.zip'
    env={k:v for k,v in os.environ.items() if k.lower() not in ['http_proxy','https_proxy','all_proxy']}
    command=[sys.executable,str(EVAL/'minimax_h3_ref2va90_local_20261112/scripts/upload_baidu_xpan.py'),'--local',str(archive),'--remote','/apps/bypy/lance_review/'+archive.name,'--state',str(ROOT/'metadata/baidu_upload.json'),'--bypy-config',str(EVAL.parent/'dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json')]
    with (log/'baidu_upload.log').open('a') as stream:subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    save(ROOT/'metadata/delivery_complete.json',{'status':'complete','finished_unix':time.time(),'local_zip':str(archive),'remote':'/apps/bypy/lance_review/'+archive.name,'upload_state':'metadata/baidu_upload.json','audit':'metadata/completion_audit.json'})
    print('Completed full evaluation and ZIP delivery',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(ROOT/'metadata/pipeline_failure.json',{'error':repr(exc),'time':time.time()});raise
