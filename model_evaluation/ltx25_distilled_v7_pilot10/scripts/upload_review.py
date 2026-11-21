"""Background upload of only the completed review ZIP; preserve resumable receipts."""
from common import *
import fcntl,time
sys.path.insert(0,str(BASE/'scripts'))
from upload_baidu_xpan import main as upload
lock=(ROOT/'logs/upload.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not os.environ.get('SLURM_JOB_ID'),'CPU upload runs on login'
receipt=read(ROOT/'metadata/package_status.json')
assert receipt['successful_outputs']==10 and receipt['zip_sha256']==sha(ROOT/'review.zip')
remote='/apps/bypy/lance_review/ltx25_distilled_v7_pilot10_review.zip'
state=ROOT/'metadata/baidu_upload.json'
execution=ROOT/'metadata/upload_execution.json'
atomic(execution,{'status':'starting','pid':os.getpid(),'remote':remote,'only_zip_uploaded':True,'started_unix':time.time()})
sys.argv=[sys.argv[0],'--local',str(ROOT/'review.zip'),'--remote',remote,'--state',str(state),'--bypy-config','/scratch/prj0000000275/jcy/ljy/video_collection/dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json']
try:
    upload()
    assert read(state)['status']=='complete'
    atomic(execution,{'status':'complete','remote':remote,'only_zip_uploaded':True,'completed_unix':time.time()})
except Exception as e:
    atomic(execution,{'status':'failed','remote':remote,'error':str(e),'failed_unix':time.time()})
    raise
