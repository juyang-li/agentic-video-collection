"""Only the explicitly authorized, fully audited 300-output ZIP is sent to Baidu."""
from common import *
import time,fcntl
sys.path.insert(0,str(FROZEN_ROOT/'scripts'))
from upload_parallel import main as upload

def main():
    assert not os.environ.get('SLURM_JOB_ID'),'Upload runs on login, after GPU release'
    lock=(ROOT/'logs/upload.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    package=read(ROOT/'metadata/package_status.json')
    assert package['status']=='complete' and package['successful_outputs']==300 and package['zip_crc_verified']
    archive=Path(package['zip_path']);assert sha(archive)==package['zip_sha256']
    execution=ROOT/'metadata/upload_execution.json'
    atomic(execution,{'status':'starting','pid':os.getpid(),'remote':REMOTE,'only_zip_uploaded':True,'started_unix':time.time()})
    sys.argv=[sys.argv[0],'--local',str(archive),'--remote',REMOTE,'--state',str(ROOT/'metadata/baidu_upload.json'),'--bypy-config','/scratch/prj0000000275/jcy/ljy/video_collection/dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json','--workers','4']
    try:
        upload();assert read(ROOT/'metadata/baidu_upload.json')['status']=='complete'
        atomic(execution,{'status':'complete','remote':REMOTE,'only_zip_uploaded':True,'completed_unix':time.time(),'extra_remote_verification':False})
    except Exception as exc:
        atomic(execution,{'status':'failed','remote':REMOTE,'error':str(exc),'failed_unix':time.time()});raise
if __name__=='__main__':main()
