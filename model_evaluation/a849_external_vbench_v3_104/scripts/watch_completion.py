"""Independent login CPU completion workflow; never converts pending direct review into approval."""
import argparse,subprocess,time,traceback,sys
from common import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--interval',type=int,default=60);p.add_argument('--once',action='store_true');p.add_argument('--upload',action='store_true');a=p.parse_args();assert a.interval>=60 and not os.environ.get('SLURM_JOB_ID')
 vace=ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace/bin/python';vbench=ROOT.parent/'a849_external_vbench_v2_96/runtime/vbench/bin/python';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONNOUSERSITE='1')
 def run(py,name,*args):subprocess.run([str(py),str(ROOT/'scripts'/name),*args],check=True,env=env)
 last=None
 while True:
  ready=(ROOT/'metadata/gpu_scores_complete.json').exists()and(ROOT/'metadata/h3_derivation.json').exists()and read(ROOT/'metadata/h3_derivation.json')['status']=='complete'
  if not ready:
   atomic(ROOT/'metadata/completion_watch.json',dict(status='waiting_for_gpu_artifacts',checked_unix=time.time()));print(time.strftime('%F %T'),'waiting for GPU artifacts',flush=True)
  else:
   visual=ROOT/'metadata/visual_observations.json';signature=sha(visual)if visual.exists()else'missing'
   if signature!=last:
    run(vbench,'score.py','raw');run(vace,'visual.py','sync');run(vace,'aggregate.py');run(vace,'review.py','--stage','all');run(vace,'validate.py','--allow-pending-visual');last=sha(visual);validation=read(ROOT/'metadata/final_validation.json');atomic(ROOT/'metadata/completion_watch.json',dict(status=validation['status'],checked_unix=time.time(),note='Pending direct inspection is never passed. No external VLM API.'))
    if validation['status']=='passed':
     run(vace,'review.py','--stage','all','--pack')
     if a.upload:
      package=read(ROOT/'metadata/package.json');upload_env=dict(env)
      for key in('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):upload_env.pop(key,None)
      state=ROOT/'metadata/baidu_delivery.json';repo=ROOT.parents[1];remote='/apps/bypy/lance_review/a849_external_vbench_v3_104_scroll.zip';uploader=ROOT.parent/'minimax_h3_ref2va90_local_20261112/scripts/upload_baidu_xpan.py';credentials=repo/'dataset_research/ref_edit_review_20261030/metadata/.baidu_session.json'
      subprocess.run([str(vace),str(uploader),'--local',package['path'],'--remote',remote,'--state',str(state),'--bypy-config',str(credentials)],check=True,env=upload_env)
      assert read(state)['status']=='complete'
     atomic(ROOT/'metadata/completion_watch.json',dict(status='complete',uploaded=a.upload,completed_unix=time.time()));break
  if a.once:break
  time.sleep(a.interval)
if __name__=='__main__':
 try:main()
 except BaseException:
  atomic(ROOT/'metadata/completion_watch_error.json',dict(error=traceback.format_exc(),failed_unix=time.time()));raise
