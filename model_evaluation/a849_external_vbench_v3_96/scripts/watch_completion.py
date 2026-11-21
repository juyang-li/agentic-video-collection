"""CPU login watcher; finish scoring tables/packages after the independent GPU job.

Never treats pending direct visual review as passed. It does not call any VLM.
"""
import argparse,os,subprocess,time,traceback,sys
from common import *
from contracts import DIMS,MODELS
from score import inventory,DIAG


def completion():
 videos=inventory();missing=[r for r in videos if not r['output_sha256']]
 quality_missing=[];static_missing=[]
 for r in videos:
  h=r['output_sha256']
  if not h:continue
  for dim in DIMS:
   if dim=='temporal_flickering':continue
   p=ROOT/'metadata/vbench/quality'/f'{dim}--{h}.json'
   if not p.exists() or read(p).get('status')!='complete':quality_missing.append([r['cohort'],r['id'],r['model'],dim])
 media={r['output_sha256'] for r in videos if r['output_sha256']}
 for root in [ROOT,DIAG]:
  for item in rows(root/'metadata/inputs.jsonl'):media.add(sha(root/'inputs/prepared'/item['id']/'source.mp4'))
 for h in media:
  p=ROOT/'metadata/vbench/static'/f'{h}.json'
  if not p.exists() or read(p).get('status')!='complete':static_missing.append(h)
 return dict(missing_generations=len(missing),missing_quality_scores=len(quality_missing),missing_static_decisions=len(static_missing),generation_missing=missing,quality_missing=quality_missing,static_missing=static_missing)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--job-id',type=int,required=True);parser.add_argument('--interval',type=int,default=900);parser.add_argument('--once',action='store_true');args=parser.parse_args()
 assert args.interval>=60
 # CPU watcher is deliberately started on login, outside a GPU allocation.
 assert not os.environ.get('SLURM_JOB_ID'),'Run CPU completion watcher on login'
 while True:
  state=completion()
  correction=read(ROOT/'metadata/identity_protocol_correction.json') if (ROOT/'metadata/identity_protocol_correction.json').exists() else {'status':'validated'}
  state['identity_retest_pending']=correction['status']!='validated'
  ready=not state['identity_retest_pending'] and not any(state[k] for k in ['missing_generations','missing_quality_scores','missing_static_decisions'])
  atomic(ROOT/'metadata/completion_watch.json',dict(status='gpu_artifacts_ready' if ready else 'waiting_for_gpu_artifacts',job_id=args.job_id,checked_unix=time.time(),interval_seconds=args.interval,**state))
  print(time.strftime('%Y-%m-%d %H:%M:%S'),{k:state[k] for k in ['missing_generations','missing_quality_scores','missing_static_decisions']},flush=True)
  if ready:
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONNOUSERSITE='1')
   vbench=ROOT.parent/'a849_external_vbench_v2_96/runtime/vbench/bin/python'
   vace=ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace/bin/python'
   def run(py,name,*extra):subprocess.run([str(py),str(ROOT/'scripts'/name),*extra],check=True,env=env)
   try:
    run(vbench,'score.py','raw')
    run(vace,'aggregate.py')
    run(vace,'finalize_diagnostic.py')
    run(vace,'review.py','--stage','all','--pack')
    run(vace,'validate.py','--allow-pending-visual')
    validation=read(ROOT/'metadata/final_validation.json');atomic(ROOT/'metadata/completion_watch.json',dict(status=validation['status'],job_id=args.job_id,completed_unix=time.time(),note='All computational artifacts produced; awaiting_visual_review is not final approval. No VLM API used.'))
    break
   except Exception:
    atomic(ROOT/'metadata/completion_watch_error.json',dict(error=traceback.format_exc(),unix=time.time()));raise
  if args.once:break
  time.sleep(args.interval)
if __name__=='__main__':main()
