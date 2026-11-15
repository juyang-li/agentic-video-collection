"""Login controller: overlap frozen prepared batches and GPU scoring; audit all 300."""
from common import *
import subprocess,time,fcntl,traceback

def allocation_time():
 # Revalidate at launch: preparation can take hours after the first resource check.
 info=subprocess.check_output(['scontrol','show','job','202896','--oneliner'],text=True)
 fields=dict(x.split('=',1) for x in info.split() if '=' in x)
 assert fields.get('Partition')=='testqueue' and fields.get('JobState')=='RUNNING',info
 remaining=subprocess.check_output(['squeue','-h','-j','202896','-o','%L'],text=True).strip()
 days,clock=remaining.split('-',1) if '-' in remaining else ('0',remaining)
 parts=[int(p) for p in clock.split(':')]
 seconds=int(days)*86400+sum(v*60**i for i,v in enumerate(reversed(parts)))
 minutes=min(420,(seconds-300)//60)
 assert minutes>=20, f'Allocation has insufficient time: {remaining}'
 return str(minutes),remaining,info

def check_prepare_live():
 pid=int((ROOT/'metadata/prepare.pid').read_text())
 try:os.kill(pid,0)
 except ProcessLookupError:raise RuntimeError('CPU preparation/raw scorer stopped before completed receipt')

def prepared_media():
 outputs=[];sources=[]
 for item in rows(PKG/'inputs.jsonl'):
  p=ROOT/f'metadata/media/{item["id"]}.json'
  if not p.exists():continue
  r=read(p);assert r['status']=='complete' and r['id']==item['id']
  assert {v['model'] for v in r['outputs']}==set(MODELS) and len(r['outputs'])==4
  assert all(v['id']==item['id'] and v['uid']==item['uid'] and v['num_frames']==item['num_frames'] for v in r['outputs'])
  assert r['source']['id']==item['id'] and r['source']['model']=='source'
  outputs.extend(r['outputs']);sources.append(r['source'])
 return outputs,sources

def snapshot_batch(tag,outputs,sources):
 dest=ROOT/'metadata/batches';dest.mkdir(exist_ok=True)
 paths=[]
 for name,records in [('scoring',outputs),('source',sources)]:
  p=dest/f'{tag}_{name}.jsonl';assert not p.exists()
  p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));paths.append(p)
 return paths

def main():
 assert not os.environ.get('SLURM_JOB_ID')
 lock=(ROOT/'logs/supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 atomic(ROOT/'metadata/run_status.json',{'status':'preparing','pid':os.getpid()})
 batch=0;previous=0
 while True:
  outputs,sources=prepared_media();ready=len(sources)
  if ready<300 and ready-previous<32:
   check_prepare_live();time.sleep(30);continue
  tag=f'batch{batch:03d}'
  while (ROOT/f'metadata/batches/{tag}_scoring.jsonl').exists():
   batch+=1;tag=f'batch{batch:03d}'
  scoring_path,source_path=snapshot_batch(tag,outputs,sources)
  minutes,remaining,resource_info=allocation_time()
  cmd=['srun','--jobid=202896','--partition=testqueue','--exclusive','--nodes=1','--ntasks=1','--cpus-per-task=56','--gres=gpu:4','--mem=384G',f'--time={minutes}','bash',str(ROOT/'scripts/gpu.sh')]
  submission={'status':'starting_step','phase':tag,'prepared_tasks':ready,'allocation':'202896','partition':'testqueue','gpus':4,'cpus':56,'time_minutes':int(minutes),'allocation_remaining':remaining,'resource_info':resource_info,'command':cmd,'scoring_manifest':str(scoring_path),'source_manifest':str(source_path),'scoring_manifest_sha256':sha(scoring_path),'source_manifest_sha256':sha(source_path)}
  atomic(ROOT/'metadata/submission.json',submission);atomic(ROOT/f'metadata/batches/{tag}_submission.json',submission)
  env=os.environ.copy();env.update(VB_SCORING_MANIFEST=str(scoring_path),VB_SOURCE_MANIFEST=str(source_path),VB_PHASE_TAG=tag)
  with (ROOT/f'logs/driver_{tag}.log').open('ab',buffering=0) as log:
   p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
   atomic(ROOT/'metadata/gpu_process.json',{'pid':p.pid,'allocation':'202896','phase':tag})
   atomic(ROOT/'metadata/run_status.json',{'status':'gpu_running_or_waiting_step','allocation':'202896','srun_pid':p.pid,'phase':tag,'prepared_tasks':ready})
   rc=p.wait()
  atomic(ROOT/f'metadata/batches/{tag}_exit.json',{'gpu_exit_code':rc,'prepared_tasks':ready})
  assert rc==0,f'GPU batch {tag} failed with exit code {rc}; inspect logs before any retry'
  failures=[str(p) for kind in ['quality','static'] for p in (ROOT/f'metadata/vbench/{kind}').glob('*.json') if read(p)['status']!='complete']
  assert not failures,f'GPU score failures require inspection: {failures}'
  if ready==300:break
  previous=ready;batch+=1
 # Wait for CPU raw scorer independently; it must not be moved to the GPU step.
 while not (ROOT/'metadata/raw_completion.json').exists():
  check_prepare_live();time.sleep(30)
 env=os.environ.copy();env['PYTHONNOUSERSITE']='1'
 py=str(OLD/'runtime/vbench/bin/python')
 with (ROOT/'logs/aggregate.log').open('ab',buffering=0) as log:subprocess.run([py,str(ROOT/'scripts/aggregate.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 complete=read(ROOT/'metadata/completion.json')
 if complete['status']=='complete':
  with (ROOT/'logs/validate.log').open('w') as log:subprocess.run([py,str(ROOT/'scripts/validate.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 atomic(ROOT/'metadata/run_status.json',{'status':complete['status'],'gpu_exit_code':rc,'completion':complete})
 print('VBENCH_FINISHED',complete['status'],flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  atomic(ROOT/'metadata/supervisor_failure.json',{'error':traceback.format_exc()});raise
