from common import *
import subprocess,concurrent.futures as cf,queue,fcntl,traceback

def launch_cpu(scope,phase):
 log=(ROOT/f'logs/cpu_{scope}_{phase}.log').open('ab',buffering=0)
 return subprocess.Popen([str(GTPY),str(ROOT/'scripts/cpu.py'),'--scope',scope,'--phase',phase],stdout=log,stderr=subprocess.STDOUT)

def gpu_phase(phase,devices):
 selected=cases_for(phase);jobs=[]
 for offset in range(0,len(selected),8):jobs.append({'kind':'lpips','cases':selected[offset:offset+8],'tag':f'{phase}_lpips_{offset}'})
 for dim in VBDIMS:
  if dim=='temporal_flickering':continue
  for offset in range(0,len(selected),64):jobs.append({'kind':'vbench','dimension':dim,'cases':selected[offset:offset+64],'tag':f'{phase}_{dim}_{offset}'})
 tasks=queue.Queue()
 for task in jobs:tasks.put(task)
 def worker(device,index):
  statuses=[]
  while True:
   try:task=tasks.get_nowait()
   except queue.Empty:return statuses
   tag=task['tag'];p=ROOT/f'runtime/tasks/{tag}.json';atomic(p,task)
   env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device
   if task['kind']=='lpips':py=GTPY;env['TORCH_HOME']='/scratch/prj0000000275/jcy/.cache/torch'
   else:py=VBPY
   start=time.time();atomic(ROOT/f'metadata/gpu_worker_{index}.json',{'status':'running','task':tag,'device':device,'started_unix':start})
   with (ROOT/f'logs/{tag}.log').open('ab',buffering=0) as log:rc=subprocess.run([str(py),str(ROOT/'scripts/gpu_task.py'),'--task',str(p)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
   atomic(ROOT/f'metadata/tasks/{tag}.json',{'status':'complete' if rc==0 else 'failed','return_code':rc,'device':device,'started_unix':start,'completed_unix':time.time(),'worker':index})
   statuses.append(rc)

 with cf.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(lambda x:worker(x[1],x[0]),enumerate(devices)))
 return all(rc==0 for group in results for rc in group)

def main():
 require_node();lock=(ROOT/'logs/node.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 import torch
 assert torch.cuda.device_count()==4
 for i in range(4):
  free,total=torch.cuda.mem_get_info(i);assert 'H200' in torch.cuda.get_device_name(i) and total-free<2*1024**3,'GPU is busy'
 devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==4
 atomic(ROOT/'metadata/node_resources.json',{'gpu_names':[torch.cuda.get_device_name(i) for i in range(4)],'visible_devices':devices,'job':JOB,'pid':os.getpid(),'cpu_count':len(os.sched_getaffinity(0)),'cpu_workers':8,'cpu_frame_threads':4,'raw_workers':2,'gpu_workers':4})
 for phase in ['smoke','full']:
  cpu=launch_cpu('node',phase)
  log=(ROOT/f'logs/raw_{phase}.log').open('ab',buffering=0)
  raw=subprocess.Popen([str(VBPY),str(ROOT/'scripts/raw.py'),'--phase',phase],stdout=log,stderr=subprocess.STDOUT)
  gpu_ok=gpu_phase(phase,devices);cpu_rc=cpu.wait();raw_rc=raw.wait()
  assert gpu_ok and cpu_rc==raw_rc==0,f'{phase}: GPU={gpu_ok}, CPU={cpu_rc}, RAW={raw_rc}'
  if phase=='smoke':
   marker=ROOT/'metadata/cpu_login_smoke.json'
   while not marker.exists():time.sleep(5)
   assert read(marker)['status']=='complete'
   for r in cases_for('smoke'):
    assert compatible(path_score('cpu',r),r) and compatible(path_score('lpips',r),r)
    for dim in VBDIMS:assert compatible(path_score('vbench',r,dim),r)
   atomic(ROOT/'metadata/smoke.json',{'status':'passed','outputs':12,'score_keys':108,'ids':SMOKE,'static_filter':'not_requested','completed_unix':time.time()})
   print('SMOKE_PASSED_108',flush=True)
 atomic(ROOT/'metadata/node_done.json',{'status':'complete','completed_unix':time.time()})
if __name__=='__main__':
 try:main()
 except Exception:
  atomic(ROOT/'metadata/node_failure.json',{'error':traceback.format_exc()});raise
