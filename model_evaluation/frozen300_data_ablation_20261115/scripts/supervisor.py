"""Login controller: score, summarize, validate, then release only job205167."""
from common import *
import subprocess,signal,fcntl,traceback,collections

def command(args,check=True):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 if check and p.returncode:raise RuntimeError(f'{args}: {p.stdout}')
 return p.stdout.strip()

def job_check():
 import re,pwd
 s=command(['scontrol','show','job','-o',JOB])
 fields=dict(re.findall(r'(\w+)=([^\s]+)',s))
 assert fields['JobId']==JOB and fields['JobState']=='RUNNING' and fields['Partition']=='testqueue'
 assert fields['UserId'].split('(')[0]==pwd.getpwuid(os.getuid()).pw_name
 assert int(fields['NumCPUs'])>=56 and 'gres/gpu=4' in fields['AllocTRES']
 from datetime import datetime
 remaining=(datetime.fromisoformat(fields['EndTime'])-datetime.now()).total_seconds()
 assert remaining>7200,'Less than two hours remain; do not launch blindly'
 steps=command(['squeue','--steps','-h','-j',JOB,'-o','%i']).splitlines()
 assert all(x.strip().split('.')[-1] in ['interactive','extern'] for x in steps),steps
 atomic(ROOT/'metadata/allocation_before.json',dict(job=JOB,details=fields,steps=steps,checked_unix=time.time()))
 return min(28800,int(remaining)-600)

def release():
 # The user expressly authorized releasing the entire allocation at task end.
 output=command(['scancel',JOB],check=False)
 for _ in range(12):
  remaining=command(['squeue','-h','-j',JOB,'-o','%i %T'],check=False)
  if not any(x.split()[0]==JOB for x in remaining.splitlines() if x.split()):break
  time.sleep(5)
 else:raise RuntimeError(f'Job still appears in squeue: {remaining}')
 accounting=command(['sacct','-j',JOB,'--noheader','--parsable2','--format=JobIDRaw,State,End,ExitCode'],check=False)
 atomic(ROOT/'metadata/release.json',dict(status='released',job=JOB,authorized_by_user=True,scancel_output=output,squeue_after=remaining,sacct=accounting,released_unix=time.time()))

def timing(start,preparation_start,end,status):
 tasks=[read(p) for p in (ROOT/'metadata/tasks').glob('*.json')]
 groups=collections.defaultdict(list)
 for p in (ROOT/'runtime/tasks').glob('*.json'):
  task=read(p);result=ROOT/f'metadata/tasks/{task["tag"]}.json'
  if result.exists():
   r=read(result);groups[task.get('dimension',task['kind'])].append(r['completed_unix']-r['started_unix'])
 cpu={p.stem:read(p) for p in (ROOT/'metadata').glob('cpu_*.json')}
 raw={p.stem:read(p) for p in (ROOT/'metadata').glob('raw_*.json')}
 wall=end-start;overall=end-preparation_start
 complete=read(ROOT/'metadata/completion.json').get('valid_new_score_keys') if (ROOT/'metadata/completion.json').exists() else 0
 r=dict(status=status,started_unix=start,ended_unix=end,scoring_and_reporting_wall_seconds=wall,including_preparation_wall_seconds=overall,
  scoring_outputs_per_hour=600*3600/wall if complete==5400 else None,gt_frame_pairs_per_second=62424/wall if complete==5400 else None,
  cpu_stages=cpu,raw_stages=raw,gpu_subprocess_wall_sums_seconds={k:sum(v) for k,v in groups.items()},gpu_subprocesses=len(tasks),
  note='GPU subprocess sums overlap and include startup/model loading, decode, inference and writing. Do not add parallel stages to obtain wall time. First/cold tasks retained.',
  resources=dict(gpu='4 H200',job=JOB,node_cpu_workers=8,node_frame_threads=4,login_cpu_workers=2,login_frame_threads=2,raw_cpu_workers=2,gpu_torch_threads=2))
 atomic(ROOT/'metadata/timing.json',r)
 lines=['# 实际用时报告','',f'状态：{status}；评分与汇总验收墙钟 {wall/3600:.3f} 小时；含前置检查 {overall/3600:.3f} 小时。',
  '资源：205167的4张H200、56 CPU；SSIM/PSNR节点8进程×4线程，login 2进程×2线程；原始闪烁2进程。GPU按独立进程队列运行。',
  '首条、模型加载和冷启动均保留；GPU子进程时间包含启动/加载/解码/评分/写盘，不能称纯前向。CPU/GPU并行，阶段耗时不能相加。']
 if complete==5400:lines.append(f'600个模型输出的九指标端到端实测吞吐：{r["scoring_outputs_per_hour"]:.1f} 输出/小时；{r["gt_frame_pairs_per_second"]:.2f} GT帧对/秒。不是5400个视频，也不是单卡吞吐。')
 lines+=['','|GPU指标|子进程墙钟之和（秒，跨卡重叠）|','|---|---:|']+[f'|{k}|{sum(v):.2f}|' for k,v in groups.items()]
 lines+=['','各CPU阶段和GPU任务原始耗时见 `../metadata/timing.json`、`../metadata/tasks/`。本批含49–121帧混合长度，不能当作纯10秒或其他机器速度保证。']
 (ROOT/'report/timing_report.md').write_text('\n'.join(lines)+'\n')

def run():
 assert not os.environ.get('SLURM_JOB_ID')
 lock=(ROOT/'logs/supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(ROOT/'metadata/preparation.json')['status']=='complete'
 assert not (ROOT/'metadata/release.json').exists(),'Allocation already released; do not relaunch'
 seconds=job_check();start=time.time();prepare_start=(ROOT/'metadata/prepare.pid').stat().st_mtime
 children=[];failure=None;validated=False;launched=False
 atomic(ROOT/'metadata/run_status.json',dict(status='running',pid=os.getpid(),started_unix=start,job=JOB))
 try:
  with (ROOT/'logs/login_cpu.log').open('ab',buffering=0) as f:
   login=subprocess.Popen([str(GTPY),'-u',str(ROOT/'scripts/login_cpu.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append(login)
  h,rem=divmod(seconds,3600);m,s=divmod(rem,60)
  args=['srun',f'--jobid={JOB}','--partition=testqueue','--exclusive','--nodes=1','--ntasks=1','--cpus-per-task=56','--gres=gpu:4','--mem=384G',f'--time={h:02}:{m:02}:{s:02}','--kill-on-bad-exit=1','bash',str(ROOT/'scripts/node.sh')]
  with (ROOT/'logs/node.log').open('ab',buffering=0) as f:
   node=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append(node);launched=True
  atomic(ROOT/'metadata/controller_processes.json',dict(supervisor=os.getpid(),login=login.pid,srun=node.pid,srun_command=args))
  while any(p.poll() is None for p in children):
   if any(p.poll() not in [None,0] for p in children):raise RuntimeError(f'Worker failed: login={login.poll()}, srun={node.poll()}')
   time.sleep(10)
  assert all(p.returncode==0 for p in children)
  subprocess.run([str(GTPY),str(ROOT/'scripts/report.py')],check=True)
  subprocess.run([str(GTPY),str(ROOT/'scripts/validate.py')],check=True)
  validated=True
 except BaseException:
  failure=traceback.format_exc();atomic(ROOT/'metadata/supervisor_failure.json',dict(error=failure,failed_unix=time.time()))
  for p in children:
   if p.poll() is None:
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:pass
  for p in children:
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  try:subprocess.run([str(GTPY),str(ROOT/'scripts/report.py')],check=True)
  except Exception:atomic(ROOT/'metadata/partial_report_failure.json',dict(error=traceback.format_exc()))
 finally:
  state='complete' if validated else 'failed'
  try:timing(start,prepare_start,time.time(),state)
  except Exception:atomic(ROOT/'metadata/timing_failure.json',dict(error=traceback.format_exc()))
  if launched:
   try:release()
   except Exception:
    state='release_failed';atomic(ROOT/'metadata/release_failure.json',dict(error=traceback.format_exc()))
  atomic(ROOT/'metadata/run_status.json',dict(status=state,validated=validated,job=JOB,started_unix=start,finished_unix=time.time(),error=failure))
 print('FINAL',state,flush=True)
 if state!='complete':raise SystemExit(1)

if __name__=='__main__':run()
