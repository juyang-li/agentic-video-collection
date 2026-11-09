from pathlib import Path
import json,os,time,subprocess,hashlib
R=Path(__file__).resolve().parent;JOB='183514'
def save(**d):
 p=R/'metadata/monitor_status.json';tmp=p.with_suffix('.part');tmp.write_text(json.dumps(dict(job=JOB,pid=os.getpid(),updated_at=time.time(),**d),indent=2)+'\n');tmp.replace(p)
def call(args):return subprocess.run(args,capture_output=True,text=True,timeout=40)
def state():
 p=call(['squeue','-h','-j',JOB,'-o','%T'])
 if p.returncode==0 and p.stdout.strip():return p.stdout.strip()
 p=call(['sacct','-X','-n','-P','-j',JOB,'-o','State'])
 return p.stdout.strip().split('|')[0] if p.returncode==0 and p.stdout.strip() else 'UNKNOWN'
def ended(js):return js.split()[0] in ['CANCELLED','TIMEOUT','FAILED','COMPLETED','NODE_FAIL','OUT_OF_MEMORY']
def launch(phase,args,ntasks=2):
 env=os.environ.copy();env['LTX_BENCH_ROOT']=str(R)
 script='source "$LTX_BENCH_ROOT/stage_job.sh"'
 if args:script+='; exec "$LTX_BENCH_ROOT/runtime/venv/bin/python" "$LTX_BENCH_ROOT/run_worker.py" '+args
 command=['srun','--jobid='+JOB,'--exact','--ntasks='+str(ntasks),'--cpus-per-task=14','--gpus-per-task=1','--gpu-bind=single:1','bash','-c',script]
 with (R/f'logs/{phase}.log').open('a') as f:
  p=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT)
  while p.poll() is None:save(phase=phase,launcher_pid=p.pid);time.sleep(15)
 return p.returncode
while True:
 js=state();ready=(R/'metadata/dev_model.json').exists();save(phase='waiting_model_and_allocation',job_state=js,model_ready=ready)
 if js=='RUNNING' and ready:break
 if ended(js):save(phase='allocation_ended',job_state=js);raise SystemExit(1)
 time.sleep(60)
weights=json.loads((R/'metadata/shared_model_lock.json').read_text());weights.insert(0,json.loads((R/'metadata/dev_model.json').read_text()))
assert all(Path(d['local']).is_file() and Path(d['local']).stat().st_size==d['bytes'] for d in weights)
(R/'metadata/model_lock.json').write_text(json.dumps(weights,indent=2)+'\n')
# Stage the large transformer once, before concurrent workers inspect the same cache.
if launch('stage',None,ntasks=1):save(phase='stage_failed');raise SystemExit(1)
round_number=1
while True:
 code=launch('attempt'+str(round_number),'--round '+str(round_number)+' --shard "$SLURM_PROCID"')
 if code:save(phase='probe_failed',attempt_round=round_number,exit_code=code);raise SystemExit(1)
 save(phase='awaiting_visual_review',attempt_round=round_number,maximum_rounds=4)
 while True:
  p=R/'metadata/decision.json';decision=json.loads(p.read_text()) if p.exists() else {}
  if decision.get('reviewed_round')==round_number:break
  js=state()
  if ended(js):save(phase='allocation_ended_during_review',attempt_round=round_number,job_state=js);raise SystemExit(1)
  time.sleep(5)
 action=decision['action']
 if action=='next':
  assert round_number<4
  round_number+=1;continue
 if action=='stop':
  call(['scancel',JOB]);save(phase='stopped_after_review',attempt_round=round_number,reason=decision.get('reason'),allocation_released=True);break
 assert action=='expand' and decision.get('target_edit_observed') is True
 probes=[json.loads(s) for s in (R/'probe_inputs.jsonl').read_text().splitlines()];winner=next(d for d in probes if d['attempt_round']==round_number and not d['control'])
 templates=[json.loads(s) for s in (R/'expansion_templates.jsonl').read_text().splitlines()];exp=[]
 for d in templates:
  config={k:winner[k] for k in ['route','attention_strength','cfg','steps','denoise'] if k in winner}
  exp.append({**d,**config,'sample_id':d['base_sample_id']+'_expand_attempt'+str(round_number),'attempt_round':round_number})
 (R/'expansion_inputs.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in exp))
 save(phase='expansion_gate_passed',attempt_round=round_number,target_edit_observed=True,review=decision.get('evidence'))
 code=launch('expand','--expand --shard "$SLURM_PROCID"')
 records=[]
 for d in exp:
  p=R/f"outputs/records/{d['sample_id']}.json";rec=json.loads(p.read_text()) if p.exists() else {}
  records.append(rec.get('status') in ['generated','recovered_tmp'] and Path(rec.get('output','')).is_file())
 if code==0 and all(records):
  call(['scancel',JOB]);save(phase='complete',attempt_round=round_number,expanded_completed=10,exit_code=0,allocation_released=True)
 else:save(phase='expansion_needs_recovery',attempt_round=round_number,expanded_completed=sum(records),exit_code=code)
 break
