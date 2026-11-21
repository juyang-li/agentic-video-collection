"""One persistent four-H200 PDD8 Ref2VA service; exact requests, resumable case outputs."""
import argparse,os,time,traceback,requests
from common import *
from contracts import digest
from h3_service import H3Service
from h3_prepare import H3,PDD

def inspect_native(path,canvas):
 import av
 with av.open(str(path))as c:
  stream=c.streams.video[0];assert stream.average_rate==24;count=0
  for frame in c.decode(video=0):assert [frame.width,frame.height]==canvas;count+=1
 assert count==107,(path,count)
 return dict(frames=count,fps=24,canvas_wh=canvas)

def run_case(item,server,runtime_sha):
 sid=item['id'];body=read(ROOT/'metadata/h3_requests'/f'{sid}.json');request_hash=digest(body)
 root=ROOT/'outputs/h3_pdd8_native';root.mkdir(parents=True,exist_ok=True);output=root/f'{sid}.mp4';receipt=output.with_suffix('.json')
 contract=dict(manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),request_sha256=request_hash,runtime_contract_sha256=runtime_sha,worker_sha256=sha(__file__),model='h3_pdd8',steps=8,seed=42,tp_size=2,ulysses_degree=2,native_frames=107,native_fps=24)
 if receipt.exists():
  prior=read(receipt)
  if prior['status']=='complete':assert prior['contract']==contract and sha(output)==prior['output_sha256'];return prior
  atomic(ROOT/'metadata/h3_attempt_history'/f"{sid}_{int(time.time()*1000)}.json",prior)
 assert body['num_inference_steps']==8 and body['target']['duration_seconds']==4 and item['prompt']['en']in body['prompt']
 attempt=ROOT/'metadata/h3_attempts'/sid/str(int(time.time()*1000));attempt.mkdir(parents=True,exist_ok=True)
 atomic(attempt/'request.json',body);record=dict(id=sid,model='h3_pdd8',status='running',contract=contract,attempt_directory=str(attempt),started_unix=time.time(),service_url=server.url,slurm_job_id=os.environ['SLURM_JOB_ID']);atomic(receipt,record)
 session=requests.Session();session.trust_env=False;start=time.monotonic()
 try:
  response=session.post(server.url+'/v1/videos',json=body,timeout=(20,240));response.raise_for_status();submission=response.json();atomic(attempt/'submission.json',submission);request_id=submission['id'];record['request_id']=request_id;atomic(receipt,record)
  deadline=time.monotonic()+3600
  while time.monotonic()<deadline:
   if server.proc.poll()is not None:raise RuntimeError('Own H3 service exited during inference')
   try:
    response=session.get(server.url+'/v1/videos/'+request_id,timeout=(10,60));response.raise_for_status();status=response.json()
   except requests.RequestException:time.sleep(5);continue
   atomic(attempt/'status.json',status)
   if status['status']=='completed':break
   if status['status']in ['failed','cancelled']:raise RuntimeError(status)
   log=(server.directory/'server.log').read_text(errors='replace')[-30000:]
   if 'torch.OutOfMemoryError'in log or 'CUDA out of memory'in log:raise RuntimeError('H3 GPU OOM; preserve logs, do not change protocol silently')
   time.sleep(3)
  else:raise TimeoutError('H3 request timed out after1h; inspect exact saved request/status')
  tmp=output.with_suffix('.part.mp4')
  with session.get(server.url+'/v1/videos/'+request_id+'/content',stream=True,timeout=(20,240))as response:
   response.raise_for_status()
   with tmp.open('wb')as f:
    for block in response.iter_content(4*1024**2):f.write(block)
  canvas=[1344,768]if item['canvas_wh'][0]>item['canvas_wh'][1]else[768,1344]
  media=inspect_native(tmp,canvas);tmp.replace(output)
  record.update(status='complete',output=str(output),output_sha256=sha(output),full_decode=media,request_seconds=time.monotonic()-start,server_inference_seconds=status.get('inference_time_s'),completed_unix=time.time(),valid_low_quality_outputs_retained=True)
 except BaseException:
  record.update(status='failed',error=traceback.format_exc(),completed_unix=time.time());atomic(receipt,record);raise
 atomic(receipt,record);print('H3 complete',sid,record['request_seconds'],flush=True);return record

def main():
 require_job();parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','all','retained'],default='all');a=parser.parse_args()
 preflight=read(ROOT/'metadata/h3_preflight.json');runtime_sha=sha(ROOT/'metadata/h3_runtime_contract.json');assert preflight['status']=='passed'and preflight['input_manifest_sha256']==sha(ROOT/'metadata/inputs.jsonl')and preflight['runtime_contract_sha256']==runtime_sha
 runtime=read(ROOT/'metadata/h3_runtime_contract.json');weights=read(H3/'metadata/weights_verified.json');assert weights['status']=='passed'and weights['revision']==runtime['base_revision']
 assert read(PDD/'pdd_config.json')==runtime['pdd_config']and sha(PDD/'pdd_fused_heads.safetensors')==runtime['fused_heads_sha256']
 from urllib.parse import urlparse,unquote
 items=rows(ROOT/'metadata/inputs.jsonl');by_id={i['id']:i for i in items};proofs={r['id']:r for r in preflight['requests']};checked=set()
 assert set(proofs)==set(by_id)
 for item in items:
  sid=item['id'];body=read(ROOT/'metadata/h3_requests'/f'{sid}.json');proof=proofs[sid]
  assert digest(body)==proof['request_sha256'] and body['conditions'][0]['type']=='video'
  source=Path(unquote(urlparse(body['conditions'][0]['uri']).path))
  if str(source)not in checked:assert sha(source)==proof['source_sha256'];checked.add(str(source))
  assert len(body['conditions'])==1+len(item['references'])
  for condition,ref in zip(body['conditions'][1:],item['references']):
   path=Path(unquote(urlparse(condition['uri']).path));assert condition['type']=='image'and path.resolve()==Path(ref['absolute_path']).resolve()and sha(path)==ref['sha256']
 pilot_ids=read(ROOT/'metadata/freeze.json')['smoke'];ordered=[by_id[sid]for sid in pilot_ids]+[i for i in items if i['id']not in pilot_ids]
 if a.stage=='retained':
  keep={f'hair_{n:02d}'for n in[1,2,3,4,6,8,9,10]};retained=[i for i in items if i['cohort']=='original96'and(i['group']!='hair'or i['id']in keep)];assert len(retained)==88
  by_id={i['id']:i for i in retained};pilot_ids=['hair_01','hair_color_02','tripvvt_03','object_removal_02','accessory_03','native_03'];ordered=[by_id[sid]for sid in pilot_ids]+[i for i in retained if i['id']not in pilot_ids]
 if a.stage=='pilot':ordered=ordered[:len(pilot_ids)]
 pending=[i for i in ordered if not (ROOT/'outputs/h3_pdd8_native'/f"{i['id']}.json").exists()or read(ROOT/'outputs/h3_pdd8_native'/f"{i['id']}.json")['status']!='complete']
 if not pending:
  for item in ordered:run_case(item,None,runtime_sha)
  atomic(ROOT/'metadata/h3_generation.json',dict(status='complete'if a.stage=='all'else'retained_complete'if a.stage=='retained'else'pilot_complete',cases=len(ordered),runtime_contract_sha256=runtime_sha,revalidated_complete_receipts=True))
  print('All requested native H3 outputs already complete',flush=True);return
 devices=os.environ['CUDA_VISIBLE_DEVICES'];assert len(devices.split(','))==4
 directory=ROOT/'logs/h3_service'/str(int(time.time()));server=H3Service(directory,devices,pdd_dir=PDD,tp_size=2,ulysses_degree=2);started=time.monotonic()
 try:
  server.start();load_seconds=time.monotonic()-started
  assert '8 fused output heads loaded'in (directory/'server.log').read_text(errors='replace'),'PDD8 output heads not evidenced in service log'
  atomic(ROOT/'metadata/h3_service_ready.json',dict(status='ready',job_id=os.environ['SLURM_JOB_ID'],service=str(directory),runtime_contract_sha256=runtime_sha,load_seconds=load_seconds,pdd_heads_log_verified=True))
  for i,item in enumerate(ordered):
   run_case(item,server,runtime_sha)
   if i+1==len(pilot_ids):
    records=[read(ROOT/'outputs/h3_pdd8_native'/f'{sid}.json')for sid in pilot_ids];assert all(r['status']=='complete'for r in records)
    atomic(ROOT/'metadata/h3_pilot.json',dict(status='technical_passed_visual_review_pending',ids=pilot_ids,outputs=[dict(id=r['id'],sha256=r['output_sha256'],request_seconds=r.get('request_seconds'))for r in records],runtime_contract_sha256=runtime_sha,validation='Complete107 frames24fps, expected canvas, exact original instruction/ref ordering and PDD8 loaded. No edit-quality filtering or regeneration.'))
 finally:server.stop()
 atomic(ROOT/'metadata/h3_generation.json',dict(status='complete'if a.stage=='all'else'retained_complete'if a.stage=='retained'else'pilot_complete',cases=len(ordered),runtime_contract_sha256=runtime_sha))
if __name__=='__main__':main()
