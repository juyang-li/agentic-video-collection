"""Sequential resident-service client with durable request IDs and strict resume."""
from common import *
import argparse,time,requests,traceback
BASE='http://127.0.0.1:30011'
PILOT=['hair_color_01','single_piece_01','hair_01','site_01']

def session():
 s=requests.Session();s.trust_env=False;return s

def verify(path,canvas):
 import av
 with av.open(str(path)) as c:
  stream=c.streams.video[0];assert stream.average_rate==24
  frames=0
  for f in c.decode(video=0):
   assert (f.width,f.height)==tuple(canvas);frames+=1
  assert frames==107,(str(path),frames)
 return {'full_decode_frames':frames,'fps':24,'canvas_wh':canvas}

def main(stage):
 require_job();s=session();deadline=time.monotonic()+1800
 while True:
  try:
   r=s.get(BASE+'/v1/models',timeout=5)
   if r.ok:break
  except requests.RequestException:pass
  assert time.monotonic()<deadline,'Server did not become ready within 30 minutes'
  time.sleep(5)
 atomic(ROOT/'metadata/server_models.json',r.json())
 assert read(ROOT/'metadata/weights_verified.json')['status']=='passed'
 inputs=rows(ROOT/'metadata/inputs.jsonl')
 selected=PILOT if stage=='pilot' else [g+'_01' for g in GROUPS] if stage=='first' else [r['id'] for r in inputs]
 byid={r['id']:r for r in inputs}
 for sid in selected:
  item=byid[sid];request=read(ROOT/'metadata/requests'/f'{sid}.json')
  assert request==item['request'] and 'target_caption' not in request and item['group']!='action'
  assert sha(item['prepared_source']['path'])==item['prepared_source']['sha256']
  for ref in item['references']:assert sha(ref['absolute_path'])==ref['sha256']
  contract={'model_revision':read(ROOT/'metadata/model_download_manifest.json')['revision'],'weights_verification_sha256':sha(ROOT/'metadata/weights_verified.json'),'sglang_build_patch_sha256':sha(ROOT/'metadata/sglang_build_portability.patch'),'sglang_commit':(ROOT/'metadata/sglang_commit.txt').read_text().strip(),'requirements_sha256':sha(ROOT/'metadata/requirements.lock.txt'),'request_sha256':sha(ROOT/'metadata/requests'/f'{sid}.json'),'source_sha256':item['prepared_source']['sha256'],'reference_sha256s':[r['sha256'] for r in item['references']],'serve_script_sha256':sha(ROOT/'scripts/serve.sh'),'telemetry_script_sha256':sha(ROOT/'runtime/instrumentation/sitecustomize.py'),'topology':'TP2_Ulysses2','precision':'native_bf16_fp32','context_ir':False}
  out=ROOT/'outputs'/f'{sid}.mp4';status=out.with_suffix('.json');old=read(status) if status.exists() else None
  if old and old['status']=='complete':
   assert same_inference_contract(old['contract'],contract) and sha(out)==old['output_sha256']
   if old['contract']!=contract:atomic(ROOT/'metadata/telemetry_compatibility'/f'{sid}.json',{'previous_observer_sha256':old['contract']['telemetry_script_sha256'],'current_observer_sha256':contract['telemetry_script_sha256'],'inference_contract_unchanged':True,'output_reused':True})
   continue
  if old:
   assert old['contract']==contract,'Changed contract; archive old attempts before changing configuration'
   assert old['status']=='running',f'Failed request needs investigation: {sid}'
   state=old
  else:
   body={**request,'perf_dump_path':str(ROOT/'metadata/perf'/f'{sid}.json')}
   (ROOT/'metadata/perf').mkdir(exist_ok=True)
   state={'id':sid,'group':item['group'],'status':'running','contract':contract,'started_unix':time.time(),'posted_request':body}
   atomic(status,state)
   response=s.post(BASE+'/v1/videos',json=body,timeout=(10,180))
   atomic(ROOT/'metadata/submissions'/f'{sid}.json',{'http_status':response.status_code,'response':response.json()})
   response.raise_for_status();result=response.json();state['request_id']=result['id'];atomic(status,state)
  started=time.monotonic()
  while True:
   response=s.get(BASE+'/v1/videos/'+state['request_id'],timeout=(10,60));response.raise_for_status();result=response.json()
   atomic(ROOT/'metadata/server_tasks'/f'{sid}.json',result)
   if result['status']=='completed':break
   if result['status'] in ('failed','cancelled'):
    state.update(status='failed',server_response=result,completed_unix=time.time());atomic(status,state);raise RuntimeError(f'{sid}: {result}')
   if time.monotonic()-started>3600:raise TimeoutError(f'{sid} still active; durable request ID retained for reattachment')
   time.sleep(2)
  tmp=out.with_suffix('.part.mp4')
  with s.get(BASE+'/v1/videos/'+state['request_id']+'/content',stream=True,timeout=(10,180)) as response:
   response.raise_for_status()
   with tmp.open('wb') as f:
    for b in response.iter_content(4*1024**2):
     if b:f.write(b)
  specs=verify(tmp,item['h3_canvas_wh']);tmp.replace(out)
  state.update(status='complete',completed_unix=time.time(),total_seconds=time.time()-state['started_unix'],output_sha256=sha(out),server_response=result,**specs)
  atomic(status,state);print(json.dumps({'id':sid,'status':'complete','seconds':state['total_seconds'],'server_seconds':result.get('inference_time_s'),'server_peak_mb':result.get('peak_memory_mb')}),flush=True)
 atomic(ROOT/'metadata'/f'gate_{stage}.json',{'status':'passed','ids':selected,'scope':'identity, protocol and full decode; no human quality acceptance','completed_unix':time.time()})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','first','all'],required=True);a=p.parse_args();main(a.stage)
