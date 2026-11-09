"""One persistent ComfyUI process per GPU, isolated samples and durable progress."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time,uuid,traceback
import av,requests,websocket
from workflow import ROOT,graph

def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.{uuid.uuid4().hex}.part')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def media_info(path):
    try:
        with av.open(str(path)) as c:
            s=c.streams.video[0];n=sum(1 for _ in c.decode(s))
            return dict(width=s.width,height=s.height,frames=n,fps=float(s.average_rate or 24)) if n else None
    except Exception:return None
def rows():
    data=[json.loads(s) for s in (ROOT/'inputs.jsonl').read_text().splitlines()]
    byid={r['sample_id']:r for r in data}
    for i,d in enumerate(json.loads((ROOT/'reference_controls.json').read_text())):
        base=dict(byid[d['base_sample_id']]);ref=byid[d['replacement_reference_from']]
        base.update(sample_id=d['sample_id'],reference_image=ref['reference_image'],reference_sha256=ref['reference_sha256'],
                    shard=i,diagnostic=True,base_sample_id=d['base_sample_id'])
        data.append(base)
    return data

def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--smoke',action='store_true');p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    role='smoke' if a.smoke else f'shard{a.shard}'
    statusfile=ROOT/f'metadata/{role}_status.json';logpath=ROOT/f'logs/comfy_{role}_{int(time.time())}.log'
    for folder in ['logs','metadata','outputs/records','outputs/workflows','outputs/generated','runtime/comfy_user_'+role]:(ROOT/folder).mkdir(parents=True,exist_ok=True)
    port=28125+a.shard;url=f'http://127.0.0.1:{port}'
    env=os.environ.copy();env['LTX_BENCH_ROOT']=str(ROOT);env['OMP_NUM_THREADS']='14';env['OPENBLAS_NUM_THREADS']='1'
    env['NO_PROXY']=env.get('NO_PROXY','')+',127.0.0.1,localhost'
    env['no_proxy']=env['NO_PROXY']
    os.environ['NO_PROXY']=env['NO_PROXY'];os.environ['no_proxy']=env['NO_PROXY']
    http=requests.Session();http.trust_env=False
    # Local HF token is passed only in the environment for tokenizer/config retrieval.
    tokenfile=Path('/scratch/prj0000000275/jcy/.cache/huggingface/token')
    if tokenfile.exists():env['HF_TOKEN']=tokenfile.read_text().strip()
    env['HF_HOME']=str(ROOT/'runtime/hf-cache-runtime')
    command=[sys.executable,str(ROOT/'runtime/ComfyUI/main.py'),'--listen','127.0.0.1','--port',str(port),
             '--extra-model-paths-config',os.environ.get('LTX_MODEL_PATHS',str(ROOT/'runtime/model_paths.yaml')),'--disable-auto-launch','--offline',
             '--database-url','sqlite:///:memory:','--user-directory',str(ROOT/('runtime/comfy_user_'+role)),
             '--preview-method','none','--disable-metadata']
    state=dict(role=role,phase='starting_server',pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),
               heartbeat=time.time(),last_progress=time.time(),server_log=str(logpath),completed=0)
    save(statusfile,state)
    with logpath.open('w') as log:
        server=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    state['server_pid']=server.pid
    try:
        deadline=time.time()+600
        while True:
            if server.poll() is not None:raise RuntimeError(f'ComfyUI exited {server.returncode}; see {logpath}')
            try:
                probe=http.get(url+'/system_stats',timeout=3)
                state['startup_endpoint_status']=probe.status_code
                if probe.status_code==200:break
            except requests.RequestException as exc:state['startup_connection_error']=type(exc).__name__
            if time.time()>deadline:raise TimeoutError('ComfyUI startup exceeded 10 minutes')
            state.update(heartbeat=time.time());save(statusfile,state);time.sleep(3)
        selected=rows()
        if a.smoke:
            ids=['cloth_004','cloth_010','appearance_017','camera_005','cloth_004_wrong_ref']
            byid={r['sample_id']:r for r in selected};selected=[byid[i] for i in ids]
            demo=dict(byid['cloth_004']);demo.update(sample_id='official_reference_only',
                instruction='A person wearing the garment from the reference sheet stands in a simple studio and breathes naturally.',reference_only=True)
            selected.insert(0,demo)
        else:selected=[r for r in selected if r['shard']==a.shard]
        node_info=http.get(url+'/object_info',timeout=90).json()
        model_lock=json.loads((ROOT/'metadata/model_lock.json').read_text())
        io_text=(ROOT/'bench_nodes.py').read_text()
        config=dict(models=model_lock,workflow_sha256=hashlib.sha256((ROOT/'workflow.py').read_bytes()).hexdigest(),
                    preprocessing_sha256=hashlib.sha256(io_text.split('class BenchSave:')[0].encode()).hexdigest(),
                    comfy_revision=(ROOT/'metadata/comfyui_revision.txt').read_text().strip(),
                    nodes_revision=(ROOT/'metadata/ltx_nodes_revision.txt').read_text().strip())
        config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        save(ROOT/'metadata/run_config.json',{**config,'config_sha256':config_hash,'io_sha256':hashlib.sha256(io_text.encode()).hexdigest()})
        failures=[];elapsed=[]
        for r in selected:
            sid=r['sample_id'];out=ROOT/f'outputs/generated/{sid}.mp4';recordfile=ROOT/f'outputs/records/{sid}.json'
            previous=json.loads(recordfile.read_text()) if recordfile.exists() else {}
            prior_attempts=previous.get('attempts',0) if previous.get('config_sha256')==config_hash and previous.get('prompt_id') else 0
            state.update(phase='sample',sample_id=sid,heartbeat=time.time(),last_progress=time.time());save(statusfile,state)
            spec=media_info(out) if out.exists() else None
            if spec and previous.get('config_sha256')==config_hash:
                state['completed']+=1;save(statusfile,state);continue
            decoded=out.with_suffix('.decoded.pt')
            if decoded.exists() and previous.get('config_sha256')==config_hash:
                import torch
                from bench_nodes import BenchSave
                BenchSave().save(torch.load(decoded,map_location='cpu',weights_only=True),str(out),24.0)
                previous.update(status='recovered_tmp',media=media_info(out),output=str(out),recovered_from='decoded_frames')
                save(recordfile,previous);state['completed']+=1;continue
            if previous.get('status')=='failed' and prior_attempts>=2:
                failures.append(sid);continue
            if out.exists():
                old=out.with_name(out.stem+f'.previous_{int(time.time())}.mp4');out.replace(old)
            tmp=out.with_suffix('.tmp.mp4')
            if tmp.exists() and media_info(tmp) and previous.get('config_sha256')==config_hash:
                tmp.replace(out);previous.update(status='recovered_tmp',output=str(out));save(recordfile,previous)
                state['completed']+=1;continue
            workflow=graph(r,out,reference_only=r.get('reference_only',False))
            for n in workflow.values():
                if n['class_type'] not in node_info:raise RuntimeError('Required node not registered: '+n['class_type'])
            save(ROOT/f'outputs/workflows/{sid}.json',workflow)
            if a.validate_only:continue
            record=dict(sample_id=sid,task=r['task'],instruction=r['instruction'],source_video=r['source_video'],
                        reference_image=r.get('reference_image'),seed=r['seed'],config_sha256=config_hash,
                        status='running',attempts=prior_attempts,started_at=time.time(),
                        source_video_used=not r.get('reference_only',False),reference_only=r.get('reference_only',False),
                        diagnostic=r.get('diagnostic',False),output=str(out),protocol='experimental_rgb_guide_plus_optional_image')
            save(recordfile,record);client=str(uuid.uuid4());ws=None;start=time.monotonic()
            try:
                ws=websocket.create_connection(f'ws://127.0.0.1:{port}/ws?clientId={client}',timeout=2,http_no_proxy=['127.0.0.1','localhost'])
                response=http.post(url+'/prompt',json={'prompt':workflow,'client_id':client},timeout=90)
                if response.status_code!=200:raise RuntimeError('Prompt rejected: '+response.text[:6000])
                response_data=response.json();prompt_id=response_data['prompt_id'];record.update(prompt_id=prompt_id,attempts=prior_attempts+1);save(recordfile,record)
                last_heartbeat=0;last_history=0
                while True:
                    if server.poll() is not None:raise RuntimeError('ComfyUI server exited during inference')
                    try:
                        message=ws.recv()
                        if isinstance(message,str):
                            event=json.loads(message)
                            if event.get('type') in ['progress','executing','executed','execution_success']:
                                state.update(last_progress=time.time(),event=event.get('type'),detail=event.get('data'))
                    except websocket.WebSocketTimeoutException:pass
                    now=time.time()
                    if now-last_heartbeat>=30:
                        state.update(heartbeat=now);save(statusfile,state);last_heartbeat=now
                    if now-last_history>=3:
                        history=http.get(url+'/history/'+prompt_id,timeout=15).json().get(prompt_id)
                        last_history=now
                        if history:
                            if history.get('status',{}).get('status_str')=='error':raise RuntimeError(json.dumps(history.get('status'),ensure_ascii=False)[:12000])
                            if history.get('status',{}).get('completed'):break
                    if time.monotonic()-start>3600:raise TimeoutError('sample exceeded one hour')
                spec=media_info(out)
                if not spec:raise RuntimeError('No decodable video was saved')
                record.update(status='generated',seconds=time.monotonic()-start,media=spec,finished_at=time.time())
                elapsed.append(record['seconds']);state['completed']+=1
            except Exception as exc:
                record.update(status='failed',error_type=type(exc).__name__,error=str(exc),seconds=time.monotonic()-start,finished_at=time.time())
                failures.append(sid)
            finally:
                if ws:ws.close()
                save(recordfile,record);state.update(heartbeat=time.time(),last_progress=time.time());save(statusfile,state)
                print(role,sid,record['status'],round(record.get('seconds',0),1),flush=True)
        state.update(phase='complete' if not failures else 'failed_items',failures=failures,heartbeat=time.time(),
                     max_sample_seconds=max(elapsed,default=0),validation_only=a.validate_only)
        save(statusfile,state)
        if failures:sys.exit(2)
    except Exception as exc:
        state.update(phase='crashed',error_type=type(exc).__name__,error=str(exc),heartbeat=time.time());save(statusfile,state)
        raise
    finally:
        server.terminate()
        try:server.wait(timeout=30)
        except subprocess.TimeoutExpired:server.kill()

if __name__=='__main__':main()
