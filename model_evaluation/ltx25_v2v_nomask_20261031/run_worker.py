from pathlib import Path
import os,sys,json,time,subprocess,hashlib,traceback
import requests,av
from workflow import ROOT,graph,PROMPT,CASES
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
    tmp=p.with_suffix('.part');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def main():
    old=ROOT.parent/'ltx25_singleturn_v1/runtime/ComfyUI';(ROOT/'runtime/comfy_user').mkdir(exist_ok=True)
    env=os.environ.copy();env.update(LTX_BENCH_ROOT=str(ROOT),NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost',OMP_NUM_THREADS='14',OPENBLAS_NUM_THREADS='1')
    command=[sys.executable,str(old/'main.py'),'--listen','127.0.0.1','--port','28825','--extra-model-paths-config',os.environ['LTX_MODEL_PATHS'],'--disable-auto-launch','--offline','--database-url','sqlite:///:memory:','--user-directory',str(ROOT/'runtime/comfy_user'),'--preview-method','none','--disable-metadata']
    http=requests.Session();http.trust_env=False;url='http://127.0.0.1:28825'
    state=dict(phase='starting',job_id=os.environ.get('SLURM_JOB_ID'),completed=[],heartbeat=time.time());status=ROOT/'metadata/worker_status.json';save(status,state)
    with (ROOT/'logs/comfy.log').open('w') as log:server=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.time()+600
        while time.time()<deadline:
            if server.poll() is not None:raise RuntimeError('Comfy server exited')
            try:
                if http.get(url+'/system_stats',timeout=3).status_code==200:break
            except requests.RequestException:pass
            time.sleep(3)
        else:raise TimeoutError('Comfy startup')
        for sid,adapter in CASES:
            out=ROOT/f'outputs/generated/{sid}.mp4';record=ROOT/f'outputs/records/{sid}.json'
            if adapter and not any(p.name==adapter for p in Path(os.environ['LTX_LOCAL_MODELS']).rglob('*.safetensors')):
                save(record,dict(status='blocked_access',sample_id=sid,adapter=adapter,error='Official default Deblur adapter requires HF gated access; other cases continue.'));continue
            if record.exists() and out.exists() and json.loads(record.read_text()).get('status')=='generated':
                state['completed'].append(sid);continue
            wf=graph(sid,adapter);save(ROOT/f'outputs/workflows/{sid}.json',wf)
            d=dict(status='running',sample_id=sid,started_at=time.time(),job_id=os.environ.get('SLURM_JOB_ID'),prompt=PROMPT,adapter=adapter,adapter_strength=1.0 if adapter else None,spatial_mask=False,green_preprocessing=False,laplacian_blending=False,reference_image=None,reference_video=str(ROOT/'assets/source49.mp4'),reference_sha256=sha(ROOT/'assets/source49.mp4'),source_audio='synthetic silence encoded and frozen as official fallback',seed=43,steps=8,cfg=1.0,guide_strength=1.0,protocol='V2V_ICLoRA_single_stage_distilled',output=str(out),workflow_sha256=sha(ROOT/f'outputs/workflows/{sid}.json'),deviations=['768x448 49 frames 24FPS','optional I2V and prompt enhancer disabled','conv VAE with recommended tiled decode 768/192/128/32'])
            save(record,d);state.update(phase='inference',sample_id=sid,heartbeat=time.time());save(status,state)
            try:
                response=http.post(url+'/prompt',json={'prompt':wf},timeout=90)
                if response.status_code!=200:raise RuntimeError(response.text)
                pid=response.json()['prompt_id'];d['prompt_id']=pid;save(record,d);deadline=time.time()+1200
                while time.time()<deadline:
                    if server.poll() is not None:raise RuntimeError('Comfy server exited')
                    h=http.get(url+'/history/'+pid,timeout=30).json().get(pid)
                    if h:
                        save(ROOT/f'metadata/{sid}_history.json',h)
                        if h.get('status',{}).get('status_str')=='error':raise RuntimeError(str(h['status']))
                        if h.get('status',{}).get('completed'):break
                    state['heartbeat']=time.time();save(status,state);time.sleep(10)
                else:raise TimeoutError('sample exceeded20min')
                with av.open(str(out)) as c:
                    s=c.streams.video[0];n=sum(1 for _ in c.decode(s));assert n==49
                    d['media']=dict(frames=n,width=s.width,height=s.height,fps=float(s.average_rate))
                d.update(status='generated',output_sha256=sha(out),seconds=time.time()-d['started_at'],finished_at=time.time());save(record,d);state['completed'].append(sid)
                print(sid,'generated',d['seconds'],flush=True)
            except Exception as exc:
                d.update(status='failed',error=str(exc),traceback=traceback.format_exc());save(record,d);raise
        state.update(phase='complete',heartbeat=time.time());save(status,state)
    except Exception as exc:
        state.update(phase='failed',error=str(exc),heartbeat=time.time());save(status,state);raise
    finally:
        server.terminate()
        try:server.wait(timeout=30)
        except subprocess.TimeoutExpired:server.kill()
if __name__=='__main__':main()
