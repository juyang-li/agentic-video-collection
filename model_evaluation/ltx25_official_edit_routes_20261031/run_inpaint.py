from pathlib import Path
import os,sys,json,time,subprocess,hashlib
import requests,av
from workflow import ROOT,graph,PROMPT
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
    old=ROOT.parent/'ltx25_singleturn_v1/runtime/ComfyUI'
    (ROOT/'runtime/comfy_route_user').mkdir(exist_ok=True)
    env=os.environ.copy();env.update(LTX_BENCH_ROOT=str(ROOT),NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost',OMP_NUM_THREADS='14',OPENBLAS_NUM_THREADS='1')
    command=[sys.executable,str(old/'main.py'),'--listen','127.0.0.1','--port','28725','--extra-model-paths-config',os.environ['LTX_MODEL_PATHS'],'--disable-auto-launch','--offline','--database-url','sqlite:///:memory:','--user-directory',str(ROOT/'runtime/comfy_route_user'),'--preview-method','none','--disable-metadata']
    http=requests.Session();http.trust_env=False;url='http://127.0.0.1:28725'
    record=ROOT/'outputs/records/inpaint.json';d=dict(status='starting',started_at=time.time(),job_id=os.environ.get('SLURM_JOB_ID'),prompt=PROMPT,route='official_inpaint_two_stage_distilled',extra_condition='manual eye-region masks, keyframes 0/24/48 interpolated',adapter='ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors',differences_from_official_gui=['stage1 768x448, stage2 1536x896','silent source: empty audio latent, no source audio reference tokens','source high resize bicubic; stage1 output upscale official Lanczos','prompt enhancer and optional I2V disabled'])
    save(record,d)
    with (ROOT/'logs/inpaint_comfy.log').open('w') as log:server=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.time()+600
        while time.time()<deadline:
            if server.poll() is not None:raise RuntimeError('Comfy server exited')
            try:
                if http.get(url+'/system_stats',timeout=3).status_code==200:break
            except requests.RequestException:pass
            time.sleep(3)
        workflow=graph();save(ROOT/'outputs/workflows/inpaint.json',workflow)
        response=http.post(url+'/prompt',json={'prompt':workflow},timeout=90)
        if response.status_code!=200:raise RuntimeError(response.text)
        pid=response.json()['prompt_id'];d.update(status='running',prompt_id=pid);save(record,d)
        deadline=time.time()+2400
        while time.time()<deadline:
            if server.poll() is not None:raise RuntimeError('Comfy server exited')
            history=http.get(url+'/history/'+pid,timeout=30).json().get(pid)
            if history:
                save(ROOT/'metadata/inpaint_history.json',history)
                if history.get('status',{}).get('status_str')=='error':raise RuntimeError(str(history['status']))
                if history.get('status',{}).get('completed'):break
            d['heartbeat']=time.time();save(record,d);time.sleep(10)
        else:raise TimeoutError('Inpaint exceeded 40 minutes')
        d['outputs']=[]
        for p in sorted((ROOT/'outputs/generated').glob('inpaint_*.mp4')):
            with av.open(str(p)) as c:
                s=c.streams.video[0];n=sum(1 for _ in c.decode(s))
                assert n==49
                d['outputs'].append(dict(file=str(p),frames=n,width=s.width,height=s.height,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        assert len(d['outputs'])==4
        d.update(status='generated',seconds=time.time()-d['started_at'],finished_at=time.time());save(record,d)
    except Exception as exc:
        d.update(status='failed',error=str(exc),finished_at=time.time());save(record,d);raise
    finally:
        server.terminate()
        try:server.wait(timeout=30)
        except subprocess.TimeoutExpired:server.kill()
if __name__=='__main__':main()
