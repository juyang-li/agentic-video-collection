"""Own one new H3 service; never connect to or stop historical services."""
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from common import ROOT,atomic
H3=ROOT.parent/'minimax_h3_ref2va90_local_20261112'
atomic_json=atomic


class H3Service:
    def __init__(self,directory,devices,pdd_dir=None,tp_size=2,ulysses_degree=2,text_encoder_cpu_offload=False):
        self.directory=Path(directory);self.devices=devices;self.proc=None;self.stream=None
        self.pdd_dir=Path(pdd_dir).resolve() if pdd_dir else None
        self.num_gpus=len(devices.split(','));self.tp_size=tp_size;self.ulysses_degree=ulysses_degree
        self.text_encoder_cpu_offload=text_encoder_cpu_offload
        if tp_size*ulysses_degree!=self.num_gpus:raise ValueError('TP × Ulysses must equal visible GPU count')
    def start(self):
        import requests
        self.directory.mkdir(parents=True,exist_ok=True)
        # H3 also binds HTTP port+1 for its result broker. Keep both below
        # Linux's ephemeral range so readiness HTTP requests cannot take it.
        port=None
        for candidate in range(20000+os.getpid()%5000,30000,2):
            sockets=[]
            try:
                for p in (candidate,candidate+1):
                    sock=socket.socket();sockets.append(sock)
                    sock.bind(('127.0.0.1',p))
                port=candidate
                break
            except OSError:
                pass
            finally:
                for sock in sockets:sock.close()
        if port is None:raise RuntimeError('No free H3 HTTP/broker port pair')
        self.url=f'http://127.0.0.1:{port}'
        # Match the tested deployment: compiled libraries live on node-local /tmp,
        # not the shared scratch filesystem. Reuse within this allocation.
        cache=Path('/tmp')/f'datamake_h3_{os.getuid()}_{os.environ.get("SLURM_JOB_ID",os.getpid())}'
        cache.mkdir(mode=0o700,parents=True,exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=self.devices,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
          HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
          TRITON_CACHE_DIR=str(cache/'triton'),FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'))
        for key in ('PYTHONPATH','http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):env.pop(key,None)
        env['PATH']=str(H3/'runtime/bin')+':'+str(H3/'runtime/env/bin')+':'+env.get('PATH','')
        cmd=[str(H3/'runtime/env/bin/sglang'),'serve','--model-path',str(H3/'runtime/models/H3/Ref2VA'),
          '--served-model-name','MiniMaxAI/MiniMax-H3','--num-gpus',str(self.num_gpus),'--tp-size',str(self.tp_size),'--ulysses-degree',str(self.ulysses_degree),
          '--encoder-parallel','auto','--performance-mode','speed','--enable-torch-compile','false','--warmup-mode','off',
          '--batching-max-size','1','--host','127.0.0.1','--port',str(port),'--output-path',str(self.directory/'outputs')]
        if self.text_encoder_cpu_offload:cmd.extend(['--text-encoder-cpu-offload','true'])
        if self.pdd_dir:
            for filename in ('pdd_config.json','pdd_fused_heads.safetensors'):
                if not (self.pdd_dir/filename).is_file():raise FileNotFoundError(self.pdd_dir/filename)
            cmd.extend(['--transformer-weights-path',str(self.pdd_dir/'transformer')])
            env['SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS']=str(self.pdd_dir/'pdd_fused_heads.safetensors')
        self.stream=(self.directory/'server.log').open('a',buffering=1)
        # Initialize the small CUDA utility library once before four ranks use it.
        subprocess.run([str(H3/'runtime/env/bin/python'),'-c',
          'from triton.backends.nvidia.driver import CudaUtils; CudaUtils()'],
          env=env,stdout=self.stream,stderr=subprocess.STDOUT,check=True,timeout=180)
        self.proc=subprocess.Popen(cmd,env=env,stdout=self.stream,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(self.directory/'service.json',{'pid':self.proc.pid,'base_url':self.url,'devices':self.devices,'command':cmd,'compile_cache':str(cache)})
        s=requests.Session();s.trust_env=False;deadline=time.monotonic()+1200
        try:
            while time.monotonic()<deadline:
                if self.proc.poll() is not None:raise RuntimeError('H3 startup failed; inspect '+str(self.directory/'server.log'))
                try:
                    r=s.get(self.url+'/v1/models',timeout=5)
                    if r.status_code==200:return self
                except requests.RequestException:pass
                time.sleep(3)
            raise TimeoutError('H3 service startup timed out')
        except BaseException:
            self.stop();raise
    def stop(self):
        if self.proc:
            try:os.killpg(self.proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:os.killpg(self.proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                self.proc.wait()
        if self.stream:self.stream.close()
