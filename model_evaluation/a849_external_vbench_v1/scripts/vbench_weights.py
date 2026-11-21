"""TLS-verified mirror fetch with pinned LFS SHA256, no metric modifications."""
import concurrent.futures
from urllib.parse import quote
import requests
from common import *

if __name__=='__main__':
    require_job()
    s=requests.Session();s.trust_env=False
    repo='dc-ai/vbench_pretrained_models'
    r=s.get('https://huggingface.co/api/models/'+repo,timeout=120);r.raise_for_status();info=r.json()
    pin=ROOT/'metadata/vbench_weight_mirror.json'
    if pin.exists():info=read(pin)
    else:atomic(pin,info)
    revision=info['sha'];files=[f['rfilename'] for f in info['siblings']]
    wanted=['dino_vitbase16_pretrain.pth','ViT-B-32.pt','ViT-L-14.pt','amt-s.pth','raft-things.pth','musiq_spaq_ckpt-358bb6af.pth','sa_0_4_vit_l_14_linear.pth']
    chosen=['dino_model/dino_vitbase16_pretrain.pth','clip_model/ViT-B-32.pt','clip_model/ViT-L-14.pt','amt_model/amt-s.pth','raft_model/models/raft-things.pth','pyiqa_model/musiq_spaq_ckpt-358bb6af.pth','aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth']
    assert set(chosen)<=set(files)
    print('mirror',revision,'selected',chosen,flush=True)
    assert len(chosen)==len(wanted)
    def fetch(path):
        url=f'https://huggingface.co/{repo}/resolve/{revision}/{quote(path)}'
        dest=ROOT/'runtime/vbench_models'/path;dest.parent.mkdir(parents=True,exist_ok=True)
        # HF exposes Git-LFS SHA256 as X-Linked-ETag on the resolve redirect.
        head=s.head(url,timeout=120,allow_redirects=False);head.raise_for_status()
        expected=head.headers.get('X-Linked-ETag','').strip('"')
        if not dest.exists():
            res=s.get(url,timeout=300,stream=True);res.raise_for_status()
            tmp=dest.with_suffix('.tmp')
            with tmp.open('wb') as out:
                for chunk in res.iter_content(8*1024*1024):out.write(chunk)
            tmp.replace(dest)
        digest=sha(dest)
        assert len(expected)==64 and digest==expected,(path,expected,digest)
        if dest.name=='dino_vitbase16_pretrain.pth':assert digest=='bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e'
        print('verified',path,flush=True)
        return dict(path=str(dest),repo_path=path,revision=revision,sha256=digest,download_url=url)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,chosen))
    atomic(ROOT/'metadata/vbench_weight_receipts.json',dict(transport='TLS verified; official DINO endpoint expired certificate; pinned SHA256-verified public mirror',files=records))
