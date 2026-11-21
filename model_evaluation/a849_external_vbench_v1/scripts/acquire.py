"""Pinned official metadata and candidates. Run only in testqueue; no VLM calls."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote
import requests

ROOT = Path(__file__).resolve().parents[1]
assert os.environ.get('SLURM_JOB_PARTITION') == 'testqueue'
s = requests.Session()
s.trust_env = False
s.headers['User-Agent'] = 'A849-research-evaluation/1.0 (external video benchmark)'

def get(url):
    r = s.get(url, timeout=180)
    r.raise_for_status()
    return r

def save(url, path):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = get(url).content
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)
    print(path.name, len(data), flush=True)

if __name__ == '__main__':
    repo = 'OmniEdit-Bench/OmniEdit-Bench'
    info = get('https://huggingface.co/api/datasets/' + repo).json()
    pin = ROOT/'metadata/omni_revision.json'
    if pin.exists():
        info = json.loads(pin.read_text())
    else:
        pin.write_text(json.dumps(info, indent=2))
    rev = info['sha']
    base = f'https://huggingface.co/datasets/{repo}/resolve/{rev}/'
    files = [x['rfilename'] for x in info['siblings']]
    print('revision', rev, 'files', len(files), flush=True)
    candidates = [p for p in files if p.endswith('.json') or p == 'README.md' or (
        p.endswith('.mp4') and any('/'+g+'/' in p for g in ['Appearance_change','Clothes_change','Color_change']))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda p: save(base+quote(p), ROOT/'inputs/omni'/p), candidates))
    # Public-domain official portraits: original URL and SHA are retained.
    portraits = {
        'trump': 'https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg',
        'obama': 'https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg',
        'biden': 'https://upload.wikimedia.org/wikipedia/commons/6/68/Joe_Biden_presidential_portrait.jpg',
        'harris': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Kamala_Harris_Vice_Presidential_Portrait.jpg',
    }
    records = []
    for name, url in portraits.items():
        dst = ROOT/'references'/f'{name}_original.jpg'
        try:
            save(url, dst)
            records.append(dict(name=name, url=url, path=str(dst), sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
        except Exception as e:
            records.append(dict(name=name, url=url, error=str(e)))
    (ROOT/'metadata/portrait_sources.json').write_text(json.dumps(records, indent=2))
