import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path('/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/ltx2.5')
BUNDLE = NATIVE / 'data/mixed/outfit_camera_720p/sources.json'
CHECKPOINT = NATIVE / 'experiments/training/mixed720p_sequential_visual_dropout/job_197731/checkpoints/lora_weights_step_00849.safetensors'
CONFIG = NATIVE / 'configs/training/mixed720p_sequential_visual_dropout/train.yaml'
GROUPS = ['identity', 'single_piece', 'whole_outfit', 'action', 'site', 'hair', 'hair_color', 'camera', 'coinve', 'tripvvt']
LABELS = dict(zip(GROUPS, ['身份','单件服装','整套服装','动作','场景','发型','发色','镜头','CoinVE 复合编辑','TripVVT 参考图换衣']))

def read(path):
    return json.loads(Path(path).read_text())

def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def jsonl(path, records):
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f'Frozen manifest differs: {path}')
    path.write_text(text)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def require_job(exact=False):
    if os.environ.get('SLURM_JOB_PARTITION') != 'testqueue':
        raise RuntimeError('This entry requires SLURM_JOB_PARTITION=testqueue')
    if exact and os.environ.get('SLURM_JOB_ID') != '199052':
        raise RuntimeError('Use the user supplied allocation 199052')
