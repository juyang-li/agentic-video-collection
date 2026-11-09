#!/usr/bin/env python3
"""Create round-wise Gemini judge tasks from a completed VACE CoinVE run."""
import argparse, csv, hashlib, json
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    rows = {r['sample_id']: r for r in csv.DictReader(a.manifest.open(newline=''))}
    run = json.loads((a.run / 'run_manifest.json').read_text())
    tasks = []
    for chain in run['chains']:
        sid = chain['sample_id']
        for r in chain['rounds']:
            source = Path(r['input_video'])
            generated = Path(r['output_video'])
            if r.get('status') != 'generated' or not source.is_file() or not generated.is_file():
                continue
            task_id = f'{sid}-round-{int(r["round_index"]):02d}'
            tasks.append({
                'stage': 'coinve_vace_multiround',
                'checkpoint_step': 'wan2.1-vace-1.3b-25steps',
                'sample_id': task_id,
                'chain_sample_id': sid,
                'sample_rank': chain['sample_rank'],
                'round_index': int(r['round_index']),
                'instruction': r['instruction'],
                'source_path': str(source.resolve()),
                'source_sha256': sha256(source),
                'generated_video': str(generated.resolve()),
                'generated_sha256': sha256(generated),
                'generation_seed': run['config']['seed'] + chain['sample_rank'] * 100 + int(r['round_index']),
                'coinve_manifest_row': rows[sid],
            })
    tasks.sort(key=lambda x: (x['sample_rank'], x['round_index']))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + '\n')
    print(f'prepared {len(tasks)} round-wise tasks: {a.output}')

if __name__ == '__main__': main()
