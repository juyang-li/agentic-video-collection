#!/usr/bin/env python3
"""Stream offline GPU Stage 1 scoring as reserve media arrives; never run Stage 2."""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_stage1_quality as quality
from run_stage1_metadata import read_csv, write_csv
from stage1_prefetch import prepared_jobs, score_prepared

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, data):
    temporary = path.with_suffix('.part.json')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def load_journal(path):
    result = {}
    if not path.exists():
        return result
    # A terminated process can leave only the final JSONL line incomplete.
    with path.open('rb+') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                if stream.read(1):
                    raise ValueError('journal corruption before final record')
                stream.truncate(offset)
                break
            if item['score']['technical_status'] in {'pass', 'fail_short'}:
                result[item['path']] = item
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--decode-workers', type=int, default=4)
    args = ap.parse_args()
    if args.decode_workers < 1:
        raise ValueError('decode-workers must be positive')
    out = args.run_dir.resolve()
    rows = read_csv(out/'manifest.csv')
    if (out/'completion_summary.json').exists():
        raise FileExistsError('Finalized run already exists.')
    assert len(rows)==20000 and not any(row['dataset']=='coinve' for row in rows)
    # No shared video IDs enter the old decision script: its decision cache is
    # pair-context dependent. Reuse raw metrics only, then expand to pair roles.
    jobs = quality.video_jobs(rows, role_mode='both')
    pairs = {row['pair_id']:row for row in rows}
    wanted = {}
    for job in jobs:
        path = job['local_path']
        needs_full = job['role'] in quality.required_roles(pairs[job['pair_id']])
        if path not in wanted or needs_full:
            wanted[path] = {'job':job,'full':needs_full}
    journal = out/'score_journal.jsonl'
    scored = load_journal(journal)
    scored = {p:v for p,v in scored.items() if p in wanted and (v['full'] or not wanted[p]['full'])}
    attempts = Counter()
    import torch
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated CUDA GPU is unavailable')
    models = quality.FullModels('cuda', 'facebook/dinov2-small')
    start = time.monotonic()
    resumed = len(scored)
    runtime = {'pid':os.getpid(),'host':socket.gethostname(),'slurm_job_id':os.getenv('SLURM_JOB_ID'),
               'device':str(torch.cuda.get_device_name(0)), 'stage2_requested':False,'api_judge_calls':0,
               'decode_workers':args.decode_workers,'prefetch_window':2*args.decode_workers}
    atomic_json(out/'gpu_runtime.json', runtime)

    def save_progress():
        state = {'status':'scoring','unique_videos':len(wanted),'scored_unique_videos':len(scored),
                 'resumed_unique_videos':resumed,'technical_status_counts':dict(Counter(v['score']['technical_status'] for v in scored.values())),
                 'elapsed_seconds':round(time.monotonic()-start,2), **runtime}
        atomic_json(out/'progress.json',state)
        print(json.dumps(state),flush=True)

    save_progress()
    with journal.open('a',buffering=1) as stream, ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
        consecutive_cuda_errors=0
        while True:
            state_path=out/'acquisition_state.json'
            acq=json.loads(state_path.read_text()) if state_path.exists() else {}
            acquisition_done=acq.get('status','').startswith('complete')
            advanced=0
            ready=[]
            for path,item in wanted.items():
                if path in scored:
                    continue
                local=quality.resolve_local_path(path)
                if not local.is_file() or local.stat().st_size==0:
                    continue
                if attempts[path]>=3:
                    continue
                ready.append((path,item))
            for (path,item),prepared in prepared_jobs(pool,ready,window=2*args.decode_workers):
                score=score_prepared(item['job'],prepared,models if item['full'] else None)
                attempts[path]+=1
                advanced+=1
                record={'path':path,'full':item['full'],'score':score}
                stream.write(json.dumps(record)+'\n')
                if score['technical_status']=='error':
                    if 'cuda' in score.get('error','').lower():
                        consecutive_cuda_errors+=1
                        if consecutive_cuda_errors>=3:
                            raise RuntimeError('Repeated CUDA errors; stopping without quality rejection.')
                    if attempts[path]<3:
                        continue
                else:
                    consecutive_cuda_errors=0
                scored[path]=record
                if len(scored)%100==0:
                    save_progress()
            save_progress()
            if len(scored)==len(wanted) or (acquisition_done and not advanced):
                break
            if not advanced:
                time.sleep(30)

    expanded=[]
    for job in jobs:
        record=scored.get(job['local_path'])
        if record is not None:
            expanded.append({**record['score'],**job})
    write_csv(out/'scores.csv',expanded,quality.FIELDS)
    eligible,quarantine=[],[]
    for row in rows:
        errors=[]
        for role in ('source','target'):
            record=scored.get(row[role+'_local'])
            if record is None:
                errors.append(role+':acquisition_unavailable')
            elif record['score']['technical_status']=='error':
                errors.append(role+':metric_or_decode_error')
            elif record['score']['technical_status']=='pass' and role in quality.required_roles(row):
                if not all(record['score'].get(k) for k in ('musiq_p10','sharpness_p10','dino_max_feature_jump','frame_diff_mean','frame_diff_p95')):
                    errors.append(role+':metric_incomplete')
        if errors:
            quarantine.append({**row,'quarantine_reasons':'|'.join(errors)})
        else:
            eligible.append(row)
    fields=list(rows[0])
    write_csv(out/'scored_manifest.csv',eligible,fields)
    write_csv(out/'quarantine.csv',quarantine,fields+['quarantine_reasons'])
    if eligible:
        subprocess.run([sys.executable,str(ROOT/'scripts/apply_stage1_v2_thresholds.py'),
            '--manifest',str(out/'scored_manifest.csv'),'--scores',str(out/'scores.csv'),
            '--technical-scores',str(out/'scores.csv'),'--thresholds',str(out/'thresholds.json'),
            '--policy',str(out/'policy.json'),'--video-output',str(out/'video_decisions.csv'),
            '--pair-output',str(out/'pair_decisions.csv'),'--survivors',str(out/'stage1_survivors.csv'),
            '--summary',str(out/'decision_summary.json')],check=True)
    decisions=read_csv(out/'pair_decisions.csv') if eligible else []
    summary={'status':'complete' if not quarantine else 'complete_with_quarantine',
        'input_pairs':len(rows),'decided_pairs':len(decisions),'quarantined_pairs':len(quarantine),
        'decision_counts':dict(Counter(row['stage1_1_decision'] for row in decisions)),
        'by_branch':{branch:dict(Counter(row['stage1_1_decision'] for row in decisions if row['branch']==branch)) for branch in sorted({r['branch'] for r in rows})},
        'scored_unique_media':len(scored),'unique_media':len(wanted),'pair_role_score_rows':len(expanded),
        'protocol':{'frame_samples':12,'resize':'224x224 INTER_AREA','musiq':'musiq-koniq','dino':'facebook/dinov2-small',
                    'minimum_duration_seconds':3,'policy_revision':'v2_reverse_removal_generated_side_20260905'},
        'stage2_status':'not_requested','api_judge_calls':0,'training_approved':False,**runtime}
    assert len(decisions)+len(quarantine)==len(rows)
    assert all(row['dataset']!='coinve' for row in decisions)
    atomic_json(out/'completion_summary.json',summary)
    atomic_json(out/'progress.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
