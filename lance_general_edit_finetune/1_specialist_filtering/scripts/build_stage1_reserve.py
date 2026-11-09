#!/usr/bin/env python3
"""Freeze new Stage-1-only candidates, excluding history and held-out components."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_stage1_expansion_manifest import candidate_map, stratified_sample
from run_stage1_metadata import evaluate, read_csv, write_csv

ROOT = Path(__file__).resolve().parents[1]
QUOTAS = dict(goku_remove=3000, goku_add=2000, egoedit_replace=5000,
              reco_add=3000, reco_remove=3000, reco_replace=4000)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def endpoint(row, role):
    return row['dataset'] + ':' + row[role + '_locator']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=20260906)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    if (out/'manifest.csv').exists():
        raise FileExistsError('Frozen manifest already exists; use a new run directory.')
    out.mkdir(parents=True, exist_ok=True)
    config_path = ROOT/'filter_config.json'
    config = json.loads(config_path.read_text())
    config['seed'] = args.seed
    history = {}
    files = []
    for directory in [ROOT/f'artifacts/filter_run_v{v}' for v in (1, 2, 3)]:
        for path in sorted(directory.glob('*.csv')):
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                if not {'dataset', 'pair_id', 'instruction', 'source_locator', 'target_locator'} <= set(reader.fieldnames or []):
                    continue
                files.append(str(path.relative_to(ROOT)))
                for row in reader:
                    history[(row['dataset'], row['pair_id'])] = row
    # Include every initial manual survey, not merely the last survey revision.
    for path in sorted((ROOT/'artifacts').glob('survey_manifest*.csv')):
        files.append(str(path.relative_to(ROOT)))
        for row in read_csv(path):
            history[(row['dataset'], row['pair_id'])] = row
    final_path = ROOT.parent/'3_lance_lora_finetuning/artifacts/final_edit_pool_gemini_full_refill_v1/final_edit_pool_4k.csv'
    final = read_csv(final_path)
    for row in final:
        history[(row['dataset'], row['pair_id'])] = row
    print(f'history_pairs={len(history)}', flush=True)
    candidates = candidate_map(config_path, config, set(QUOTAS))
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != key:
            key, parent[key] = parent[key], root
        return root

    def union(row):
        left, right = find(endpoint(row, 'source')), find(endpoint(row, 'target'))
        if left != right:
            parent[right] = left

    for row in history.values():
        if row.get('source_locator') and row.get('target_locator'):
            union(row)
    for rows in candidates.values():
        for row in rows:
            union(row)
    blocked = {find(endpoint(row, role)) for row in final if row['split'] in {'validation', 'test'} for role in ('source', 'target')}
    prior_media = {}
    for row in history.values():
        for role in ('source', 'target'):
            if row.get(role+'_local') and row.get(role+'_locator'):
                prior_media.setdefault(endpoint(row, role), []).append(row[role+'_local'])
    selected, metadata_rejected, summary = [], [], {}
    for branch, quota in QUOTAS.items():
        eligible, counts, seen = [], Counter(), set()
        for row in candidates[branch]:
            key = (row['dataset'], row['pair_id'])
            if key in seen:
                counts['duplicate_pair'] += 1
                continue
            seen.add(key)
            if key in history:
                counts['previously_used'] += 1
                continue
            if find(endpoint(row, 'source')) in blocked:
                counts['heldout_component'] += 1
                continue
            if row['dataset'] == 'reco':
                row.update(source_provenance='generation_side_unresolved', target_provenance='generation_side_unresolved',
                           provenance_basis='official_training_roles_without_pair_lineage', quality_checked_roles='both')
            # This pass checks metadata; media existence is checked after acquisition.
            result = evaluate({**row, 'source_local': '', 'target_local': ''})
            row.update(result)
            if result['metadata_decision'] != 'accepted':
                counts['metadata_rejected'] += 1
                metadata_rejected.append(row)
                continue
            eligible.append(row)
        counts['eligible'] = len(eligible)
        chosen = stratified_sample(eligible, branch, quota, args.seed)
        selected.extend(chosen)
        summary[branch] = dict(counts)
        print(branch, dict(counts), 'selected', len(chosen), flush=True)
    reused = set()
    for rank, row in enumerate(selected, 1):
        row.update(filter_version='general_edit_stage1_quality_v2', selection_rank=str(rank),
                   selection_seed=str(args.seed), pilot_group='stage1_reserve_20k_20260906',
                   sample_id='reserve-20260906-'+row['sample_id'].removeprefix('pilot-'),
                   stage2_status='not_requested', split_eligibility='excludes_frozen_heldout_components')
        for role in ('source', 'target'):
            key = endpoint(row, role)
            path = out/'media'/f"{row['dataset']}_{digest(key)}.mp4"
            for previous in prior_media.get(key, []):
                old = ROOT/previous
                if old.is_file() and old.stat().st_size > 0:
                    path = old
                    reused.add(key)
                    break
            row[role+'_local'] = str(path.relative_to(ROOT))
            row[role+'_media_key'] = key
    assert len(selected) == sum(QUOTAS.values())
    assert len({(r['dataset'], r['pair_id']) for r in selected}) == len(selected)
    fields = list(selected[0])
    write_csv(out/'manifest.csv', selected, fields)
    if metadata_rejected:
        write_csv(out/'metadata_rejected.csv', metadata_rejected, list(metadata_rejected[0]))
    frozen = {
        'status': 'metadata_complete_media_and_gpu_pending', 'rows': len(selected), 'seed': args.seed,
        'quotas': QUOTAS, 'census': summary, 'history_manifest_paths': files,
        'history_unique_pairs': len(history), 'heldout_components': len(blocked),
        'heldout_definition': 'Full candidate and history graph; same (dataset, locator) as frozen 4K split.',
        'source_archives': dict(Counter(row['source_archive'] for row in selected)),
        'quality_checked_roles': dict(Counter(row['quality_checked_roles'] for row in selected)),
        'unique_media': len({endpoint(r,k) for r in selected for k in ('source','target')}),
        'reused_media': len(reused), 'excluded_datasets': ['coinve','insave','openve'],
        'stage2_requested': False, 'api_judge_calls': 0,
        'config': config,
        'manifest_sha256': hashlib.sha256((out/'manifest.csv').read_bytes()).hexdigest(),
    }
    (out/'selection_summary.json').write_text(json.dumps(frozen, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in frozen.items() if k not in {'history_manifest_paths','config'}}), flush=True)


if __name__ == '__main__':
    main()
