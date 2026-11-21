"""Inspect local held-out candidates without mislabelling project/official splits."""
import random
from collections import Counter
from common import *

PARENT = ROOT.parent / 'a849_external_vbench_v3_116'
POOL = ROOT.parents[1] / 'dataset_research/coinve_tripvvt_11k_20261108'

def walk(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                yield from walk(v)
            else:
                yield k, v
    elif isinstance(value, list):
        for v in value:
            yield from walk(v)

def main():
    previous = rows(PARENT / 'metadata/inputs.jsonl')
    old_hashes = {r['source']['sha256'] for r in previous}
    old_uids = {r['uid'] for r in previous}
    all_rows = [r for r in rows(POOL / 'manifest.jsonl') if r['dataset'] == 'tripvvt']
    old_indices = {int(r['uid'].split(':')[-1]) for r in previous if r['group'] == 'tripvvt'}
    old_groups = {r['group_id'] for r in all_rows if r['metadata_index'] in old_indices}
    inventory = read(PARENT / 'metadata/training_overlap.json')
    training_hashes, training_groups, training_original_ids = set(), set(), set()
    for manifest in inventory['manifest_inventory']:
        path = Path(manifest['path'])
        assert sha(path) == manifest['sha256'], path
        for record in rows(path):
            for key, value in walk(record):
                if key in ('sha256', 'media_sha256', 'media_sha256_verified'):
                    training_hashes.add(value)
                if manifest['dataset'] == 'tripvvt':
                    if key == 'group_id': training_groups.add(value)
                    if key == 'original_sample_id': training_original_ids.add(value)
    eligible, excluded = [], []
    for r in all_rows:
        if r['split'] != 'test': continue
        reasons = []
        if r['source_sha256'] in old_hashes: reasons.append('existing_source_sha256')
        if f"tripvvt:{r['metadata_index']}" in old_uids: reasons.append('existing_uid')
        if r['group_id'] in old_groups: reasons.append('existing_groundtruth_group')
        if any(r[k] in training_hashes for k in ('source_sha256', 'target_sha256', 'reference_sha256')):
            reasons.append('A849_lineage_media_overlap')
        if r['group_id'] in training_groups or r['sample_id'] in training_original_ids:
            reasons.append('A849_lineage_group_or_original_id_overlap')
        if reasons: excluded.append(dict(sample_id=r['sample_id'], reasons=reasons))
        else: eligible.append(r)
    rng = random.Random(20261115)
    ordered = []
    for category, count in [('upper_body',10), ('lower_body',10), ('dresses',12)]:
        pool = sorted((r for r in eligible if r['category'] == category), key=lambda r:r['sample_id'])
        rng.shuffle(pool)
        assert len(pool) >= count, (category,len(pool))
        ordered.extend(pool[:count])
    rng.shuffle(ordered)
    provisional = [dict(r, pilot_id=f'tripvvt_{i:02d}') for i,r in enumerate(ordered,1)]
    jsonl(ROOT/'metadata/provisional32.jsonl', provisional)
    atomic(ROOT/'metadata/pool_audit.json', dict(status='scope_confirmation_pending',seed=20261115,
        project_test_count=sum(r['split']=='test' for r in all_rows), eligible=len(eligible),
        eligible_categories=dict(Counter(r['category'] for r in eligible)), excluded=excluded,
        official_split_counts=dict(Counter(r['original_metadata']['official_split'] for r in ordered)),
        warning='project split=test is group-disjoint local holdout, NOT official test; actual official split retained',
        training_manifest_inventory=inventory['manifest_inventory'],
        overlap_scope=inventory['scope'],selected_ids=[r['sample_id'] for r in ordered]))
    print('Provisional32; scope confirmation pending', Counter(r['category'] for r in ordered), flush=True)

if __name__ == '__main__': main()
