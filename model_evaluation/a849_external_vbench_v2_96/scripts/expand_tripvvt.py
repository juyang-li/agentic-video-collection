"""Keep the original six and extend each seeded garment stratum to eight."""
import random
from common import *

if __name__=='__main__':
    require_job()
    from candidates import sheet
    old=ROOT.parent/'a849_external_vbench_v1'
    prior=[r for r in rows(old/'metadata/candidates.jsonl') if r['id'].startswith('tripvvt_')]
    prior_by_source={r['record']['source']['sha256']:r for r in prior}
    source=next(r for r in read(BUNDLE)['sources'] if r['name']=='tripvvt')
    pool=rows(source['test_manifest']);rng=random.Random(20261112)
    additions=[]
    for cat in ['upper_body','lower_body','dresses']:
        group=sorted([r for r in pool if r['references'][0]['attributes']['category']==cat],key=lambda r:r['sample_id']);rng.shuffle(group)
        retained=[r['record'] for r in prior if r['record']['references'][0]['attributes']['category']==cat]
        assert {r['sample_id'] for r in group[:2]}=={r['sample_id'] for r in retained}
        used={r['source']['sha256'] for r in retained};more=[]
        for rec in group[2:]:
            key=rec['source']['sha256']
            if key in used:continue
            used.add(key);more.append(rec)
            if len(more)==6:break
        assert len(more)==6
        additions.extend(more)
    records=prior+[dict(id=f'tripvvt_{i+7:02d}',path=r['source']['absolute_path'],instruction=r['prompt']['en'],record=r) for i,r in enumerate(additions)]
    assert len(records)==24 and len({r['record']['source']['sha256'] for r in records})==24
    pages=sheet(records,'expanded_trip_candidates')
    jsonl(ROOT/'metadata/trip_candidates.jsonl',records)
    atomic(ROOT/'metadata/trip_sampling.json',dict(seed=20261112,scope=source['test_manifest'],scope_sha256=sha(source['test_manifest']),per_category=8,original_six_retained=True,selection='Original seeded first2/category plus next6 unique-source/category; chosen before inference',sheets=pages))
    print('TripVVT24 source-only sheets ready',flush=True)
