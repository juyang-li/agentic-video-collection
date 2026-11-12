"""Freeze only current formal test manifests; never inspect generated outputs."""
from collections import Counter
import random
from common import *

SEED = 20261111

def attributes(r):
    a = r['source']['attributes']
    t = r['target'].get('attributes', {})
    key = {'identity':'identity_id','action':'action_id','site':'scene_id','hair':'hair_id','hair_color':'hair_color_id','camera':'camera_id'}.get(r['axis'], 'clothing')
    return (a.get('sex_id'), a.get('aspect_id'), json.dumps(t.get(key), sort_keys=True))

def freeze():
    rng = random.Random(SEED)
    bundle = read(BUNDLE)
    pools, provenance = {}, {}
    for s in bundle['sources']:
        pool = rows(s['test_manifest'])
        assert all(r['split'] == 'test' for r in pool)
        pools[s['name']] = pool
        provenance[s['name']] = {'path':s['test_manifest'],'sha256':sha(s['test_manifest']),'count':len(pool)}
    selected, used = [], set()
    def take(pool, n, group, subgroup=None, balance=None):
        candidates = sorted(pool, key=lambda r:r['sample_id'])
        rng.shuffle(candidates)
        counts = [Counter(),Counter(),Counter()]
        for _ in range(n):
            assert candidates, (group, subgroup)
            def score(r):
                v = attributes(r)
                # Source uniqueness first; balance sex/aspect, then target diversity.
                return (r['source']['media_key'] in used, counts[0][v[0]] + counts[1][v[1]], counts[2][v[2]])
            r = min(candidates, key=score)
            candidates.remove(r)
            for c,v in zip(counts,attributes(r)): c[v] += 1
            used.add(r['source']['media_key'])
            dataset = 'ue' if group in GROUPS[:8] else group
            selected.append({'id':f'{group}_{1+sum(x["group"]==group for x in selected):02d}', 'uid':f'{dataset}:{r["sample_id"]}', 'dataset':dataset, 'group':group, 'subgroup':subgroup, 'record':r})
    ue = pools['ue']
    for group in GROUPS[:8]:
        pool = [r for r in ue if (r['mode'] == group if group in ('single_piece','whole_outfit') else r['axis']==group)]
        if group == 'single_piece':
            for slot,n in [('shirts',3),('full_body',3),('skirts',2),('pants',2)]:
                take([r for r in pool if r['sample_metadata'].get('slot')==slot],n,group,slot)
        else: take(pool,10,group)
    for edits,n in [(2,7),(3,3)]:
        take([r for r in pools['coinve'] if len(r['provenance']['instructions'])==edits],n,'coinve',str(edits)+'_edits')
    for category,n in [('upper_body',4),('lower_body',3),('dresses',3)]:
        take([r for r in pools['tripvvt'] if r['references'][0]['attributes']['category']==category],n,'tripvvt',category)
    assert len(selected)==100 and Counter(x['group'] for x in selected)==Counter(dict.fromkeys(GROUPS,10))
    assert len({x['uid'] for x in selected})==100
    for s in selected:
        r=s['record']
        for m in [r['source'],r['target'],*r['references']]:
            assert Path(m['absolute_path']).is_file(), m['absolute_path']
    jsonl(ROOT/'metadata/selection.jsonl',selected)
    # GT and target-derived attributes never enter model input manifest.
    inputs = [{k:x[k] for k in ('id','uid','dataset','group','subgroup')} | {
        'prompt':x['record']['prompt'], 'source':x['record']['source'], 'references':x['record']['references'],
        'fps':12,'num_frames':49,'timestamps_seconds':[i/12 for i in range(49)],
        'canvas_wh':[1280,736] if x['record']['source']['attributes']['aspect_id']==1 else [736,1280]
    } for x in selected]
    jsonl(ROOT/'metadata/inputs.jsonl',inputs)
    summary={'seed':SEED,'authority_bundle':str(BUNDLE),'authority_bundle_sha256':sha(BUNDLE),'pools':provenance,
        'counts':dict(Counter(x['group'] for x in selected)), 'subgroups':dict(Counter(x['subgroup'] for x in selected if x['subgroup'])),
        'unique_sources':len(used), 'ue_balance':{g:{'sex':dict(Counter(str(attributes(x['record'])[0]) for x in selected if x['group']==g)), 'aspect':dict(Counter(str(attributes(x['record'])[1]) for x in selected if x['group']==g))} for g in GROUPS[:8]},
        'time_window':'timestamps 0 through 4 seconds inclusive; 49 frames at 12fps; encoded playback 49/12 seconds',
        'sampling':'seeded greedy: unique source first, then balanced sex/aspect and target attributes; no output-dependent replacement',
        'real_source_note':'TripVVT source includes synthetic try-on; real dataset origin does not mean native live-action source',
        'camera_note':'Only first 4 seconds evaluated; a 10-second full orbit is not required to complete.'}
    atomic(ROOT/'metadata/sampling.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': freeze()
