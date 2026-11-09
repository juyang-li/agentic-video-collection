from pathlib import Path
from collections import Counter,defaultdict
import gzip,json,random,re,statistics,hashlib
from inventory import ROOT,DATA,SOURCE

GROUPS=[('identity','身份'),('wardrobe/single_piece','单件服装'),('wardrobe/whole_outfit','整套服装'),
        ('action','动作'),('site','场景'),('hair','发型'),('hair_color','发色'),('camera','镜头')]

def load(kind):
    with gzip.open(ROOT/f'metadata/{kind}.jsonl.gz','rt') as f:return [json.loads(s) for s in f]

def group(t):return 'wardrobe/'+t['mode'] if t['axis']=='wardrobe' else t['axis']

def main():
    singles=load('single_turn');chains=load('multi_turn')
    nodes={x['id']:x for x in json.loads((DATA/'nodes/index.json').read_text())['nodes']}
    rng=random.Random(20261030);selected=[];used_edges=set();used_sources=set()
    contexts=[]
    for c in chains:
        for t in c['turns']:
            contexts.append((t,c))
    for gi,(g,label) in enumerate(GROUPS):
        counts=Counter()
        def features(t,c=None):
            a=nodes[t['source_node_id']]['attributes'];b=nodes[t['target_node_id']]['attributes']
            f=[('sex',a['sex_id']),('aspect',a['aspect_id']),('identity',a['identity_id']),
               ('scene',a['scene_id']),('camera',a['camera_id']),('action',a['action_id'])]
            key={'identity':'identity_id','hair':'hair_id','hair_color':'hair_color_id','site':'scene_id','action':'action_id','camera':'camera_id'}.get(t['axis'])
            if key:f.append(('target_attribute',b[key]))
            if t['axis']=='wardrobe':
                f.extend([('family',b['clothing']['family_id']),('slot',t.get('slot','outfit')),
                          ('ref_count',len(t['references']))])
            if c:f.extend([('chain_length',c['length']),('round',t['edit_round'])])
            return f
        def pick(pool):
            rng.shuffle(pool)
            def score(pair):
                t,c=pair
                fs=features(t,c)
                return sum((8 if k in ['sex','aspect','slot'] else 3)/(1+counts[(k,v)]) for k,v in fs) - (20 if t['source_node_id'] in used_sources else 0)
            t,c=max(pool,key=score)
            counts.update(features(t,c));eid=t.get('edge_id',t.get('id'));used_edges.add(eid);used_sources.add(t['source_node_id'])
            rid=f'review_{gi+1}_{sum(x["group"]==g for x in selected)+1:02}'
            selected.append(dict(review_id=rid,group=g,group_zh=label,kind='multi_turn' if c else 'single_turn',
                                 split=c['split'] if c else t['split'],edge_id=eid,sample=t,chain=c,
                                 source_node=nodes[t['source_node_id']],target_node=nodes[t['target_node_id']]))
        for split,n in [('train',6),('test',2)]:
            for _ in range(n):pick([(t,None) for t in singles if group(t)==g and t['split']==split and t['id'] not in used_edges])
        for kind,split in [('fork','train'),('linear','test')]:
            pool=[(t,c) for t,c in contexts if group(t)==g and c['kind']==kind and c['split']==split
                  and t['edge_id'] not in used_edges and t['edit_round']>=2
                  and (kind!='fork' or t['source_history_index']!=t['edit_round']-1)]
            assert pool,(g,kind,split)
            pick(pool)
    assert len(selected)==80
    (ROOT/'selection.json').write_text(json.dumps(dict(seed=20261030,method='8 equal task strata; each 8 single + 2 context turns; greedy attribute diversity with seeded tie-breaking; no visual-quality filtering',samples=selected),ensure_ascii=False,indent=2)+'\n')
    allturns=[t for c in chains for t in c['turns']]
    stats={}
    for name,rows in [('single_turn',singles),('multi_turn_occurrences',allturns)]:
        lang={}
        for l in ['en','zh']:
            ps=[t['prompt'].get(l,'') for t in rows];lens=sorted(len(p) for p in ps)
            lang[l]=dict(nonempty=sum(bool(p) for p in ps),unique=len(set(ps)),min_chars=min(lens),
                         median_chars=statistics.median(lens),p90_chars=lens[int(.9*(len(lens)-1))],max_chars=max(lens),
                         common_openings=Counter(re.split(r'[.!?。！？;；]',p)[0] for p in ps).most_common(8))
        stats[name]=dict(count=len(rows),languages=lang,reference_rows=sum(bool(t['references']) for t in rows),
           no_reference_rows=sum(not t['references'] for t in rows),
           source_wrapper=sum(t['prompt']['en'].startswith('Using the provided source video, apply the following edit:') for t in rows),
           preservation_keyword=sum(bool(re.search(r'\b(keep|preserve|retain|unchanged|leave)\b',t['prompt']['en'],re.I)) for t in rows),
           reference_video_mention=sum('reference video' in t['prompt']['en'].lower() for t in rows),
           reference_image_or_photo_mention=sum(bool(re.search(r'reference (?:image|photo)',t['prompt']['en'],re.I)) for t in rows))
    stats['multi_history']=dict(chains=len(chains),turn_occurrences=len(allturns),
       branch_turns=sum(t['source_history_index']!=t['edit_round']-1 for t in allturns),
       numeric_video_mentions=sum(bool(re.search(r'video\s+\d',t['prompt']['en'],re.I)) for t in allturns))
    stats['sampling']=dict(total=80,groups=dict(Counter(s['group'] for s in selected)),
      split=dict(Counter(s['split'] for s in selected)),kind=dict(Counter(s['kind'] for s in selected)),
      source_aspect=dict(Counter(nodes[s['sample']['source_node_id']]['attributes']['aspect_id'] for s in selected)),
      source_sex=dict(Counter(nodes[s['sample']['source_node_id']]['attributes']['sex_id'] for s in selected)),
      single_piece_slots=dict(Counter(s['sample'].get('slot') for s in selected if s['group']=='wardrobe/single_piece')),
      unique_edges=len(used_edges),unique_source_nodes=len(used_sources),
      multi_lengths=dict(Counter(s['chain']['length'] for s in selected if s['chain'])))
    (ROOT/'metadata/prompt_statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
