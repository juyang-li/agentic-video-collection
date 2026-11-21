"""Choose three pairs per official synthetic SOURCE identity; never inspect model outputs."""
from build import *

def main():
 xs=rows(ROOT/'package/inputs.jsonl');rng=random.Random(SEED);counts=Counter();picked=[]
 identities=sorted({x['source']['attributes']['identity_id'] for x in xs if x['dataset']=='ue'})
 for identity in identities:
  pool=[x for x in xs if x['dataset']=='ue' and x['source']['attributes']['identity_id']==identity];rng.shuffle(pool);used=set()
  for _ in range(3):
   r=min(pool,key=lambda x:(x['group'] in used,counts[x['group']]))
   pool.remove(r);picked.append(r);used.add(r['group']);counts[r['group']]+=1
 for d in ['coinve','tripvvt']:
  pool=[x for x in xs if x['dataset']==d];rng.shuffle(pool);used=set()
  for _ in range(3):
   r=min(pool,key=lambda x:(x['subgroup'] in used,));pool.remove(r);picked.append(r);used.add(r['subgroup'])
 selection={'scope':'compact review only; frozen300 remains unchanged','seed':SEED,'source_identity_quota':3,'ue_identity_ids':identities,'counts':dict(Counter(x['dataset'] for x in picked)),'groups':dict(Counter(x['group'] for x in picked)),'ids':[x['id'] for x in picked],'real_note':'CoinVE/TripVVT have no official identity IDs; three examples per dataset do not claim identity-level coverage.'}
 p=ROOT/'metadata/compact_selection_proposed.json'
 if p.exists():assert read(p)['ids']==selection['ids'],'Frozen review selection changed'
 else:write(p,selection)
 print('REVIEW_SELECTION',len(picked),'pairs;',len(identities),'synthetic identities, 3 each')
if __name__=='__main__':main()
