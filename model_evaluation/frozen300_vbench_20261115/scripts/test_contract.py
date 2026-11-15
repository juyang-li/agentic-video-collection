from common import *
from aggregate import static_summary
flags={(sid,m):True for sid in ['a','b'] for m in ['source']+MODELS};scores={(sid,m):.9 for sid in ['a','b'] for m in MODELS}
assert static_summary(['a','b'],flags,scores)['n']==2
flags['b',MODELS[-1]]=False;assert static_summary(['a','b'],flags,scores)['task_ids']==['a']
flags['a','source']=None;assert static_summary(['a','b'],flags,scores)['status']=='pending'
flags['a','source']=False;assert static_summary(['a','b'],flags,scores)['status']=='not_applicable'
flags['a','source']=True;scores.pop(('a',MODELS[0]));assert static_summary(['a','b'],flags,scores)['status']=='incomplete'
atomic(ROOT/'metadata/contract_tests.json',{'status':'passed','static_intersection_all_four_and_source':True,'missing_filter_not_nonstatic':True,'empty_only_after_filter_complete':True,'missing_score_not_applicable':False})
print('Contract tests passed')
