from common import *
import time,statistics

def report():
 states=[read(p) for p in (ROOT/'outputs').glob('*.json')]
 complete=[s for s in states if s['status']=='complete'];running=[s['id'] for s in states if s['status']=='running'];failed=[s['id'] for s in states if s['status']=='failed']
 per_group={g:[s['total_seconds'] for s in complete if s['group']==g] for g in GROUPS}
 forecast=None
 if all(per_group.values()):
  mean=statistics.mean(statistics.mean(v) for v in per_group.values());forecast={'four_gpu_clips_per_hour':3600/mean,'projected_90_request_seconds':90*mean,'note':'Estimate from completed cases, excludes setup; four GPUs cooperate on one video'}
 result={'checked_unix':time.time(),'completed':len(complete),'total':90,'running':running,'failed':failed,'group_counts':{g:len(v) for g,v in per_group.items()},'forecast':forecast}
 atomic(ROOT/'metadata/status.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':report()
