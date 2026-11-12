"""Compact local progress/throughput view; no scheduler query or video re-read."""
import statistics
from common import *
report={}
for model in ('ltx_base','a849','vace'):
    states=[read(p) for p in (ROOT/'outputs'/model).glob('*.json') if not p.stem.endswith('_swapref')]
    done=[s for s in states if s['status']=='complete']
    times=[s.get('generation_seconds',s.get('denoise_seconds',0)+s.get('decode_seconds',0)) for s in done]
    report[model]={'complete':len(done),'failed':sum(s['status']=='failed' for s in states),'running':sum(s['status']=='running' for s in states)}
    if times:
        report[model].update(median_seconds=round(statistics.median(times),1),rough_remaining_4gpu_hours=round(statistics.median(times)*(100-len(done))/4/3600,2))
for kind,path in [('allocation',ROOT/'metadata/allocation_status.json'),('delivery',ROOT/'review/a100_inputs_scroll_20261111.baidu_upload.json')]:
    if path.exists():
        data=read(path)
        report[kind]={k:data[k] for k in (('state','checked_unix') if kind=='allocation' else ('status','remote')) if k in data}
print(json.dumps(report,ensure_ascii=False,indent=2))
