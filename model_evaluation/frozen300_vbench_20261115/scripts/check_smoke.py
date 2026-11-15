from common import *
import math
media,sources=smoke_media()
assert len(media)==24 and len(sources)==6
for r in media:
 for dim in DIMS:
  if dim=='temporal_flickering':continue
  p=ROOT/f'metadata/vbench/quality/{dim}--{r["output_sha256"]}.json';v=read(p);assert v['status']=='complete' and math.isfinite(v['score'])
  assert v['output_sha256']==r['output_sha256'] and v['implementation_sha256']==sha(ROOT/'metadata/implementation.json')
for r in media+sources:
 p=ROOT/f'metadata/vbench/static/{r["output_sha256"]}.json';v=read(p);assert v['status']=='complete' and type(v['is_static']) is bool
 assert v['media_sha256']==r['output_sha256'] and v['implementation_sha256']==sha(ROOT/'metadata/implementation.json')
atomic(ROOT/'metadata/gpu_smoke.json',{'status':'passed','tasks':SMOKE,'model_outputs':24,'quality_scores':120,'filter_task_media':30})
print('GPU smoke passed')
