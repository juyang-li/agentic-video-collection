from common import *
import argparse,concurrent.futures as cf,traceback
def one(case):
 p=path_score('vbench',case,'temporal_flickering')
 if compatible(p,case):return
 try:
  import cv2
  cv2.setNumThreads(1)
  from vbench.temporal_flickering import cal_score
  start=time.perf_counter();v=float(cal_score(case['path']));assert math.isfinite(v)
  atomic(p,receipt(case,status='complete',dimension='temporal_flickering',raw_score=v,score=v,seconds=time.perf_counter()-start,official_function='vbench.temporal_flickering.cal_score',static_filter='not_requested'))
 except Exception:failures('vbench',case,traceback.format_exc(),'temporal_flickering')
def main():
 require_node();p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','full'],required=True);a=p.parse_args();start=time.time()
 selected=cases_for(a.phase)
 with cf.ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(one,selected))
 ok=all(compatible(path_score('vbench',r,'temporal_flickering'),r) for r in selected)
 atomic(ROOT/f'metadata/raw_{a.phase}.json',{'status':'complete' if ok else 'failed','outputs':len(selected),'workers':2,'seconds':time.time()-start})
 if not ok:raise RuntimeError('Raw flicker scoring failed')
if __name__=='__main__':main()
