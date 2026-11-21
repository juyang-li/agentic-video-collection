"""CPU full-decode currently complete scoring clips; incremental evidence for final validation."""
import concurrent.futures,time
from common import *
from score import inventory
from validate import check_media

def main():
 assert not os.environ.get('SLURM_JOB_ID'),'Pure CPU media audit runs on login'
 items={r['id']:r for r in rows(ROOT/'metadata/inputs.jsonl')};complete=[r for r in inventory()if r['generation_status']=='complete'];jobs=[]
 for r in complete:
  item=items[r['id']];canvas=item['canvas_wh']if r['model']!='h3_pdd8'else([1344,768]if item['canvas_wh'][0]>item['canvas_wh'][1]else[768,1344]);jobs.append((Path(r['path']),49,12,canvas))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
  for n,path in enumerate(pool.map(check_media,jobs),1):print(n,'/',len(jobs),path,flush=True)
 atomic(ROOT/'metadata/available_media_audit.json',dict(status='passed_for_available_outputs',checked_outputs=len(jobs),expected_final_outputs=416,final_complete=len(jobs)==416,input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),completed_unix=time.time()))
if __name__=='__main__':main()
