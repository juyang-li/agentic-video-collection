"""CPU: bind all existing generated videos to frozen task and condition identities."""
from common import *

def main():
 items=rows(PKG/'inputs.jsonl');manifest=sha(PKG/'inputs.jsonl');checked=[]
 assert len(items)==len({r['id'] for r in items})==300
 for item in items:
  sid=item['id'];refs=[r['sha256'] for r in item['references']]
  for model in MODELS:
   if model=='ltx_distilled_v7':p=EVAL/f'ltx25_distilled_v7_300/outputs/{sid}.json'
   elif model=='a849_v7':p=EVAL/f'frozen300_20261113/outputs/a849/{sid}.json'
   else:p=METRICS/f'remote/results/{"h3" if model=="h3_pdd8" else "vace"}/{sid}.json'
   r=read(p)
   assert r['status']=='complete' and r['id']==sid,(model,sid,'status/id')
   assert r.get('uid',item['uid'])==item['uid'],(model,sid,'uid')
   assert r.get('input_manifest_sha256',r.get('contract',{}).get('input_manifest_sha256'))==manifest,(model,sid,'manifest')
   assert r['original_prompt']==item['prompt']['en'],(model,sid,'prompt')
   assert r['source_sha256']==item['source']['sha256'],(model,sid,'source')
   assert r['ordered_reference_sha256s']==refs,(model,sid,'references')
   assert r['comparison_interval_seconds']==[item['timestamps_seconds'][0],item['timestamps_seconds'][-1]],(model,sid,'interval')
   assert sha(p.with_suffix('.mp4'))==r['output_sha256'],(model,sid,'output')
   checked.append({'id':sid,'uid':item['uid'],'model':model,'record_sha256':sha(p),'output_sha256':r['output_sha256']})
 atomic(ROOT/'metadata/pairing_audit.json',{'status':'passed','tasks':300,'model_outputs':1200,'input_manifest_sha256':manifest,'checks':['id','uid where recorded','full original prompt','source SHA','ordered reference SHAs','frozen interval','output SHA'],'records':checked})
 print('Frozen pairing audit passed: 300 tasks, 1200 outputs')

if __name__=='__main__':main()
