"""Reuse direct observations only when the exact reviewed output was reused."""
from common import *
from contracts import MODELS

def main():
 items=rows(ROOT/'metadata/inputs.jsonl');dest=ROOT/'metadata/visual_observations.json';old=ROOT.parent/'a849_external_vbench_v2_96'
 if dest.exists():
  result=read(dest);assert result['input_manifest_sha256']==sha(ROOT/'metadata/inputs.jsonl')
 else:result=dict(input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),reviewer='Codex direct source/ref/output temporal inspection; no VLM API',inspection_complete=False,samples={r['id']:{'models':{},'revision_reason':r['revision_reason']} for r in items})
 previous=read(old/'metadata/visual_observations.json');n=0
 for proof in read(ROOT/'metadata/reuse_outputs.json')['reused']:
  sid=proof['id'];model=proof['model'];entry=previous['samples'][proof['old_id']]['models'][model]
  assert entry['reviewed'] is True and entry['output_sha256']==proof['output_sha256']
  if model not in result['samples'][sid]['models']:
   result['samples'][sid]['models'][model]={**entry,'visual_reused_from':str(old/'metadata/visual_observations.json'),'reuse_basis':'Exact output SHA256 and effective input/protocol verified; original observations retained.'};n+=1
 count=sum(len(r['models']) for r in result['samples'].values());result.update(inspection_complete=count==len(items)*3,summary=f'Direct observations recorded for{count}/{len(items)*3} outputs; new generation review pending where absent.')
 atomic(dest,result);print('Reused visual observations',n,'; total',count,flush=True)
if __name__=='__main__':main()
