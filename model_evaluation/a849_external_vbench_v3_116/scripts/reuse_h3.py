"""Reuse H3 only on identical request/runtime digests; retain original receipts unchanged."""
from common import *
from contracts import digest
from reuse import link

def main():
 parent=ROOT.parent/'a849_external_vbench_v3_104';origins=read(ROOT/'metadata/input_origins.json');runtime=sha(ROOT/'metadata/h3_runtime_contract.json');proofs=[]
 for item in rows(ROOT/'metadata/inputs.jsonl'):
  if item['cohort']!='retained88':continue
  sid=item['id'];root,old=origins[sid];assert Path(root).resolve()==parent.resolve()
  native=parent/'outputs/h3_pdd8_native'/f'{old}.mp4';receipt=native.with_suffix('.json')
  if not receipt.exists()or read(receipt)['status']!='complete':continue
  r=read(receipt);request=digest(read(ROOT/'metadata/h3_requests'/f'{sid}.json'));assert request==digest(read(parent/'metadata/h3_requests'/f'{old}.json'))==r['contract']['request_sha256']
  assert runtime==r['contract']['runtime_contract_sha256']and sha(native)==r['output_sha256']
  for k,v in dict(steps=8,seed=42,tp_size=2,ulysses_degree=2,native_frames=107,native_fps=24).items():assert r['contract'][k]==v
  link(native,ROOT/'outputs/h3_pdd8_native'/f'{sid}.mp4');link(receipt,ROOT/'outputs/h3_pdd8_native'/f'{sid}.json')
  proof=dict(status='passed',id=sid,parent_id=old,output_sha256=r['output_sha256'],original_receipt_sha256=sha(receipt),original_receipt=str(receipt),new_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),request_sha256=request,runtime_contract_sha256=runtime,reason='Exact unchanged structured request, source adapter URI and SHA, ordered reference SHA, full English prompt, seed and PDD8 TP2/Ulysses2 protocol')
  atomic(ROOT/'metadata/h3_reuse'/f'{sid}.json',proof);proofs.append(proof)
  comparison=parent/'outputs/h3_pdd8'/f'{old}.mp4';cp=comparison.with_suffix('.json')
  if cp.exists()and read(cp)['status']=='complete':
   rr=read(cp);preview=Path(rr['browser_path']);assert rr['native_sha256']==r['output_sha256']and sha(comparison)==rr['output_sha256']and sha(preview)==rr['browser_sha256']
   dest=ROOT/'outputs/h3_pdd8'/f'{sid}.mp4'
   if not dest.exists():link(comparison,dest);link(cp,dest.with_suffix('.json'));link(preview,ROOT/'review/h3_browser'/f'{sid}.mp4')
   else:
    current=read(dest.with_suffix('.json'));assert current['status']=='complete'and current['native_sha256']==r['output_sha256']and sha(dest)==current['output_sha256']
 atomic(ROOT/'metadata/h3_reuse.json',dict(status='complete'if len(proofs)==88 else'pending',reused=len(proofs),expected=88,new_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),proofs=proofs))
 print('H3 exact request outputs reused',len(proofs),'/88',flush=True)
if __name__=='__main__':main()
