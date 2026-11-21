"""CPU: import pinned official metric records by exact media SHA only."""
import shutil
from common import *
from score import inventory

def main():
 hashes={r['output_sha256']for r in inventory()if r['output_sha256']};hashes.update(sha(ROOT/'inputs/prepared'/r['id']/'source.mp4')for r in rows(ROOT/'metadata/inputs.jsonl'));proofs=[]
 for parent_name in ['a849_external_vbench_v3_96','a849_external_vbench_v3_104']:
  parent=ROOT.parent/parent_name
  for kind in ['quality','static','raw']:
   for f in (parent/'metadata/vbench'/kind).glob('*.json'):
    h=f.stem.split('--')[-1]
    if h not in hashes:continue
    r=read(f)
    if r.get('status')!='complete':continue
    assert r.get('media_sha256',r.get('output_sha256'))==h
    dest=ROOT/'metadata/vbench'/kind/f.name
    if dest.exists():continue
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);proofs.append(dict(path=str(dest),original_path=str(f),original_receipt_sha256=sha(f),media_sha256=h))
 atomic(ROOT/'metadata/metric_reuse_latest.json',dict(status='complete',newly_reused=len(proofs),proofs=proofs))
 print('Exact media metric records reused',len(proofs),flush=True)
if __name__=='__main__':main()
