"""Reuse complete pinned metric records only for exactly matching media hashes."""
import shutil
from common import *
from score import inventory

def main():
    hashes={r['output_sha256']for r in inventory()if r['output_sha256']}
    hashes.update(sha(ROOT/'inputs/prepared'/r['id']/'source.mp4')for r in rows(ROOT/'metadata/inputs.jsonl'))
    proofs=[]
    for name in ['a849_external_vbench_v3_116','a849_v4_extensions16_20261115','a849_tripvvt_pilot32_20261115']:
        parent=ROOT.parent/name
        for kind in ['quality','static','raw']:
            for p in (parent/'metadata/vbench'/kind).glob('*.json'):
                h=p.stem.split('--')[-1]
                if h not in hashes:continue
                record=read(p)
                if record.get('status')!='complete':continue
                assert record.get('media_sha256',record.get('output_sha256'))==h
                dst=ROOT/'metadata/vbench'/kind/p.name
                if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
                assert read(dst)==record
                proofs.append(dict(path=str(dst),original_path=str(p),original_receipt_sha256=sha(p),media_sha256=h))
    atomic(ROOT/'metadata/metric_reuse.json',dict(status='passed',count=len(proofs),proofs=proofs))
    print('Reused exact media metrics',len(proofs),flush=True)

if __name__=='__main__':main()
