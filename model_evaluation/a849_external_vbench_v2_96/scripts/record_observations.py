"""Bind manually authored direct visual observations to actual output SHA."""
import argparse
from common import *

if __name__=='__main__':
    require_job()
    p=argparse.ArgumentParser();p.add_argument('--notes',required=True,type=Path);a=p.parse_args()
    observations=read(ROOT/'metadata/visual_observations.json')
    assert observations['input_manifest_sha256']==sha(ROOT/'metadata/inputs.jsonl')
    notes=read(a.notes)
    for note in notes:
        sid=note['id'];model=note['model'];assert model in ['ltx_base','a849','vace']
        assert note['edit'] in ['failed','partial','success','uncertain'] and note['preservation'] in ['good','partial','poor']
        record=read(ROOT/'outputs'/model/f'{sid}.json');assert record['status']=='complete'
        entry={k:v for k,v in note.items() if k not in ['id','model']}
        entry.update(output_sha256=record['output_sha256'],reviewed=True,evidence=f'review/all/{sid}_temporal.jpg;9 frames0,0.5,1,1.5,2,2.5,3,3.5,4s; direct inspection',manual_notes_source=str(a.notes),manual_notes_sha256=sha(a.notes))
        observations['samples'][sid]['models'][model]=entry
    count=sum(len(r['models']) for r in observations['samples'].values())
    observations['inspection_complete']=count==288
    observations['summary']=f'Direct visual observations recorded for{count}/288 outputs. Final trend pending complete inspection and metrics.'
    atomic(ROOT/'metadata/visual_observations.json',observations)
    print('Direct observations',count,'/288',flush=True)
