"""Bind manually adjudicated comparisons to the two models' actual reviewed videos."""
from common import *

def main():
    observations=read(ROOT/'metadata/visual_observations.json');items=rows(ROOT/'metadata/inputs.jsonl')
    comparisons=read(ROOT/'metadata/comparisons.json');assert set(comparisons)=={r['id']for r in items}
    cases=[]
    for item in items:
        sid=item['id'];case=dict(id=sid,uid=item['uid'],category=item['subgroup'],reviewed=True,
            comparison=comparisons[sid]['comparison'],comparison_notes=comparisons[sid]['notes'])
        for model in ['a849','h3_pdd8']:
            note=observations['samples'][sid]['models'][model];receipt=read(ROOT/'outputs'/model/f'{sid}.json')
            assert note['reviewed']is True and note['output_sha256']==receipt['output_sha256']==sha(ROOT/'outputs'/model/f'{sid}.mp4')
            assert Path(note['evidence']).exists()and sha(note['evidence'])==note['evidence_sha256']
            case[model]=dict(edit_success={'success':'successful','partial':'partial','failed':'failed'}[note['edit']],
                preservation={'good':'good','partial':'mixed','poor':'bad'}[note['preservation']],
                output_sha256=note['output_sha256'],notes=note['notes'],evidence=note['evidence'],evidence_sha256=note['evidence_sha256'])
        cases.append(case)
    atomic(ROOT/'metadata/pair_reviews.json',dict(status='complete',cases=cases,input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),
        rubric='A849 requires clear correct-person reference-garment transfer and good preservation. Missing major reference structure (sleeve, neckline, category), weak color-only changes, or unrequested garment/person/scene changes fail the gate. A849 better if successful/good while H3 edit or preservation is materially worse; comparable only when both complete the edit with comparable fidelity/preservation. H3 has no quality floor.',
        method='Direct source/reference/output inspection over nine timestamps; no VLM API; VBench never used as edit-success proxy'))
    print('32 SHA-bound pair reviews assembled',flush=True)

if __name__=='__main__':main()
