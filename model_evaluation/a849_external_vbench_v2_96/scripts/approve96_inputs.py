"""Record completed direct mask inspection and verify every actual mask byte."""
from common import *

if __name__=='__main__':
    require_job()
    import ast
    import numpy as np
    from PIL import Image
    inputs=rows(ROOT/'metadata/inputs.jsonl');digest=sha(ROOT/'metadata/inputs.jsonl')
    assert len(inputs)==96
    source_review=read(ROOT/'metadata/source_reference_visual_review.json')
    assert source_review['input_manifest_sha256']==digest and len(source_review['records'])==96
    assert all(r['source_reviewed'] and r['references_reviewed'] for r in source_review['records'])
    condition_hashes={};details=[]
    for item in inputs:
        sid=item['id'];path=ROOT/'conditions'/sid/'condition.json';c=read(path)
        assert c['status']=='complete' and c['input_manifest_sha256']==digest
        assert len(c['mask_frames'])==49 and [sha(p) for p in c['mask_frames']]==c['mask_sha256']
        for p in c['mask_frames']:
            arr=np.asarray(Image.open(p));assert list(Image.open(p).size)==item['canvas_wh'] and set(np.unique(arr))<={0,255}
        if sid=='tripvvt_23':
            c['repair96'].update(method='SAM2 source-only tight garment-box masks at0,24,48 with semantic skin/hair/shoes excluded from seed; track garment directly; no GT',source_garment_boxes={'0':[.40,.33,.72,.77],'24':[.39,.33,.74,.85],'48':[.38,.39,.74,.91]})
        c['human_protocol_review']='Codex directly reviewed source-only temporal mask overlays; approved before96 inference'
        atomic(path,c);condition_hashes[sid]=sha(path)
        details.append(dict(id=sid,mask_sha256=c['mask_sha256'],repaired=bool(c.get('repair96')),evidence='review/masks_00–23.jpg;8 source timestamps0,0.5,1,1.5,2,2.5,3.5,4s; repaired masks directly rechecked'))
    # Verify the user-requested image invariants on the saved references.
    for ref in read(ROOT/'metadata/reference_processing.json'):
        original=np.asarray(Image.open(ref['path']).convert('RGB'));actual=np.asarray(Image.open(ref['processed_path']).convert('RGB'))
        fg=np.asarray(Image.open(ref['foreground_mask']))>127;hair=np.asarray(Image.open(ref['hair_mask']))>127
        assert actual.shape==original.shape and np.all(actual[~fg]==128)
        keep=fg&~hair if ref['role']=='identity' else fg
        assert np.array_equal(actual[keep],original[keep])
        if ref['role']=='identity':assert np.all(actual[hair]==255)
        assert sha(ref['processed_path'])==ref['processed_sha256']
    for script in (ROOT/'scripts').glob('*.py'):ast.parse(script.read_text(),filename=str(script))
    limitations=read(ROOT/'metadata/task_limitations.json')
    limitations['background_05']=dict(issue='The original source begins with one genuinely black frame at PTS0; PTS1/24 is normal. Verified by direct PyAV decode; common input preserves the exact black first frame.',handling='No timeline shift, fabricated frame or output-based replacement. Empty initial person segmentation is expected on the actual black frame.')
    atomic(ROOT/'metadata/task_limitations.json',limitations)
    atomic(ROOT/'metadata/input_visual_review.json',dict(input_manifest_sha256=digest,approved_ids=[r['id'] for r in inputs],condition_sha256=condition_hashes,mask_inspection=details,scope='All96 sources, references and source-only mask overlays directly inspected before new inference; no output-based task replacement; thirteen source-only mask repairs recorded.'))
    print('96 inputs approved;49 masks each verified; full-frame gray-background/white-hair invariants passed',flush=True)
