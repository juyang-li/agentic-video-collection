from common import *
inputs=rows(ROOT/'metadata/inputs.jsonl');frozen={r['id']:r for r in rows(FROZEN/'inputs.jsonl')};index=read(ROOT/'metadata/cache_index.json');checks=[]
for r in inputs:
    e=index[r['id']];receipt=e['source']['receipt'];pixels=receipt['pixel_metadata'];old=read(BASE/f'outputs/ltx_base/{r["id"]}.json')
    assert receipt['canvas_wh']==r['canvas_wh'] and receipt['num_frames']==121
    assert pixels['indices']==frozen[r['id']]['source']['sample_frame_indices'] and pixels['timestamps_seconds']==r['timestamps_seconds']
    assert old['contract']['input_manifest_sha256']==sha(BASE/'metadata/inputs.jsonl')
    assert old['contract']['reference_protocol']==PROTOCOL and old['protocol_audit']['reference_order']==[[x['type'],x['view']] for x in e['references']]
    for ref in e['references']:
        if ref['type']=='motion':
            assert ref['receipt']['num_frames']==121 and ref['receipt']['pixel_metadata']['timestamps_seconds']==r['timestamps_seconds']
    checks.append({'id':r['id'],'timeline_exact':True,'reference_order_matches_dev':True,'historical_input_manifest_matches':True})
atomic(ROOT/'metadata/timeline_audit.json',{'status':'passed','checks':checks})
print('Timeline/cache order checks passed')
