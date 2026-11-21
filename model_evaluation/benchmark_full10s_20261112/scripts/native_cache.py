import sys
from common import *
sys.path.insert(0,str(NATIVE/"scripts/common"))
from project_layout import add_script_paths
add_script_paths()
def checked_load(ref):
    import torch
    p=Path(ref['path'])
    receipt=read(p.with_suffix('.json')) if ref.get('derived') else ref['receipt']['artifact']
    assert sha(p)==receipt['sha256'],f'cache changed {p}'
    return torch.load(p,map_location='cpu',weights_only=True)


def load_sample(item):
    from compact_text_cache import unpack_text_features
    entry=read(ROOT/'metadata/native_cache_index.json')[item['id']]
    assert entry['input_sha256']==sha(ROOT/'metadata/inputs.jsonl')
    source=checked_load(entry['source'])
    w,h=item['canvas_wh'];assert list(source['latents'].shape)==[128,16,h//32,w//32]
    conditions=unpack_text_features(checked_load(entry['text']),expected_contract=entry['text']['receipt']['contract'])
    refs=[{k:r[k] for k in ('type','view','index')}|{'latent':checked_load(r)} for r in entry['references']]
    assert len(refs)==len(item['references'])
    assert all(r['latent']['latents'].shape[0]==128 and r['latent']['latents'].shape[1]==(16 if r['type']=='motion' else 1) for r in refs)
    return {'source':source,'references':refs,'conditions':conditions}

