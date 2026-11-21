"""User revision: full original identity photo, only parsed hair filled white.

Keep provenance for the earlier smoke; reuse only byte-identical task inputs.
No source, task or prompt is changed based on model quality.
"""
import shutil
import sys
from common import *

def main():
    require_job()
    import numpy as np
    from PIL import Image,ImageDraw,ImageOps
    revision=ROOT/'metadata/before_whitehair'
    assert not revision.exists(),'Revision already applied; do not apply twice'
    revision.mkdir()
    for p in (ROOT/'metadata').iterdir():
        if p.is_file():shutil.copy2(p,revision/p.name)
    old_inputs=rows(ROOT/'metadata/inputs.jsonl');old_hash=sha(ROOT/'metadata/inputs.jsonl')
    sys.path.insert(0,str(ROOT.parent/'ltx25_vace_a100_20261111/scripts'))
    from human_parsing import HumanParser
    ready=read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json')
    parser=HumanParser(ready['human_parsing'])
    records=read(ROOT/'metadata/reference_processing.json');thumbs=[]
    names=['trump','obama','biden','harris']
    for rec in records:
        if rec.get('role')=='background':continue
        original=Image.open(rec['path']).convert('RGB')
        mask=parser.segment(original,[2])>0
        assert mask.sum()>500
        pixels=np.asarray(original).copy();pixels[mask]=255
        assert pixels.shape==np.asarray(original).shape
        assert np.array_equal(pixels[~mask],np.asarray(original)[~mask])
        assert np.all(pixels[mask]==255)
        dest=ROOT/'references'/f'{rec["name"]}_identity_whitehair_full.png'
        maskfile=dest.with_name(dest.stem+'_mask.png')
        Image.fromarray(pixels).save(dest);Image.fromarray(mask.astype(np.uint8)*255).save(maskfile)
        rec.update(face_path=str(dest),face_sha256=sha(dest),identity_reference_path=str(dest),identity_reference_sha256=sha(dest),identity_hair_mask_path=str(maskfile),original_dimensions=list(original.size),processed_dimensions=list(original.size),processing='User revision: original complete image, no crop, no resize; only semantic hair label2 RGB=255; all non-hair pixels bitwise unchanged',pixel_verification={'no_crop':True,'no_resize':True,'outside_hair_bitwise_original':True,'hair_rgb':[255,255,255]},review='pending')
        thumbs.extend([(rec['name']+' original',original),(rec['name']+' hair white; full frame',Image.fromarray(pixels))])
    atomic(ROOT/'metadata/reference_processing.json',records)
    sheet=Image.new('RGB',(1200,2*520),'#20242a');draw=ImageDraw.Draw(sheet)
    for i,(label,img) in enumerate(thumbs):
        x=i%4*300;y=i//4*520;sheet.paste(ImageOps.contain(img,(295,485)),(x,y+28));draw.text((x+3,y+4),label,fill='white')
    sheet.save(ROOT/'review/identity_whitehair_full.jpg',quality=94)
    inputs=read(revision/'freeze.json')
    new_inputs=[]
    by_name={r['name']:r for r in records}
    for item in old_inputs:
        item=json.loads(json.dumps(item))
        if item['group']=='identity':
            name=names[int(item['id'].split('_')[-1])-1];r=by_name[name]
            item['references']=[dict(absolute_path=r['identity_reference_path'],sha256=r['identity_reference_sha256'],type='identity',view='front')]
        new_inputs.append(item)
    manifest=ROOT/'metadata/inputs.jsonl'
    manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in new_inputs))
    new_hash=sha(manifest)
    inputs.update(revision='whitehair_fullframe_v2',supersedes_manifest_sha256=old_hash,input_manifest_sha256=new_hash,smoke_gate='skipped_by_explicit_user_request',references_reviewed='pending: identity_whitehair_full.jpg',user_instruction='不用smoke了，我看了看感觉很不错，开始全量吧。人脸部分记住，只是把头发部分变白，不要裁掉。')
    atomic(ROOT/'metadata/freeze.json',inputs)
    old_by_id={r['id']:r for r in old_inputs}
    archived=ROOT/'archive/smoke_before_whitehair';archived.mkdir(parents=True)
    shutil.move(str(ROOT/'outputs'),str(archived/'outputs'));(ROOT/'outputs').mkdir()
    reused=[];rebound=[]
    for item in new_inputs:
        sid=item['id'];old=old_by_id[sid]
        assert all(old[k]==item[k] for k in old if k!='references')
        for model in ['ltx_base','a849','vace']:
            origin=archived/'outputs'/model/f'{sid}.json'
            if not origin.exists() or old!=item:continue
            record=read(origin)
            if record['status']!='complete':continue
            video=origin.with_suffix('.mp4');assert sha(video)==record['output_sha256']
            out=ROOT/'outputs'/model;out.mkdir(exist_ok=True)
            for src in [origin,video]:
                dest=out/src.name
                __import__('os').link(src,dest)
            reused.append(dict(id=sid,model=model,old_manifest_sha256=old_hash,new_manifest_sha256=new_hash,task_record_bitwise_equal=True,original_generation_record=str(origin),output_sha256=record['output_sha256']))
        cache=ROOT/'cache'/sid
        for path in cache.glob('*.pt'):
            role=path.stem;receipt=path.with_suffix('.json');r=read(receipt)
            if role.startswith('ref_'):
                i=int(role.split('_')[1]);same=old['references'][i]==item['references'][i]
            elif role=='source':same=old['source']==item['source'] and old['canvas_wh']==item['canvas_wh'] and old['timestamps_seconds']==item['timestamps_seconds']
            elif role=='text':same=old['prompt']==item['prompt']
            else:raise RuntimeError(role)
            folder=revision/'cache_receipts';folder.mkdir(exist_ok=True);shutil.copy2(receipt,folder/f'{sid}_{role}.json')
            if not same:
                folder=archived/'changed_reference_cache'/sid;folder.mkdir(parents=True,exist_ok=True)
                shutil.move(str(path),str(folder/path.name));shutil.move(str(receipt),str(folder/receipt.name));continue
            assert sha(path)==r['sha256']
            r['manifest_rebind']=dict(old_manifest_sha256=old_hash,new_manifest_sha256=new_hash,role_input_equal=True,artifact_sha256=r['sha256'])
            r['identity']['manifest_sha256']=new_hash;atomic(receipt,r);rebound.append(dict(id=sid,role=role,sha256=r['sha256']))
        pixels=ROOT/'inputs/prepared'/sid/'pixels.json';r=read(pixels)
        assert r['source_sha256']==item['source']['sha256']
        r.update(input_manifest_sha256=new_hash,manifest_rebind={'old':old_hash,'new':new_hash,'source_and_time_grid_unchanged':True});atomic(pixels,r)
        condition=ROOT/'conditions'/sid/'condition.json';r=read(condition)
        saved=revision/'conditions';saved.mkdir(exist_ok=True);shutil.copy2(condition,saved/f'{sid}.json')
        assert all(old[k]==item[k] for k in ['source','mask_spec','prompt','vace_description'])
        r.update(input_manifest_sha256=new_hash,manifest_rebind={'old':old_hash,'new':new_hash,'source_mask_spec_and_prompt_unchanged':True});atomic(condition,r)
    atomic(ROOT/'metadata/reuse_outputs.json',dict(new_manifest_sha256=new_hash,old_manifest_sha256=old_hash,reused=reused,cache_rebound=rebound,note='Original generation contracts retained unchanged; reuse allowed only when entire task input record equals old record. All4 identity tasks use new full-frame white-haired references and are generated again.'))
    if (ROOT/'metadata/visual_smoke_partial.json').exists():(ROOT/'metadata/visual_smoke_partial.json').unlink()
    atomic(ROOT/'metadata/full_launch_authorization.json',dict(input_manifest_sha256=new_hash,explicit_user_skip_smoke=True,user_instruction=inputs['user_instruction'],status='authorized; awaiting reference visual verification'))
    print('whitehair revision frozen',new_hash,'reused completed outputs',len(reused),'pending',72-len(reused),flush=True)

if __name__=='__main__':main()
