from common import *
require_job(exact=True)
import av
import numpy as np
from PIL import Image,ImageDraw
items={i['id']:i for i in rows(ROOT/'metadata/inputs.jsonl')}
out=ROOT/'metadata/visual_checks';out.mkdir(exist_ok=True)
report=[]
for model in ('ltx_base','a849','vace'):
    for sid in ('identity_01','whole_outfit_01'):
        p=ROOT/'outputs'/model/f'{sid}_swapref.json'
        if not p.exists() or read(p).get('status')!='complete':continue
        c=read(p);original=read(ROOT/'outputs'/model/f'{sid}.json');alt=items[c['control_reference_from']]
        sheet=Image.new('RGB',(960,790),(24,24,24));d=ImageDraw.Draw(sheet)
        photos=[('original '+str(i+1),r) for i,r in enumerate(items[sid]['references'])]+[('swapped '+str(i+1),r) for i,r in enumerate(alt['references'])]
        for i,(label,r) in enumerate(photos):
            im=Image.open(r['absolute_path']).convert('RGB');im.thumbnail((240,300));sheet.paste(im,(i*240+(240-im.width)//2,25));d.text((i*240+5,5),label,fill='white')
        videos=[]
        for row,(label,stem) in enumerate([('original refs',sid),('swapped refs',sid+'_swapref')]):
            with av.open(str(ROOT/'outputs'/model/f'{stem}.mp4')) as con:frames=[f.to_image() for f in con.decode(video=0)]
            assert len(frames)==49
            videos.append(frames)
            for col,idx in enumerate((0,24,48)):
                im=frames[idx].copy();im.thumbnail((320,200));sheet.paste(im,(col*320+(320-im.width)//2,350+row*220));d.text((col*320+5,330+row*220),f'{model} {label} {idx/12:.1f}s',fill='white')
        sheet.save(out/f'{model}_{sid}_control.jpg',quality=94)
        mae=[float(np.abs(np.asarray(a).astype(np.float32)-np.asarray(b).astype(np.float32)).mean()) for a,b in zip(*videos,strict=True)]
        report.append({'model':model,'id':sid,'swapped_ref_from':alt['id'],'byte_identical':original['output_sha256']==c['output_sha256'],'mean_rgb_difference_0_255':sum(mae)/49,'note':'Semantic reference matching requires visual review; pixel difference alone is not an edit pass.'})
atomic(out/'controls_measurements.json',report)
print('Complete controls reviewed geometrically:',len(report))
