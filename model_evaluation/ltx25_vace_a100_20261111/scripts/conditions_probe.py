from common import *
require_job(exact=True)
import av
from PIL import Image,ImageDraw
out=ROOT/'metadata/visual_checks';out.mkdir(exist_ok=True)
for path in sorted((ROOT/'conditions').glob('*/condition.json')):
    c=read(path)
    if c['status']!='complete':continue
    for key in ('overlay_video','pose_video'):
        if not c.get(key):continue
        with av.open(c[key]) as con:images=[f.to_image() for f in con.decode(video=0)]
        assert len(images)==49
        sheet=Image.new('RGB',(1536,340),(24,24,24));draw=ImageDraw.Draw(sheet)
        for col,idx in enumerate([0,24,48]):
            im=images[idx].copy();im.thumbnail((512,312))
            sheet.paste(im,(col*512+(512-im.width)//2,28))
            draw.text((col*512+5,5),f'{path.parent.name} {key} {idx/12:.1f}s',fill='white')
        sheet.save(out/f'{path.parent.name}_{key}.jpg',quality=94)
print('Condition preview contact sheets saved')
