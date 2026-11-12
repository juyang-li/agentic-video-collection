from common import *
require_job(exact=True)
import av
from PIL import Image,ImageDraw
out=ROOT/'metadata/visual_checks';out.mkdir(exist_ok=True)
items=rows(ROOT/'metadata/inputs.jsonl')
for item in items:
    if item['id'] not in ['site_01','tripvvt_01','coinve_01']:continue
    path=ROOT/'review/a100_inputs_scroll_20261111/media'/f'{item["id"]}_00.mp4'
    with av.open(str(path)) as c:frames=[f.to_image() for f in c.decode(video=0)]
    sheet=Image.new('RGB',(1536,660),(24,24,24));draw=ImageDraw.Draw(sheet)
    for col,idx in enumerate([0,24,48]):
        im=frames[idx].copy();im.thumbnail((512,300));sheet.paste(im,(col*512+(512-im.width)//2,25));draw.text((col*512+5,5),f'{item["id"]} source {idx/12:.1f}s',fill='white')
    for i,r in enumerate(item['references']):
        if r['type']=='motion':continue
        im=Image.open(r['absolute_path']).convert('RGB');im.thumbnail((512,300));sheet.paste(im,(i*512+(512-im.width)//2,355));draw.text((i*512+5,335),'LEGAL REFERENCE',fill='white')
    sheet.save(out/f'{item["id"]}_inputs.jpg',quality=94)
coin=[i for i in items if i['group']=='coinve'];sheet=Image.new('RGB',(1024,5*325),(24,24,24));draw=ImageDraw.Draw(sheet)
for n,item in enumerate(coin):
    with av.open(str(ROOT/'review/a100_inputs_scroll_20261111/media'/f'{item["id"]}_00.mp4')) as c:im=next(c.decode(video=0)).to_image()
    im.thumbnail((512,300));col,row=n%2,n//2;sheet.paste(im,(col*512+(512-im.width)//2,row*325+25));draw.text((col*512+5,row*325+5),item['id'],fill='white')
sheet.save(out/'coinve_sources.jpg',quality=94)
print('Source-only input probes saved; no GT read')
