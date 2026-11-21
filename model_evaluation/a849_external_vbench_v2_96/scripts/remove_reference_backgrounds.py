"""Full-canvas person foreground; white identity hair, original-color hairstyle."""
import sys
from common import *

def main():
    require_job()
    import numpy as np
    from PIL import Image,ImageDraw,ImageOps
    sys.path.insert(0,str(ROOT.parent/'ltx25_vace_a100_20261111/scripts'))
    from human_parsing import HumanParser
    ready=read(ROOT.parent/'ltx25_vace_a100_20261111/runtime/vace_ready.json')
    parser=HumanParser(ready['human_parsing'])
    records=[];thumbs=[]
    for rec in read(ROOT/'metadata/portrait_sources.json'):
        original=Image.open(rec['path']).convert('RGB');raw=np.asarray(original)
        foreground=parser.segment(original,list(range(1,18)),main_person=True)>0
        hair=(parser.segment(original,[2])>0)&foreground
        assert foreground.mean()>.1 and hair.sum()>500
        common=np.full_like(raw,128);common[foreground]=raw[foreground]
        identity=common.copy();identity[hair]=255
        for role,pixels in [('identity',identity),('hair',common)]:
            assert pixels.shape==raw.shape and np.all(pixels[~foreground]==128)
            preserved=foreground&~hair if role=='identity' else foreground
            assert np.array_equal(pixels[preserved],raw[preserved])
            out=ROOT/'references'/f'{rec["name"]}_{role}_no_background_full.png';Image.fromarray(pixels).save(out)
            mask=out.with_name(out.stem+'_foreground.png');Image.fromarray(foreground.astype(np.uint8)*255).save(mask)
            hairmask=out.with_name(out.stem+'_hair_mask.png');Image.fromarray(hair.astype(np.uint8)*255).save(hairmask)
            records.append(dict(**rec,role=role,processed_path=str(out),processed_sha256=sha(out),foreground_mask=str(mask),foreground_mask_sha256=sha(mask),hair_mask=str(hairmask),hair_mask_sha256=sha(hairmask),original_dimensions=list(original.size),processed_dimensions=list(original.size),parser=ready['human_parsing'],processing='No crop or resize; largest parsed person foreground retained; background RGB128; identity hair RGB255 only; hair-reference original hair color',pixel_verification={'no_crop':True,'no_resize':True,'background_rgb':[128]*3,'preserved_foreground_bitwise_original':True,'identity_hair_white':role=='identity'},review='pending'))
            thumbs.append((rec['name']+' '+role+' / full canvas',Image.fromarray(pixels)))
    sheet=Image.new('RGB',(1200,1040),'#20242a');draw=ImageDraw.Draw(sheet)
    for i,(label,img) in enumerate(thumbs):
        x=i%4*300;y=i//4*520;sheet.paste(ImageOps.contain(img,(295,485)),(x,y+28));draw.text((x+3,y+4),label,fill='white')
    sheet.save(ROOT/'review/reference_background_removal.jpg',quality=95)
    atomic(ROOT/'metadata/reference_processing.json',records)
    print('Processed',len(records),'full-frame references; direct visual review pending',flush=True)

if __name__=='__main__':main()
