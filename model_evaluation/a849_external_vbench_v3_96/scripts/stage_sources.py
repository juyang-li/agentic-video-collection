"""CPU candidate source preparation on login; final receipts freeze later."""
from common import *
from media import video_frames,write_preview
from reuse import link
from PIL import Image,ImageOps,ImageDraw

def main():
 candidates=rows(ROOT/'metadata/candidates.jsonl');old=ROOT.parent/'a849_external_vbench_v2_96'
 for item in candidates:
  out=ROOT/'inputs/prepared'/item['id'];out.mkdir(parents=True,exist_ok=True)
  old_id=item.get('old_id') or item.get('source_old_id')
  if old_id:
   src=old/'inputs/prepared'/old_id
   for p in src.iterdir():
    if p.name!='pixels.json':link(p,out/p.name)
   r=read(src/'pixels.json');r.pop('input_manifest_sha256',None)
   r.update(reused_from_receipt=str(src/'pixels.json'),reused_from_sha256=sha(src/'pixels.json'))
  else:
   if (out/'pixels.json').exists():continue
   frames,audit=video_frames(item,item['source'])
   for i,im in enumerate(frames):im.save(out/f'{i:03d}.png')
   write_preview(frames,out/'source.mp4')
   r=dict(source_sha256=sha(item['source']['absolute_path']),pixel_audit=audit,frame_sha256=[sha(out/f'{i:03d}.png') for i in range(49)])
  r['candidate_manifest_sha256']=sha(ROOT/'metadata/candidates.jsonl');atomic(out/'pixels.json',r)
 groups=[('new_text',[r for r in candidates if r['group']=='native']),('hair_color',[r for r in candidates if r['group']=='hair_color'])]
 for label,items in groups:
  for start in range(0,len(items),8):
   chunk=items[start:start+8];sheet=Image.new('RGB',(1500,len(chunk)*185),'#20242a');d=ImageDraw.Draw(sheet)
   for j,r in enumerate(chunk):
    for k,f in enumerate([0,12,24,36,48]):
     im=Image.open(ROOT/'inputs/prepared'/r['id']/f'{f:03d}.png');x=k*300;y=j*185
     sheet.paste(ImageOps.contain(im,(295,155)),(x,y+25));d.text((x+2,y+3),f"{r['id']} {f/12:g}s",fill='white')
   sheet.save(ROOT/'review'/f'{label}_{start//8}.jpg',quality=95)
if __name__=='__main__':main()
