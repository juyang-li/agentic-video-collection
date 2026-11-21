"""CPU offline input review with fixed navigation, full prompts and temporal mask sheets."""
import html,argparse
from common import *
from reuse import link
from PIL import Image,ImageOps,ImageDraw
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidates',action='store_true');p.add_argument('--no-helpers',action='store_true',help='Supplemental page: main page already provides the first two helper examples');args=p.parse_args()
 manifest=ROOT/'metadata'/('candidates.jsonl' if args.candidates else 'inputs.jsonl')
 items=rows(manifest)
 if args.candidates:items=[r for r in items if not r.get('old_id')]
 dest=ROOT/'review'/('candidate_inputs' if args.candidates else 'inputs');dest.mkdir(parents=True,exist_ok=True)
 parts=[];sheets=[];seen_groups=set()
 for j,item in enumerate(items):
  if item['group'] not in seen_groups:
   parts.append(f'<h2 id="group-{item["group"]}">{item["group"]}</h2>');seen_groups.add(item['group'])
  sid=item['id'];c=read(ROOT/'conditions'/sid/'condition.json');assert c['status']=='complete'
  strip=Image.new('RGB',(1500,200),'#20242a');draw=ImageDraw.Draw(strip)
  for k,f in enumerate([0,12,24,36,48]):
   frame=Image.open(ROOT/'inputs/prepared'/sid/f'{f:03d}.png').convert('RGB');mask=np.asarray(Image.open(c['mask_frames'][f]))>127
   rgb=np.asarray(frame).copy();rgb[mask]=(rgb[mask]*.5+np.array([255,0,0])*.5).astype(np.uint8)
   strip.paste(ImageOps.contain(Image.fromarray(rgb),(295,170)),(k*300,25));draw.text((k*300+2,2),f'{sid} {f/12:g}s',fill='white')
  strip.save(dest/f'{sid}_mask.jpg',quality=96);sheets.append((sid,strip))
  link(ROOT/'inputs/prepared'/sid/'source.mp4',dest/'media'/f'{sid}_source.mp4')
  figures=[f'<figure><figcaption>SOURCE 0–4秒</figcaption><video controls muted src="media/{sid}_source.mp4"></video></figure>']
  for k,r in enumerate(item['references']):
   suffix=Path(r['absolute_path']).suffix;name=f'{sid}_ref{k}{suffix}';link(r['absolute_path'],dest/'media'/name)
   figures.append(f'<figure><figcaption>REFERENCE {html.escape(r["type"])}</figcaption><img src="media/{name}"></figure>')
  if j<2 and not args.no_helpers:
   link(c['overlay_video'],dest/'media'/f'{sid}_overlay.mp4')
   figures.append(f'<figure><figcaption>VACE mask overlay</figcaption><video controls muted src="media/{sid}_overlay.mp4"></video></figure>')
  caption=''
  if item['group']=='tripvvt':
   sel=next(r for r in rows(ROOT/'metadata/selection.jsonl') if r['id']==sid)['record'];caption='<p>官方 caption（仅审核）：'+html.escape(sel['caption'])+'</p>'
   gt=ROOT/'inputs/prepared'/sid/'gt_review_only.mp4'
   if gt.exists():
    link(gt,dest/'media'/f'{sid}_gt.mp4');figures.append(f'<figure><figcaption>GT（仅审核）</figcaption><video controls muted src="media/{sid}_gt.mp4"></video></figure>')
  parts.append(f'<article id="{sid}"><h2>{sid} · {item["group"]} / {html.escape(str(item.get("subgroup")))}</h2><p class="prompt">{html.escape(item["prompt"]["en"])}</p><p>来源：{html.escape(str(item.get("task_origin","self_created_reference_edit")))}；旧ID：{html.escape(str(item.get("old_id")))}</p><p>修改原因：{html.escape(item.get("revision_reason",""))}</p>{caption}<div class="media">'+''.join(figures)+f'</div><img class="strip" src="{sid}_mask.jpg"><a href="#top">返回顶部</a></article>')
 for n,start in enumerate(range(0,len(sheets),6)):
  chunk=sheets[start:start+6];im=Image.new('RGB',(1500,200*len(chunk)),'#20242a')
  for j,(_,strip) in enumerate(chunk):im.paste(strip,(0,j*200))
  im.save(ROOT/'review'/f'mask_review_{n:02d}.jpg',quality=96)
 nav='<nav><a href="#top">顶部</a>'+''.join(f'<a href="#group-{g}">{g}</a>' for g in dict.fromkeys(r['group'] for r in items))+''.join(f'<a href="#{r["id"]}">{r["id"]}</a>' for r in items)+'</nav>'
 page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>A849 v3 输入审核</title><style>body{font:16px/1.6 system-ui;background:#171d23;color:#eee;margin:20px;padding-top:150px}nav{position:fixed;top:0;left:0;right:0;max-height:125px;overflow:auto;background:#26323e;display:flex;flex-wrap:wrap;gap:10px;padding:10px}a{color:#9dd6ff}article{border:1px solid #65707a;padding:16px;margin:24px 0;scroll-margin-top:160px}.media{display:flex;flex-wrap:wrap;gap:12px}figure{width:310px;margin:0}figure video,figure img{width:100%;max-height:550px;object-fit:contain}.strip{width:100%;max-width:1500px}.prompt{white-space:pre-wrap;overflow-wrap:anywhere}</style><div id="top"></div>'+nav+'<h1>A849 v3 冻结输入审核</h1><p>49帧，12fps，共同0–4秒。参考前景保留原像素，Identity及Hair背景RGB128；Identity保留自然头发。无评分控件。红色仅表示VACE可编辑范围。</p>'+''.join(parts)+'</html>'
 (dest/'index.html').write_text(page)
 atomic(ROOT/'metadata/input_review_page.json',dict(path=str(dest/'index.html'),manifest_sha256=sha(manifest),tasks=len(items),mask_sheet_count=(len(items)+5)//6))
 print('Input page',dest/'index.html',flush=True)
if __name__=='__main__':main()
