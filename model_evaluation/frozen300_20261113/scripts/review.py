"""Offline, unscored scroll review with common-clock H264 browser copies. CPU/login only."""
from build import *
import html

LABELS=dict(zip(GROUPS+['coinve','tripvvt'],['身份','单件服装','整套服装','动作','场景','发型','发色','镜头','CoinVE 复合编辑','TripVVT 参考换衣']))
def esc(v):return html.escape(str(v),quote=True)

def preview(job):
 import av
 from PIL import Image,ImageOps
 item,m,indices,role=job;p=ROOT/'package';out=p/'review_media'/f'{item["id"]}_{role}.mp4';poster=out.with_suffix('.jpg');receipt=ROOT/'metadata/review_previews'/f'{out.stem}.json'
 identity={'input_manifest_sha256':sha(p/'inputs.jsonl'),'media_sha256':m['sha256'],'indices':indices,'fps':12,'max_edge':640,'encoder':'libx264','crf':'23','preset':'veryfast','pixel_contract':'display_frame_contain_640_v1'}
 if receipt.exists():
  r=read(receipt)
  if r['identity']==identity and sha(out)==r['sha256'] and poster.exists():return r
 original=p/m['path'];assert sha(original)==m['sha256']
 w,h=item['canvas_wh'];canvas=(640,368) if w>h else (368,640)
 wanted=set(indices);images={}
 with av.open(str(original)) as c:
  c.streams.video[0].codec_context.thread_count=1
  for i,f in enumerate(c.decode(video=0)):
   if i in wanted:
    scale=min(canvas[0]/f.width,canvas[1]/f.height);rw,rh=max(1,round(f.width*scale)),max(1,round(f.height*scale))
    im=f.reformat(width=rw,height=rh,format='rgb24',interpolation='BILINEAR').to_image();im2=Image.new('RGB',canvas);im2.paste(im,((canvas[0]-rw)//2,(canvas[1]-rh)//2));images[i]=im2
   if i>=max(wanted):break
 assert set(images)==wanted
 tmp=out.with_suffix('.part.mp4')
 with av.open(str(tmp),'w',options={'movflags':'+faststart'}) as c:
  s=c.add_stream('libx264',rate=12);s.width,s.height=canvas;s.pix_fmt='yuv420p';s.thread_count=1;s.options={'crf':'23','preset':'veryfast'}
  for i in indices:
   for packet in s.encode(av.VideoFrame.from_image(images[i])):c.mux(packet)
  for packet in s.encode():c.mux(packet)
 images[indices[0]].save(poster,quality=80)
 with av.open(str(tmp)) as c:
  stream=c.streams.video[0];assert stream.average_rate==12 and stream.codec_context.name=='h264';n=0
  for f in c.decode(video=0):assert (f.width,f.height)==canvas;n+=1
  assert n==item['num_frames']
 tmp.replace(out)
 r={'id':item['id'],'role':role,'identity':identity,'path':out.relative_to(p).as_posix(),'poster':poster.relative_to(p).as_posix(),'sha256':sha(out),'full_decode_frames':n,'canvas_wh':list(canvas),'original_path':m['path'],'browser_copy_only':True}
 write(receipt,r);return r

def build(compact=False):
 assert not os.environ.get('SLURM_JOB_ID'),'CPU review preparation runs on login'
 p=ROOT/'package';(p/'review_media').mkdir(exist_ok=True)
 inputs=rows(p/'inputs.jsonl')
 if compact:
  chosen=read(ROOT/'metadata/compact_selection_proposed.json')['ids'];inputs=[a for a in inputs if a['id'] in chosen]
 gts={x['id']:x for x in rows(p/'ground_truth.jsonl')}
 jobs=[]
 for a in inputs:
  b=gts[a['id']]
  jobs.extend([(a,a['source'],a['source']['sample_frame_indices'],'source'),(a,b['target'],b['sample_frame_indices'],'gt')])
  jobs.extend((a,m,m['sample_frame_indices'],f'ref_{i}') for i,m in enumerate(a['references']) if m['type']=='motion')
 records=[]
 with cf.ThreadPoolExecutor(max_workers=4) as pool:
  for i,r in enumerate(pool.map(preview,jobs)):
   records.append(r)
   if i%25==0:print('review browser copies',i+1,'/',len(jobs),flush=True)
 bykey={(r['id'],r['role']):r for r in records}
 write(p/'review_manifest.json',{'schema':'frozen300_review_v1','input_manifest_sha256':sha(p/'inputs.jsonl'),'samples':len(inputs),'clips':records,'full_decode_passed':True,'human_quality_review':'not_performed','scoring_ui':False})
 def video(a,role,label):
  r=bykey[(a['id'],role)]
  return f'<figure><figcaption>{esc(label)} · <a href="{esc(r["original_path"])}">完整原件</a></figcaption><video class="clip" controls preload="none" playsinline poster="{esc(r["poster"])}" src="{esc(r["path"])}"></video></figure>'
 sections=[]
 for group,label in LABELS.items():
  cards=[]
  for a in inputs:
   if a['group']!=group:continue
   b=gts[a['id']];w=a['temporal_window'];n=a['num_frames']
   refs=[]
   for i,m in enumerate(a['references']):
    if m['type']=='motion':refs.append(video(a,f'ref_{i}',f'REFERENCE {i+1} · motion'))
    else:refs.append(f'<figure><figcaption>REFERENCE {i+1} · {esc(m["type"])} · {esc(m.get("view",""))}</figcaption><a href="{esc(m["path"])}"><img loading="lazy" src="{esc(m["path"])}" alt="{esc(a["id"])} reference {i+1}"></a></figure>')
   if not refs:refs=['<div class="empty">无参考输入</div>']
   prompts='<div class="prompt"><b>完整编辑指令 · English</b><pre>'+esc(a['prompt']['en'])+'</pre>'
   if a['prompt'].get('zh'):prompts+='<details><summary>中文完整指令</summary><pre>'+esc(a['prompt']['zh'])+'</pre></details>'
   prompts+='</div>'
   caption='<details class="caption" open><summary>审核用官方 caption（不进入推理）</summary><p>'+esc(b['caption'])+'</p></details>' if b.get('caption') else ''
   media='<div class="media">'+video(a,'source','SOURCE')+'<div class="refs">'+''.join(refs)+'</div>'+video(a,'gt','GT · 仅审核')+'</div>'
   control=f'<div class="playback"><button class="play-pair">同步播放</button><button class="pause-pair">暂停</button><button class="reset-pair">回到起点</button><label>共同时间 <input class="seek" type="range" min="0" max="{(n-1)/12:.9f}" step="0.083333333333" value="0"></label><output>0.00 s</output></div>'
   old='旧100保留' if a['retained_from_old100'] else '新增正式test'
   cards.append(f'<article id="{esc(a["id"])}" data-search="{esc(a["id"]+" "+a["uid"]+" "+a["prompt"]["en"])}"><h3>{esc(a["id"])} <span>{esc(a["uid"])} · {old}</span><a class="top" href="#top">↑ 顶部</a></h3><p class="meta">{w["start_seconds"]:.3f}–{w["end_seconds"]:.3f} s · {n}帧 / 12fps · {a["canvas_wh"][0]}×{a["canvas_wh"][1]} · 有序参考 {len(a["references"])} 项</p>'+prompts+caption+media+control+'</article>')
  sections.append(f'<section id="group-{group}"><h2>{esc(label)} · {len(cards)} 对</h2>'+''.join(cards)+'</section>')
 nav=''.join(f'<a href="#group-{g}">{esc(l)}</a>' for g,l in LABELS.items())
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>300对冻结测评集 · 审核</title><style>
 :root{color-scheme:light;font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:#1d2939;background:#f3f5f8}*{box-sizing:border-box}body{margin:0}nav{position:sticky;top:0;z-index:20;padding:10px 18px;background:#142338;color:white;display:flex;gap:8px;flex-wrap:wrap;align-items:center}nav a{color:#fff;text-decoration:none;border:1px solid #526176;padding:4px 9px;border-radius:5px}nav a:hover{background:#37516e}nav input{padding:7px;border-radius:5px;border:0;min-width:220px}main{max-width:1680px;margin:auto;padding:20px}h1{margin-top:0}section,article{scroll-margin-top:140px}section>h2{padding:14px 4px 4px}article{background:white;border:1px solid #d5dde8;border-radius:10px;padding:18px;margin:0 0 22px}h3{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:0}h3 span,.meta{color:#637083;font-size:14px}.top{margin-left:auto}a{color:#1461ad}.prompt{background:#f1f5fc;border-left:4px solid #376baa;padding:12px;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;margin:6px 0}.caption{border:1px solid #d5dde8;padding:8px 12px;margin:10px 0}.caption p{white-space:pre-wrap;margin:8px 0}.media{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;align-items:start}figure{margin:0;min-width:0}figcaption{font-weight:600;font-size:14px;margin-bottom:7px}video{display:block;width:100%;max-height:620px;background:#111;border-radius:5px}img{display:block;max-width:100%;max-height:440px;object-fit:contain;margin:auto}.refs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.refs>figure:only-child{grid-column:1/-1}.empty{padding:24px;color:#637083;text-align:center;background:#f5f6f8}.playback{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:14px}.playback label{display:flex;align-items:center;gap:8px;flex:1}.seek{min-width:180px;flex:1}button{border:1px solid #b4c3d5;border-radius:5px;background:#f0f5fc;padding:7px 12px;cursor:pointer}button:hover{background:#dae8fa}.notice{border-left:4px solid #d2a249;background:#fff9e8;padding:12px}small{color:#637083}article[hidden],section[hidden]{display:none}summary{cursor:pointer}@media(max-width:850px){.media{grid-template-columns:1fr}.refs{grid-template-columns:repeat(2,minmax(0,1fr));max-height:none}.top{margin-left:0}main{padding:10px}article{padding:12px}section,article{scroll-margin-top:210px}}
 </style><div id="top"></div><nav><a href="#top">顶部</a>'''+nav+'''<input id="search" type="search" placeholder="筛选 ID / uid / 英文指令" aria-label="筛选样本"><span id="visible">300 / 300</span></nav><main><h1>300 对冻结测评集 · 离线审核</h1><p>合成200（8类各25）＋CoinVE75＋TripVVT25。SOURCE → 有序REFERENCE → GT，所有样本连续展开，无评分控件。</p><div class="notice">GT与官方caption仅供审核，不进入推理。技术校验通过不代表人工质量通过。视频是共同时间轴的H.264浏览副本，12fps、最长10秒，不变速；完整原件保留。合成t=10端点沿用原生协议，H3对齐尾部不在本页展示。同步按钮控制同一样本SOURCE、动作参考与GT；拖动“共同时间”可逐时刻比较。播放到冻结末端自动暂停。</div><p><a href="README.md">交接与模型说明</a> · <a href="sampling.json">冻结抽样统计</a> · <a href="validation.json">技术验收</a> · <small>建议电脑浏览器打开解压后的 index.html；引用均为包内相对路径，无需网络。</small></p>'''+''.join(sections)+'''</main><script>
 document.querySelectorAll('article').forEach(card=>{
 const clips=[...card.querySelectorAll('video.clip')],seek=card.querySelector('.seek'),out=card.querySelector('output'),end=Number(seek.max);let guard=false;
 const display=t=>{seek.value=t;out.textContent=t.toFixed(2)+' s';};
 const align=t=>{guard=true;clips.forEach(v=>{if(Math.abs(v.currentTime-t)>.025)v.currentTime=t});display(t);guard=false;};
 const pause=()=>clips.forEach(v=>v.pause());
 card.querySelector('.play-pair').onclick=()=>{const t=Number(seek.value)>=end?0:Number(seek.value);align(t);clips.forEach(v=>{const q=v.play();if(q)q.catch(()=>{});});};
 card.querySelector('.pause-pair').onclick=pause;
 card.querySelector('.reset-pair').onclick=()=>{pause();align(0);};
 seek.oninput=()=>{pause();align(Number(seek.value));};
 const master=clips[0];master.addEventListener('timeupdate',()=>{if(guard)return;const t=Math.min(end,master.currentTime);display(t);if(master.currentTime>=end){pause();align(end);}else if(!master.paused){clips.slice(1).forEach(v=>{if(Math.abs(v.currentTime-t)>.15)v.currentTime=t;});}});
 clips.forEach(v=>v.addEventListener('timeupdate',()=>{if(!guard&&v.currentTime>end){v.pause();v.currentTime=end;}}));
 });
 document.querySelector('#search').oninput=e=>{const q=e.target.value.trim().toLowerCase();let n=0;document.querySelectorAll('article').forEach(a=>{a.hidden=!a.dataset.search.toLowerCase().includes(q);if(!a.hidden)n++;else a.querySelectorAll('video').forEach(v=>v.pause());});document.querySelectorAll('section').forEach(s=>s.hidden=![...s.querySelectorAll('article')].some(a=>!a.hidden));document.querySelector('#visible').textContent=n+' / 300';};
 </script></html>'''
 if compact:
  page=page.replace('300 / 300',f'{len(inputs)} / {len(inputs)}').replace("n+' / 300'",f"n+' / {len(inputs)}'").replace('300 对冻结测评集 · 离线审核','30 对精简审核 · 来自冻结300对').replace('合成200（8类各25）＋CoinVE75＋TripVVT25。','8个合成人物各3对（24对）＋CoinVE3对＋TripVVT3对；完整冻结300对保留本地。')
 (p/'index.html').write_text(page)
 print('HTML_COMPLETE',len(inputs),'samples',len(records),'full-decoded browser clips',flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--compact',action='store_true');args=parser.parse_args();build(args.compact)
