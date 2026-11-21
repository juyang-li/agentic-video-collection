"""CPU offline four-model scroll review with frozen input audit and full native H3 links."""
import argparse,html,shutil,zipfile,csv
from html.parser import HTMLParser
from common import *
from score import MODELS
from contracts import DIMS,LABELS

def check_links(dest):
 class Check(HTMLParser):
  def handle_starttag(self,tag,attrs):
   for key,value in attrs:
    if key not in['href','src']or value.startswith('#'):continue
    assert not value.startswith(('http:','https:','/','file:')),value
    assert(dest/value).is_file(),value
 for page in dest.glob('*.html'):Check().feed(page.read_text())

def run(stage,pack):
 from PIL import Image,ImageOps,ImageDraw
 import numpy as np
 inputs=rows(ROOT/'metadata/inputs.jsonl');pilot=read(ROOT/'metadata/freeze.json')['smoke'];selected=[r for r in inputs if stage=='all'or(stage=='supplement'and r['cohort']=='supplement8')or(stage=='smoke'and r['id']in pilot)];dest=ROOT/'review'/stage;dest.mkdir(parents=True,exist_ok=True);media=dest/'media';media.mkdir(exist_ok=True)
 order={g:n for n,g in enumerate(dict.fromkeys(r['group']for r in inputs))};selected.sort(key=lambda r:order[r['group']]);sections=[];input_sections=[];validation=[];seen=set();metrics={};visual=read(ROOT/'metadata/visual_observations.json')['samples']if(ROOT/'metadata/visual_observations.json').exists()else{}
 if(ROOT/'metadata/per_video_metrics.csv').exists():metrics={(r['id'],r['model'],r['dimension']):r['score']for r in csv.DictReader((ROOT/'metadata/per_video_metrics.csv').open())}
 selection={r['id']:r['record']for r in rows(ROOT/'metadata/selection.jsonl')}
 def copy(path):
  path=Path(path);name=sha(path)+path.suffix.lower();out=media/name
  if not out.exists():shutil.copy2(path,out)
  assert out.stat().st_size==path.stat().st_size
  return'media/'+name
 def video(label,path,sync=False,extra=''):
  url=copy(path);return f'<figure><figcaption>{html.escape(label)}</figcaption><video controls preload="none" muted {"data-sync"if sync else""} src="{url}"></video>{extra}</figure>'
 for item in selected:
  sid=item['id'];group=item['group'];index=next(i for i,x in enumerate(inputs)if x['id']==sid)
  if group not in seen:
   heading=f'<h2 id="group-{group}">{html.escape(group)}</h2>';sections.append(heading);input_sections.append(heading);seen.add(group)
  prefix=f'<article id="{sid}"><h2>{sid} · {group} · {item["cohort"]}</h2><p class="prompt">{html.escape(item["prompt"]["en"])}</p><p>来源：{html.escape(str(item.get("task_origin","self_created_reference_edit")))}；旧ID：{html.escape(str(item.get("old_id",item.get("source_task_id"))))}</p><p>{html.escape(item.get("revision_reason",""))}</p>'
  figures=[];source=ROOT/'inputs/prepared'/sid/'source.mp4';rawurl=copy(item['source']['absolute_path']);figures.append(video('SOURCE 0–4 秒',source,True,f'<p><a href="{rawurl}" target="_blank">完整原始 SOURCE</a></p>'))
  for ref in item['references']:
   url=copy(ref['absolute_path']);figures.append(f'<figure><figcaption>REFERENCE {html.escape(ref["type"])}</figcaption><img loading="lazy" src="{url}"></figure>')
  caption=''
  if group=='tripvvt':
   sel=selection[sid];caption='<p class="prompt">官方 caption（仅审核）：'+html.escape(sel['caption'])+'</p>';gt=ROOT/'inputs/prepared'/sid/'gt_review_only.mp4'
   if not gt.exists():
    from media import video_frames,write_preview
    frames,_=video_frames(item,sel['target']);write_preview(frames,gt)
   rawgt=copy(sel['target']['absolute_path']);figures.append(video('GT（仅审核，不作模型输入）',gt,True,f'<p><a href="{rawgt}" target="_blank">完整原始 GT</a></p>'))
  condition=read(ROOT/'conditions'/sid/'condition.json');strip=Image.new('RGB',(1500,200),'#20242a');draw=ImageDraw.Draw(strip)
  for k,f in enumerate([0,12,24,36,48]):
   frame=Image.open(ROOT/'inputs/prepared'/sid/f'{f:03d}.png').convert('RGB');mask=np.asarray(Image.open(condition['mask_frames'][f]))>127;rgb=np.asarray(frame).copy();rgb[mask]=(rgb[mask]*.5+np.array([255,0,0])*.5).astype(np.uint8);strip.paste(ImageOps.contain(Image.fromarray(rgb),(295,170)),(k*300,25));draw.text((k*300+2,2),f'{sid} {f/12:g}s',fill='white')
  maskpath=dest/f'{sid}_mask.jpg';strip.save(maskpath,quality=94);maskhtml=f'<p>红色仅表示 VACE 编辑范围；H3 使用原始 SOURCE。</p><img loading="lazy" class="strip" src="{sid}_mask.jpg">'
  helpers=[]
  if index<2:
   for label,path in[('VACE mask overlay',Path(condition['overlay_video'])),('VACE prepared input',ROOT/'conditions'/sid/f'conditioning_input_{sid}.mp4')]:
    if path.exists():helpers.append(video(label,path))
  footer='<a href="#top">返回顶部</a></article>'
  input_sections.append(prefix+caption+'<div class="media">'+''.join(figures)+'</div>'+maskhtml+footer)
  output_figures=[];valid=0;observations=[]
  for model in MODELS:
   path=ROOT/'outputs'/model/f'{sid}.mp4';receipt=path.with_suffix('.json')
   if receipt.exists()and read(receipt)['status']=='complete':
    r=read(receipt);assert sha(path)==r['output_sha256'];extra=''
    if model=='h3_pdd8':
     browser=Path(r['browser_path']);assert sha(browser)==r['browser_sha256'];native=Path(r['native_path']);assert sha(native)==r['native_sha256'];nativeurl=copy(native);extra=f'<p>浏览副本：共同49帧；<a href="{nativeurl}" target="_blank">完整原生107帧视频（含音频）</a></p>';display=browser
    else:display=path
    output_figures.append(video(model,display,True,extra));valid+=1;validation.append(dict(id=sid,model=model,output_sha256=r['output_sha256'],browser_path=str(display),browser_sha256=sha(display)))
   else:output_figures.append(f'<figure><figcaption>{model}</figcaption><p>尚无完整结果；不计作有效输出。</p></figure>')
   note=visual.get(sid,{}).get('models',{}).get(model,{})
   if note.get('reviewed')and receipt.exists()and note.get('output_sha256')==read(receipt).get('output_sha256'):observations.append(f'<p><b>{model}</b>：编辑 {html.escape(str(note.get("edit")))}；保持 {html.escape(str(note.get("preservation")))}。{html.escape(str(note.get("notes","")))}</p>')
   else:observations.append(f'<p>{model}：直接视觉审核待完成。</p>')
  table='<table><thead><tr><th>指标 ↑</th>'+''.join(f'<th>{m}</th>'for m in MODELS)+'</tr></thead><tbody>'
  for dim,label in zip(DIMS,LABELS):table+='<tr><td>'+label+'</td>'+''.join('<td>'+ (f'{float(metrics[sid,m,dim]):.5f}'if metrics.get((sid,m,dim))else'待评分/缺失')+'</td>'for m in MODELS)+'</tr>'
  table+='</tbody></table><p>Temporal Flickering 为全量原始值，受正常运动影响；静态配对分数见报告。质量指标不直接证明编辑成功。</p>'
  structured=ROOT/'metadata/h3_requests'/f'{sid}.json';prompt=''
  if structured.exists():prompt='<details><summary>完整 H3 结构化文字</summary><pre>'+html.escape(read(structured)['prompt'])+'</pre></details>'
  sections.append(prefix+caption+'<div class="media">'+''.join(figures)+'</div><div class="outputs">'+''.join(output_figures)+'</div>'+''.join(observations)+table+prompt+maskhtml+'<div class="media">'+''.join(helpers)+'</div>'+footer)
 new_items=[r for r in selected if r['cohort']=='supplement8'];overview=''
 if new_items:
  thumbs=[]
  for item in new_items:
   if not item['id'].startswith('hair_celeb_'):continue
   url=copy(item['references'][0]['absolute_path']);thumbs.append(f'<figure><figcaption><a href="#{item["id"]}">{item["id"]} · 新增 Hair</a></figcaption><img loading="lazy" src="{url}"></figure>')
  overview='<section id="supplement-overview"><h2>新增8条：4 Hair + 4 Identity</h2><p>下方四张为新增 Hair 的 CelebA-HQ 人物参考。原有 hair_01～16 保留原参考；新增任务是 hair_celeb_01～04。点击图片标题直接跳转对应 source、完整指令及结果。</p><div class="media">'+''.join(thumbs)+'</div><p>'+ ' · '.join(f'<a href="#{r["id"]}">{r["id"]}</a>'for r in new_items)+'</p></section>'
 nav='<nav><a href="#top">顶部</a>'+('<a href="#supplement-overview">新增8条／非总统Hair</a>'if new_items else'')+''.join(f'<a href="#group-{g}">{g}</a>'for g in dict.fromkeys(r['group']for r in selected))+''.join(f'<a href="#{r["id"]}">{r["id"]}</a>'for r in selected)+'</nav>'
 style='<style>body{font:16px/1.6 system-ui;background:#171d23;color:#eee;margin:20px;padding-top:150px}nav{position:fixed;z-index:10;top:0;left:0;right:0;max-height:125px;overflow:auto;background:#26323e;display:flex;flex-wrap:wrap;gap:10px;padding:10px}a{color:#9dd6ff}[id]{scroll-margin-top:160px}article{border:1px solid #65707a;padding:16px;margin:24px 0}.media{display:flex;flex-wrap:wrap;gap:12px}.media figure{width:310px}.outputs{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:12px;overflow:auto}figure{margin:0}figure video,figure img{width:100%;max-height:550px;object-fit:contain}.strip{width:100%;max-width:1500px}.prompt,pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse}td,th{border:1px solid #65707a;padding:6px}</style>'
 script='''<script>document.querySelectorAll('article').forEach(a=>{let busy=false;const v=[...a.querySelectorAll('video[data-sync]')];v.forEach(s=>{s.addEventListener('play',()=>{if(busy)return;busy=true;v.forEach(t=>{if(t!==s){t.currentTime=Math.min(s.currentTime,4);t.play().catch(()=>{});}});busy=false;});s.addEventListener('pause',()=>{if(busy)return;busy=true;v.forEach(t=>{if(t!==s)t.pause()});busy=false;});s.addEventListener('seeked',()=>{if(busy)return;busy=true;v.forEach(t=>{if(t!==s&&Math.abs(t.currentTime-s.currentTime)>.08)t.currentTime=Math.min(s.currentTime,4);});busy=false;});s.addEventListener('timeupdate',()=>{if(s.currentTime>4){v.forEach(t=>{t.pause();t.currentTime=0})}});});});</script>'''
 head='<!doctype html><html lang="zh"><meta charset="utf-8"><title>A849 v3 104 四模型</title>'+style+'<div id="top"></div>'+nav
 (dest/'index.html').write_text(head+f'<h1>A849 v3 · {len(selected)} 条'+('新增案例预览'if stage=='supplement'else'')+' · 四模型</h1>'+overview+'<p><a href="input_review.html">冻结输入审核页</a> · <a href="report.md">六项指标与分组报告</a></p><p>共同0–4秒；低质量输出保留，未完成项明确显示。H3 PDD8 为四卡八步，原生分辨率/编码与其他模型不同；不以 v2/v3 均值变化声称模型进步。</p>'+''.join(sections)+script+'</html>')
 (dest/'input_review.html').write_text(head+'<h1>冻结输入审核</h1>'+overview+'<p>原有96条 + 新增4Hair及4Identity；49帧12fps，共同0–4秒。Hair/Identity参考背景RGB128，保留人物前景和自然头发。H3源条件为原始视频，不使用VACE遮罩灰图。VACE辅助视频仅见结果页的全包前两条。</p>'+''.join(input_sections)+'</html>')
 for name in ['inputs.jsonl','input_origins.json','freeze.json','reuse_outputs.json','generation_gate.json','input_review.json','reference_processing.json','reference_pixel_audit.json','available_media_audit.json','h3_preflight.json','h3_runtime_contract.json','h3_pilot.json','h3_generation.json','h3_derivation.json','execution_state.json','per_video_metrics.csv','per_task_report.csv','group_metrics.json','temporal_flickering_static_paired.json','visual_observations.json','report.md']:
  p=ROOT/'metadata'/name
  if p.exists():shutil.copy2(p,dest/name)
 for parent in ['a849_external_vbench_v3_identity','a849_external_vbench_v3_identity_fullhead']:
  d=dest/'identity_diagnostics'/parent;d.mkdir(parents=True,exist_ok=True)
  for name in ['inputs.jsonl','identity_review.json','visual_observations.json','report.md']:
   p=ROOT.parent/parent/'metadata'/name
   if p.exists():shutil.copy2(p,d/name)
 records=dest/'generation_records';records.mkdir(exist_ok=True)
 for item in selected:
  for model in[*MODELS,'h3_pdd8_native']:
   p=ROOT/'outputs'/model/f'{item["id"]}.json'
   if p.exists():shutil.copy2(p,records/f'{model}--{p.name}')
 if stage=='supplement':
  shutil.copy2(ROOT/'metadata/inputs.jsonl',dest/'parent104_inputs.jsonl')
  (dest/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in selected))
  atomic(dest/'snapshot_status.json',dict(status='incomplete_generation_preview',selected_ids=[r['id']for r in selected],complete_model_outputs=len(validation),expected_four_model_outputs=4*len(selected),parent104_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),note='This ZIP only previews the8 added cases. Original96 cases remain unchanged; H3 and unfinished VACE are explicitly pending. Do not treat the preview as final evaluation.'))
 check_links(dest);atomic(ROOT/'metadata'/f'{stage}_media_validation.json',dict(status='passed'if len(validation)==4*len(selected)else'partial',complete_generations=len(validation),expected_generations=4*len(selected),videos=validation,page=str(dest/'index.html')))
 if pack:
  assert stage=='supplement'or(stage=='all'and len(validation)==416)
  output=ROOT/'review'/('a849_v3_supplement8_preview_20261114.zip'if stage=='supplement'else'a849_external_vbench_v3_104_scroll.zip');tmp=output.with_suffix('.part.zip')
  with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1)as z:
   for p in sorted(dest.rglob('*')):
    if p.is_file():z.write(p,Path('a849_external_vbench_v3_104')/p.relative_to(dest))
  with zipfile.ZipFile(tmp)as z:assert z.testzip()is None
  tmp.replace(output);atomic(ROOT/'metadata'/('supplement_package.json'if stage=='supplement'else'package.json'),dict(path=str(output),bytes=output.stat().st_size,offline_page='a849_external_vbench_v3_104/index.html',stage=stage,complete_generations=len(validation),expected_generations=4*len(selected),status='preview'if stage=='supplement'else'complete',zip_crc='passed'))
 print('Review',stage,len(validation),'/',4*len(selected),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','supplement','all'],default='all');p.add_argument('--pack',action='store_true');a=p.parse_args();run(a.stage,a.pack)
