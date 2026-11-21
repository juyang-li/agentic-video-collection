"""Portable unscored comparison. Complete raw H3 videos, common 0–4s playback."""
from common import *
import time,html,shutil,zipfile,statistics
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
NAME='minimax_h3_ref2va90_local_scroll_20261112'

def memory_summary(start,end):
 peaks={}
 for path in (ROOT/'logs/memory').glob('allocator_*.jsonl'):
  for line in path.open():
   try:r=json.loads(line)
   except json.JSONDecodeError:continue
   if not start<=r['unix']<=end:continue
   key=r['uuid'] or str(r['cuda_index']);p=peaks.setdefault(key,{'cuda_index':r['cuda_index'],'allocated_bytes':0,'reserved_bytes':0,'peak_allocated_bytes':0,'peak_reserved_bytes':0})
   for k in ('allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes'):p[k]=max(p[k],r[k])
 return peaks

def summary(states):
 start=min(s['started_unix'] for s in states);end=max(s['completed_unix'] for s in states)
 groups={}
 for g in GROUPS:
  values=[s['total_seconds'] for s in states if s['group']==g]
  groups[g]={'count':len(values),'median_request_seconds':statistics.median(values),'mean_request_seconds':statistics.mean(values),'sum_request_seconds':sum(values)}
 return {'completed':len(states),'first_request_started_unix':start,'last_request_completed_unix':end,'generation_wall_seconds':end-start,'sum_request_seconds':sum(s['total_seconds'] for s in states),'four_gpu_wall_clips_per_hour':len(states)*3600/(end-start),'request_mean_clips_per_hour':len(states)*3600/sum(s['total_seconds'] for s in states),'groups':groups,'per_gpu_allocator_peak_bytes':memory_summary(start,end),'cold_first_request_included':True,'topology':'TP2_Ulysses2','num_gpus':4,'context_ir':False,'model_revision':read(ROOT/'metadata/model_download_manifest.json')['revision']}

def build():
 require_job();started=time.time();items=rows(ROOT/'metadata/inputs.jsonl');assert len(items)==90
 history=read(ROOT/'metadata/history_recovery.json');assert history['status']=='complete'
 histcards={c['id']:c for c in read(OLD/'metadata/deliveries/a100_results_scroll_20261111/manifest.json')}
 histfiles={(r['id'],r['role']):Path(r['path']) for r in history['files']}
 out=ROOT/'review'/NAME;(out/'media').mkdir(parents=True,exist_ok=True)
 inventory=[];states=[];checks=[]
 for item in items:
  sid=item['id'];state=read(ROOT/'outputs'/f'{sid}.json');assert state['status']=='complete' and sha(ROOT/'outputs'/f'{sid}.mp4')==state['output_sha256']
  assert state['contract']['request_sha256']==sha(ROOT/'metadata/requests'/f'{sid}.json')
  assert state['contract']['source_sha256']==item['prepared_source']['sha256']
  assert state['contract']['context_ir'] is False;states.append(state)
  card={'id':sid,'uid':item['uid'],'group':item['group'],'prompt':item['prompt'],'h3_prompt':item['h3_prompt'],'caption':histcards[sid].get('caption'),'media':[]}
  previous=histcards[sid];assert previous['prompt']==item['prompt'] and previous['uid']==item['uid']
  for m in previous['media']:
   role=m['role']
   if role not in ('SOURCE','GT','ltx_base','a849') and not role.startswith('REFERENCE'):continue
   src=histfiles[sid,role];dst=out/'media'/src.name;shutil.copyfile(src,dst)
   card['media'].append({'role':role,'path':str(dst.relative_to(out)),'video':m['video'],'expected_frames':49 if m['video'] else None,'expected_fps':12 if m['video'] else None,'canvas_wh':item['canvas_wh'] if m['video'] else None})
  dst=out/'media'/f'{sid}_h3.mp4';shutil.copyfile(ROOT/'outputs'/f'{sid}.mp4',dst)
  card['media'].append({'role':'H3-Base Ref2VA · 无Context-IR','path':str(dst.relative_to(out)),'video':True,'expected_frames':107,'expected_fps':24,'canvas_wh':item['h3_canvas_wh']})
  inventory.append(card)
 def check_media(m):
  p=out/m['path'];assert p.is_file() and p.stat().st_size>0
  if not m['video']:
   from PIL import Image
   with Image.open(p) as image:image.verify()
   return {'path':m['path'],'type':'image','status':'passed'}
  import av
  with av.open(str(p)) as c:
   stream=c.streams.video[0];stream.thread_type='AUTO';stream.codec_context.thread_count=2
   assert stream.average_rate==m['expected_fps']
   n=0
   for f in c.decode(video=0):
    assert (f.width,f.height)==tuple(m['canvas_wh']);n+=1
   assert n==m['expected_frames'],(str(p),n)
  return {'path':m['path'],'frames':n,'fps':m['expected_fps'],'canvas_wh':m['canvas_wh'],'status':'passed'}
 media=[m for c in inventory for m in c['media']]
 with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(check_media,media))
 assert sum(m['video'] for m in media)==450
 esc=lambda x:html.escape(str(x));sections=[]
 for g in GROUPS:
  cards=[]
  for c in inventory:
   if c['group']!=g:continue
   panels=''.join('<figure><figcaption>'+esc(m['role'])+'</figcaption>'+('<video controls muted playsinline preload="none" src="'+m['path']+'"></video><a target="_blank" href="'+m['path']+'">独立查看完整视频</a>' if m['video'] else '<img loading="lazy" src="'+m['path']+'">')+'</figure>' for m in c['media'])
   prompts=''.join('<p class="prompt">'+esc(v)+'</p>' for v in c['prompt'].values() if v)
   caption='<p class="prompt">完整官方 caption（仅审核）：'+esc(c['caption'])+'</p>' if c.get('caption') else ''
   cards.append('<article id="'+c['id']+'"><h3>'+c['id']+' · '+c['uid']+'</h3>'+prompts+caption+'<button onclick="playCard(this)">同步从头播放共同0–4秒</button><div class="media">'+panels+'</div><details><summary>H3实际输入文字（固定格式适配）</summary><pre>'+esc(c['h3_prompt'])+'</pre></details><a href="#top">返回顶部</a></article>')
  sections.append('<section id="task-'+g+'"><h2>'+LABELS[g]+'</h2>'+''.join(cards)+'</section>')
 nav='<nav><a href="#top">顶部</a> '+''.join('<a href="#task-'+g+'">'+LABELS[g]+'</a> ' for g in GROUPS)+'<select aria-label="跳转样本" onchange="if(this.value)location.hash=this.value"><option value="">跳转样本</option>'+''.join('<option value="'+c['id']+'">'+c['id']+'</option>' for c in inventory)+'</select></nav>'
 page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>本地H3 / LTX / A849 · 非动作90条</title><style>body{font:16px/1.55 system-ui;background:#15191e;color:#edf1f4;margin:24px;padding-top:105px}nav{position:fixed;top:0;left:0;right:0;z-index:10;background:#202934;padding:12px;display:flex;flex-wrap:wrap;gap:12px}a{color:#8dccff}[id]{scroll-margin-top:110px}article{border:1px solid #65717e;padding:16px;margin:24px 0}.prompt,pre{white-space:pre-wrap;overflow-wrap:anywhere}.media{display:flex;flex-wrap:wrap;gap:12px}figure{width:350px;margin:0}video,img{width:100%;max-height:560px;object-fit:contain;background:black}figcaption{font-weight:bold}button{padding:8px;margin:8px 0}</style><div id="top"></div>'''+nav+'''<h1>本地 H3-Base Ref2VA 与基座 / A849 · 非动作90条</h1><p>同一批历史样本，排除动作组。H3没有使用官方Context-IR或在线生成API；保留完整原指令及参考身份，采用固定六段文字格式适配。GT与target caption只供审核，不进入推理；既有合法首帧参考按冻结清单照用。</p><p>共同比较0–4秒，不变速。基座/A849为历史49帧、12fps、1280×736或736×1280、40步结果；H3为原生107帧、24fps、1344×768或768×1344、50次去噪更新、seed42。H3的原生帧数对齐使完整视频约4.46秒；SOURCE在t=4秒后保持端点用于输入对齐，不读取额外原视频。页面共同播放只到4秒，可通过独立入口观看完整原生视频。</p><p>H3按参考驱动重新生成，可能改变构图、动作或时序。分别判断编辑完成、内容保持、时序和画质；原生分辨率、采样器与帧率不同，不能将像素差异直接解释为编辑质量。无评分表；技术验收不代表人工质量通过。音频不参与比较，默认静音。</p>'''+''.join(sections)+'''<script>
let run=0;
async function playCard(button){const token=++run;document.querySelectorAll('video').forEach(v=>v.pause());const vs=[...button.closest('article').querySelectorAll('video')];await Promise.all(vs.map(v=>new Promise(resolve=>{if(v.readyState>=1)resolve();else{v.addEventListener('loadedmetadata',resolve,{once:true});v.load()}})));if(token!==run)return;vs.forEach(v=>{v.muted=true;v.playbackRate=1;v.currentTime=0});await Promise.all(vs.map(v=>v.play().catch(()=>{})));function tick(){if(token!==run)return;if(vs[0].currentTime>=4){vs.forEach(v=>{v.pause();if(v.currentTime>4)v.currentTime=4});return}requestAnimationFrame(tick)}requestAnimationFrame(tick);}
</script></html>'''
 (out/'index.html').write_text(page)
 class Links(HTMLParser):
  def handle_starttag(self,tag,attrs):
   for k,v in attrs:
    if k=='src' or k=='href' and not v.startswith('#'):assert (out/v).is_file(),v
 Links().feed(page)
 report=summary(states);report['package_seconds']=time.time()-started
 atomic(ROOT/'metadata/final_summary.json',report);atomic(out/'manifest.json',inventory)
 atomic(out/'validation.json',{'status':'passed','samples':90,'new_h3_outputs':90,'videos_full_decoded':450,'images_verified':sum(not m['video'] for m in media),'checks':checks,'human_quality':'not assessed'})
 atomic(out/'generation_records.json',states)
 for name in ('preflight.json','source_relocations.json','model_download_manifest.json','execution.json'):
  shutil.copyfile(ROOT/'metadata'/name,out/name)
 text='# H3本地Ref2VA：非动作90条\n\n无官方Context-IR，无付费API；同一批历史样本、完整指令和参考身份。GT与target caption仅审核。H3原生768短边、107帧/24fps、50步、seed42；基座/A849沿用49帧/12fps、原736短边、40步历史输出。比较共同0–4秒，保留全部H3原生输出；SOURCE末端保持为帧数对齐，不压缩时间，不使用t=4秒后的原素材。\n\n实测统计（四卡协同一条，吞吐量不再乘4）：\n\n```json\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n```\n\nSGLang与权重revision、隔离依赖、构建兼容补丁、输入逐帧像素核对、服务性能报告、每卡allocator/NVML记录均保留在实验目录。启动期缺少Cargo/媒体工具、C++模板和本地模型路径识别问题已留日志；没有修改模型或采用近似推理。\n\n450段视频完整解码、尺寸/帧率/帧数及页面媒体链接检查通过，不代表人工质量通过。审核页默认静音，音频不评分。\n\n网盘目标：/apps/bypy/lance_review/'+NAME+'.zip。仅上传ZIP，以上传器complete回执为准，无额外远端哈希/大小复核或回读。历史ZIP恢复仅用于取回旧比较素材。结果和环境保留，不自动提交、推送或清理。\n'
 (ROOT/'README.md').write_text(text);(out/'README.md').write_text(text)
 zip_path=out.with_suffix('.zip');tmp=out.with_suffix('.part.zip')
 with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED) as z:
  for f in sorted(out.rglob('*')):
   if f.is_file():z.write(f,str(f.relative_to(out.parent)))
 tmp.replace(zip_path)
 atomic(ROOT/'metadata/package.json',{'status':'complete','zip':str(zip_path),'samples':90,'videos':450,'completed_unix':time.time()})
 print('PACKAGE_COMPLETE',zip_path,flush=True)
if __name__=='__main__':build()
