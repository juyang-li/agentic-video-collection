"""Offline continuous review; exact historical IDs, full prompts, common 0–10s playback."""
from common import *
import argparse,html,shutil,time,zipfile,fcntl,os,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from html.parser import HTMLParser
TITLES={'ltx_base':'LTX2.5 基座 · DataMake V7未微调对照','a849':'A849 · 相同V7协议','vace':'Wan2.1 VACE','h3':'MiniMax H3 · 本地Ref2VA'}

def decode_check(path,frames,fps,canvas):
 import av
 with av.open(str(path)) as c:
  s=c.streams.video[0];assert s.average_rate==fps;n=0
  for f in c.decode(video=0):assert (f.width,f.height)==tuple(canvas);n+=1
  assert n==frames,(str(path),n)
 return {'frames':n,'fps':fps,'canvas_wh':canvas,'sha256':sha(path),'full_decode':True}

def sample_job(task):
 item,models,name,gt,show_vace_preparation=task
 from PIL import Image
 from media import video_frames,write_preview
 sid=item['id'];cache=ROOT/'review/cache';out=ROOT/'review'/name;media=[];generation=[];checks=[]
 def browser(role,record,key):
  assert sha(record['absolute_path'])==record['sha256']
  p=cache/'media'/f'{sid}_{key}.mp4';receipt=cache/'receipts'/f'{sid}_{key}.json'
  identity={'source_sha256':record['sha256'],'pixel_script_sha256':sha(ROOT/'scripts/media.py'),'canvas_wh':item['canvas_wh'],'frames':121,'fps':12}
  if receipt.exists():
   audit=read(receipt);assert audit['identity']==identity and sha(p)==audit['validation']['sha256']
  else:
   frames,pixels=video_frames(item,record);assert len(frames)==121
   temp=p.with_suffix('.part.mp4');write_preview(frames,temp,12);temp.replace(p)
   audit={'identity':identity,'pixel_audit':pixels,'validation':decode_check(p,121,12,item['canvas_wh'])};atomic(receipt,audit)
  dst=out/'media'/p.name
  if not dst.exists():os.link(p,dst)
  assert sha(dst)==audit['validation']['sha256']
  m={'role':role,'kind':'video','path':str(dst.relative_to(out))};media.append(m);checks.append({'path':m['path'],**audit['validation']})
 browser('SOURCE',item['source'],'source')
 for idx,ref in enumerate(item['references']):
  role=f'REFERENCE {idx+1} · {ref["type"]} · {ref.get("view","")}'
  if ref['type']=='motion':browser(role,ref,f'motion_{idx}')
  else:
   assert sha(ref['absolute_path'])==ref['sha256'];src=Path(ref['absolute_path']);dst=out/'media'/f'{sid}_ref_{idx}{src.suffix}'
   if not dst.exists():os.link(src,dst)
   assert sha(dst)==ref['sha256']
   with Image.open(dst) as im:im.verify()
   media.append({'role':role,'kind':'image','path':str(dst.relative_to(out)),'sha256':ref['sha256']})
 browser('GT · 仅审核',{'absolute_path':gt['frozen_gt_path'],'sha256':gt['gt_sha256']},'gt')
 for model in models:
  base=ROOT/'h3/outputs' if model=='h3' else ROOT/'outputs'/model
  state=read(base/f'{sid}.json');assert state['status']=='complete'
  src=base/f'{sid}.mp4';assert sha(src)==state['output_sha256']
  frames,fps,canvas=(243,24,[1344,768] if item['canvas_wh'][0]>item['canvas_wh'][1] else [768,1344]) if model=='h3' else (121,12,item['canvas_wh'])
  receipt=cache/'receipts'/f'{sid}_{model}.json'
  if receipt.exists():validation=read(receipt);assert validation['sha256']==state['output_sha256']
  else:validation=decode_check(src,frames,fps,canvas);atomic(receipt,validation)
  dst=out/'media'/f'{sid}_{model}.mp4'
  if not dst.exists():os.link(src,dst)
  assert sha(dst)==validation['sha256']
  m={'role':TITLES[model],'kind':'video','path':str(dst.relative_to(out))};media.append(m);checks.append({'path':m['path'],**validation});generation.append(state)
 if 'vace' in models and show_vace_preparation:
  state=read(ROOT/'outputs/vace'/f'{sid}.json');condition=read(ROOT/'conditions'/sid/'condition.json')
  for key,source in [('prepared',state['conditioning_input_preview']),('overlay',condition['overlay_video'])]:
   src=Path(source);receipt=cache/'receipts'/f'{sid}_vace_{key}.json'
   if receipt.exists():
    validation=read(receipt);assert validation['sha256']==sha(src)
   else:
    validation=decode_check(src,121,12,item['canvas_wh']);atomic(receipt,validation)
   dst=out/'media'/f'{sid}_vace_{key}.mp4'
   if not dst.exists():os.link(src,dst)
   assert sha(dst)==validation['sha256']
   m={'role':'VACE '+key+' · 仅前两条展示','kind':'video','path':str(dst.relative_to(out))};media.append(m);checks.append({'path':m['path'],**validation})
 print(sid,'review ready',flush=True)
 return {'id':sid,'uid':item['uid'],'group':item['group'],'prompt':item['prompt'],'media':media,'checks':checks,'generations':generation}

def main(args):
 require_cpu_task();started=time.time();models=args.models
 assert len(models)==len(set(models)) and all(m in TITLES for m in models)
 lock=(ROOT/'logs/package.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # CPU-only media work, preferentially on cores separate from the GPU workers.
 allowed=set(os.sched_getaffinity(0));busy=set()
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   c=(p/'cmdline').read_bytes()
   if b'benchmark_full10s_20261112/scripts/vace_worker.py' in c or b'refine_white_hair_cpu.py' in c or b'refine_white_sam_cpu.py' in c:busy.update(os.sched_getaffinity(int(p.name)))
  except OSError:pass
 for cpu in list(busy):
  try:
   for part in Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip().split(','):
    if '-' in part:a,b=map(int,part.split('-'));busy.update(range(a,b+1))
    else:busy.add(int(part))
  except OSError:pass
 spare=allowed-busy
 if len(spare)>=4:os.sched_setaffinity(0,spare)
 name=args.name;assert '/' not in name and name
 assert set(args.stopped_models)<=set(models)
 out=ROOT/'review'/name
 for p in [out/'media',ROOT/'review/cache/media',ROOT/'review/cache/receipts']:p.mkdir(parents=True,exist_ok=True)
 inputs=rows(ROOT/'metadata/inputs.jsonl');assert len(inputs)==80
 targets={r['id']:r for r in read(ROOT/'metadata/review_targets.json')}
 if 'h3' in models:assert read(ROOT/('h3/metadata/protocol_gate_completed.json' if args.completed_only else 'h3/metadata/protocol_gate_all.json'))['status']=='passed'
 selected={}
 for item in inputs:
  available=[]
  for model in models:
   base=ROOT/'h3/outputs' if model=='h3' else ROOT/'outputs'/model
   path=base/(item['id']+'.json')
   if not args.completed_only or (path.exists() and read(path).get('status')=='complete'):available.append(model)
  selected[item['id']]=available
 if (out/'completed_snapshot.json').exists():
  snapshot=read(out/'completed_snapshot.json');assert snapshot['completed_only']==args.completed_only
  selected=snapshot['samples'];assert set(selected)=={r['id'] for r in inputs}
  assert all(set(v)<=set(models) for v in selected.values())
 else:
  snapshot={'created_unix':time.time(),'completed_only':args.completed_only,'samples':selected,'counts':{m:sum(m in v for v in selected.values()) for m in models}}
  atomic(out/'completed_snapshot.json',snapshot)
 assert snapshot['counts']=={m:sum(m in v for v in selected.values()) for m in models}
 with ProcessPoolExecutor(max_workers=min(args.workers,4),mp_context=multiprocessing.get_context('spawn')) as pool:
  inventory=list(pool.map(sample_job,[(r,selected[r['id']],name,targets[r['id']],i<2 and not args.omit_vace_preparation) for i,r in enumerate(inputs)]))
 groups=[g for g in GROUPS if any(r['group']==g for r in inputs)];cards=[];last=None
 for row in inventory:
  if last!=row['group']:cards.append('<h2 id="task-'+row['group']+'">'+html.escape(LABELS[row['group']])+'</h2>');last=row['group']
  prompts=''.join('<p class="prompt">'+html.escape(v)+'</p>' for v in row['prompt'].values() if v)
  panels=[]
  for m in row['media']:
   path=html.escape(m['path'],quote=True);role=html.escape(m['role'])
   body=('<video controls muted playsinline preload="none" src="'+path+'"></video><a target="_blank" href="'+path+'">独立查看完整视频</a>') if m['kind']=='video' else '<img loading="lazy" src="'+path+'" alt="'+role+'">'
   panels.append('<figure><figcaption>'+role+'</figcaption>'+body+'</figure>')
  missing=[TITLES[m] for m in models if m not in selected[row['id']]]
  if missing:panels.append('<p>本次快照缺失输出：'+html.escape('、'.join(missing))+'；VACE已按用户要求停止，不再补跑。</p>' if args.stopped_models == ['vace'] else '<p>本次快照尚未完成：'+html.escape('、'.join(missing))+'</p>')
  cards.append('<article id="'+row['id']+'"><h3>'+row['id']+' · '+row['uid']+'</h3>'+prompts+'<button onclick="playCard(this)">同步从头播放 0–10 秒</button><div class="media">'+''.join(panels)+'</div><a href="#top">返回顶部</a></article>')
 nav='<nav><a href="#top">顶部</a>'+''.join('<a href="#task-'+g+'">'+html.escape(LABELS[g])+'</a>' for g in groups)+'<select aria-label="样本导航" onchange="location.hash=this.value"><option value="top">跳转样本</option>'+''.join('<option value="'+r['id']+'">'+r['id']+'</option>' for r in inventory)+'</select></nav>'
 continuation='VACE已按用户要求停止，未完成条目不再补跑；本包不含VACE遮罩、prepared input或overlay视频。' if args.stopped_models == ['vace'] and args.omit_vace_preparation else '后续继续补跑。'
 names='、'.join(TITLES[m]+' '+str(snapshot['counts'][m])+'/80' for m in models)
 page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>完整10秒阶段审核</title><style>body{font:16px/1.55 system-ui;background:#15191e;color:#edf1f4;margin:24px;padding-top:120px}a{color:#8dccff}nav{position:fixed;top:0;left:0;right:0;padding:12px;background:#202934;display:flex;gap:12px;flex-wrap:wrap;z-index:10}[id]{scroll-margin-top:135px}article{border:1px solid #667;padding:16px;margin:24px 0}.prompt{white-space:pre-wrap;overflow-wrap:anywhere}.media{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}figure{margin:0}video,img{width:100%;max-height:520px;object-fit:contain;background:black}button,select{padding:8px}figcaption{min-height:2em}</style><div id="top"></div>'''+nav+'<h1>完整10秒 · 80样本阶段审核</h1><p>本阶段包含 '+html.escape(names)+'。这是已完成结果的阶段快照；各模型数量如下，缺失项在样本中标注。'+html.escape(continuation)+'</p><p>沿用历史100条ID，其中80条UE具备完整10秒；20条CoinVE/TripVVT原片不足10秒，保留ID并暂缓，未补帧或换样。SOURCE、motion与GT按训练端点规则展示121帧/12fps（取样时刻0–10秒，编码长度121/12秒）；共同播放0–10秒，不变速。H3输出为原生243帧/24fps（约10.125秒），50次更新；VACE为121帧/12fps、40步UniPC、CFG5。GT仅审核，不进入推理。</p><p>“LTX2.5基座”使用DataMake V7自定义参考条件协议，属于未微调消融；A849使用同协议和step849 LoRA。两者40步、seed42及原指导参数不变。无评分表，技术检查不代表人工质量通过。旧4秒结果见先前a100审核包；本包不作已证实改善或截取因果结论。</p>'+''.join(cards)+'''<script>
let stopSync=null;
async function playCard(b){if(stopSync)cancelAnimationFrame(stopSync);document.querySelectorAll('video').forEach(v=>v.pause());const vs=[...b.closest('article').querySelectorAll('video')];await Promise.all(vs.map(v=>new Promise(resolve=>{if(v.readyState>=1)resolve();else{v.addEventListener('loadedmetadata',resolve,{once:true});v.load();}})));vs.forEach(v=>{v.currentTime=0;v.playbackRate=1;v.muted=true});await Promise.all(vs.map(v=>v.play().catch(()=>{})));function tick(){if(vs[0].currentTime>=10){vs.forEach(v=>v.pause());stopSync=null;return}stopSync=requestAnimationFrame(tick)}tick();}
</script></html>'''
 (out/'index.html').write_text(page)
 class Links(HTMLParser):
  def handle_starttag(self,tag,attrs):
   for key,value in attrs:
    if key=='src':assert (out/value).is_file(),value
 Links().feed(page)
 checks=[v for r in inventory for v in r['checks']];generations=[v for r in inventory for v in r['generations']]
 assert len(generations)==sum(snapshot['counts'].values())
 assert len(checks)==160+len(generations)+sum(sum(x['type']=='motion' for x in r['references']) for r in inputs)+sum(2 for r in inputs[:2] if 'vace' in selected[r['id']] and not args.omit_vace_preparation)
 atomic(out/'manifest.json',inventory);atomic(out/'validation.json',{'status':'passed','samples':80,'models':models,'outputs':len(generations),'completed_only':args.completed_only,'model_counts':snapshot['counts'],'stopped_models':args.stopped_models,'vace_preparation_included':not args.omit_vace_preparation,'videos_full_decode_coverage':len(checks),'checks':checks,'human_quality':'not assessed'})
 for p in [ROOT/'metadata/preflight.json',ROOT/'metadata/short_clip_exceptions.json',ROOT/'metadata/baseline_definition.json']:
  shutil.copy2(p,out/p.name)
 (out/'README.txt').write_text('阶段审核：80条完整10秒样本，模型：'+names+'。离线解压后打开index.html。完整提示词、SOURCE、参考、GT与输出同页；无评分表。GT仅审核，技术检查不代表人工通过。'+continuation+'\n')
 z=out.with_suffix('.zip');temp=out.with_suffix('.part.zip')
 with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED) as archive:
  for p in sorted(out.rglob('*')):
   if p.is_file():archive.write(p,str(p.relative_to(out.parent)))
 temp.replace(z)
 atomic(ROOT/'metadata'/f'package_{name}.json',{'status':'complete','zip':str(z),'samples':80,'models':models,'model_counts':snapshot['counts'],'outputs':len(generations),'videos':len(checks),'package_seconds':time.time()-started,'completed_unix':time.time()});print('PACKAGE_COMPLETE',z,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',default=['ltx_base','a849']);p.add_argument('--name',default='full10s_base_a849_80_scroll_20261113');p.add_argument('--workers',type=int,default=4);p.add_argument('--completed-only',action='store_true');p.add_argument('--omit-vace-preparation',action='store_true');p.add_argument('--stopped-models',nargs='*',choices=list(TITLES),default=[]);main(p.parse_args())
