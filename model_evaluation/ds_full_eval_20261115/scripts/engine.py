"""CPU preparation and bounded asynchronous DeepSeek evaluation, no generation."""
import asyncio,base64,collections,concurrent.futures,fcntl,hashlib,io,json,math,multiprocessing,os,random,re,sys,time,uuid
from pathlib import Path
import httpx
import av
from cp_policy import VERSION as CP_VERSION, calibrate
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1];EVAL=ROOT.parent
PKG=EVAL/'frozen300_20261113/package';V4=EVAL/'a849_external_vbench_v4_116';MAP=EVAL/'frozen300_vbench_20261115/metadata'
UP=Path('/home/users/astar/cfar/stuchengyou/jcy/xks/DataMake/10_research/model_experiments/evaluation_protocol/prompts')
METRICS=['edit_achievement','content_preservation']
MODEL='deepseek/deepseek-v4.1-flash-1017'
PARAMS={'model':MODEL,'temperature':0,'max_tokens':800,'reasoning':{'enabled':False},'provider':{'only':['wafer/us'],'allow_fallbacks':False,'require_parameters':True,'quantizations':['mxfp4'],'max_price':{'prompt':0.04,'completion':0.6}}}
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):return digest(Path(p).read_bytes())
def attempt_files():
 for directory in ROOT.glob('results*'):
  if directory.is_dir():yield from directory.glob('*attempt*.json')
def save(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)
def freeze(p,data):
 data=json.loads(json.dumps(data,ensure_ascii=False))
 if Path(p).exists():assert read(p)==data,f'Frozen content changed: {p}'
 else:save(p,data)
def attribute(t):return {'hair':'hairstyle','site':'scene','background':'scene','single_piece':'clothing','whole_outfit':'clothing','tripvvt':'clothing','coinve':'instruction_conditioned','native':'instruction_conditioned'}.get(t['group'],t['group'])
def checkpoint_selection(tasks,selections):
 rng=random.Random(20261117);chosen={t['key']:t for t in tasks if t['key'] in selections['smoke']}
 targets={'frozen300':{'identity':6,'single_piece':6,'whole_outfit':6,'action':6,'site':6,'hair':6,'hair_color':6,'camera':6,'coinve':18,'tripvvt':6},'v4_116':{'hair':6,'hair_color':6,'background':6,'tripvvt':6,'native':4}}
 for ds,groups in targets.items():
  for group,n in groups.items():
   existing=sum(t['dataset']==ds and t['group']==group for t in chosen.values())
   pool=sorted([t for t in tasks if t['dataset']==ds and t['group']==group and t['key'] not in chosen],key=lambda t:t['key'])
   for t in rng.sample(pool,n-existing):chosen[t['key']]=t
 assert len(chosen)==100
 return sorted(chosen)
def setup_checkpoint():
 path=ROOT/'metadata/selections.json';sels=read(path)
 selected=checkpoint_selection(read(ROOT/'metadata/tasks.json'),sels)
 if 'checkpoint100' in sels:assert sels['checkpoint100']==selected;return
 save(ROOT/'metadata/selections_before_checkpoint100.json',sels)
 sels['checkpoint100']=selected;save(path,sels)
 save(ROOT/'metadata/checkpoint100_protocol.json',{'tasks':100,'frozen300':72,'v4_116':28,'base_requests':800,'seed':20261117,'comparison':'paired A849 minus H3 and VACE, IF/CP separately per dataset','investigate_trigger':'mean difference <= -0.3 on 1-5 scale and upper paired-bootstrap 95% CI < 0; also inspect strong category-specific gaps','prompt_changes':'only evidence-supported rubric corrections, applied to all models; preserve prior versions and rescore affected metrics','calibration_reporting':'if revised after examining checkpoint, report these tasks separately from untouched tasks','resume_requires':'metadata/checkpoint100_review.json with status approved_to_continue (agent diagnostic review, no extra user approval required)'})
 print('FROZEN checkpoint:100 tasks /800 base calls; stop for diagnostic review',flush=True)
def setup():
 manifests={};tasks=[]
 srcs={r['id']:r for r in rows(MAP/'source_media.jsonl')};outs=collections.defaultdict(list)
 for r in rows(MAP/'scoring_media.jsonl'):outs[r['id']].append({'model':r['model'],'path':r['path'],'sha256':r['output_sha256'],'native_sha256':r['native_sha256']})
 for dataset,path,base in [('frozen300',PKG/'inputs.jsonl',PKG),('v4_116',V4/'metadata/inputs.jsonl',V4)]:
  manifests[dataset]=sha(path)
  for original in rows(path):
   t={k:original.get(k) for k in ['id','uid','group','subgroup','prompt','num_frames','fps','timestamps_seconds','ordered_instructions']};t.update(dataset=dataset,key=dataset+'__'+original['id'],manifest_sha256=manifests[dataset])
   if dataset=='frozen300':
    s=srcs[t['id']];t.update(source=s['path'],source_sha256=s['output_sha256'],outputs=outs[t['id']])
   else:
    source=base/'inputs/prepared'/t['id']/'source.mp4'
    t.update(source=str(source),source_sha256=sha(source),outputs=[{'model':m,'path':str(base/'outputs'/m/(t['id']+'.mp4')),'sha256':sha(base/'outputs'/m/(t['id']+'.mp4'))} for m in ['ltx_base','a849','vace','h3_pdd8']])
   t['references']=[]
   for i,r in enumerate(original['references']):
    t['references'].append({'path':str(base/r['path']) if dataset=='frozen300' else r['absolute_path'],'sha256':r['sha256'],'type':r['type'],'view':r.get('view',''),'index':r.get('index',i),'sample_frame_indices':r.get('sample_frame_indices')})
   t['references'].sort(key=lambda r:r['index']);n=t['num_frames'];k=max(16,math.ceil(4*(n-1)/12));t['indices']=[math.floor(i*(n-1)/(k-1)+.5) for i in range(k)]
   assert len(t['outputs'])==4 and all(Path(r['path']).is_file() for r in t['outputs']) and Path(t['source']).is_file()
   tasks.append(t)
 assert len(tasks)==416 and sum(len(t['outputs']) for t in tasks)==1664
 for a in ['identity','clothing','action','camera','hairstyle','hair_color','scene','instruction_conditioned']:
  for m in METRICS:
   src=(UP/f'{a}_{m}.md') if a!='instruction_conditioned' else EVAL/'vlm_judge_pilot_20261115/prompts'/f'{a}_{m}.md'
   text=src.read_text();text=text.split('## 中文提示词\n')[-1]
   if m=='content_preservation':
    text='\n'.join(l for l in text.splitlines() if not l.startswith('- `VISUAL_REFERENCE`'))
    if a!='instruction_conditioned':text+='\n'+read(UP/'request_suffixes.json')['zh'][a]
    text+='\n本项不提供参考像素。仅按原始指令确定允许修改范围；不评价目标修改是否成功。'
   if a=='identity':text+='\n原始指令优先；不得将明确要求的身体肤色变化扣作无关变化。'
   if a=='instruction_conditioned':text+='\n对于复合编辑，须检查全部子指令及其顺序，不能只评价第一项。CP排除所有明确修改范围的并集。'
   text+='\n媒体以K张按时间顺序排列的配对图P00至P(K-1)展示。每张左侧SOURCE，右侧OUTPUT；两侧时间相同。动作参考R编号单独按冻结时间轴展示。只依据可见内容判断，不把参考人物/衣着/背景复制当作无关目标。reason引用可见P帧号。'
   if m=='content_preservation':text=calibrate(text)
   p=ROOT/'prompts'/f'{a}_{m}.md';p.parent.mkdir(exist_ok=True)
   if p.exists():assert p.read_text()==text
   else:p.write_text(text)
 rng=random.Random(20261115)
 def choose(pool,n):return rng.sample(sorted(pool,key=lambda t:t['key']),n)
 f=[t for t in tasks if t['dataset']=='frozen300'];v=[t for t in tasks if t['dataset']=='v4_116'];smoke=[]
 for group in ['identity','single_piece','whole_outfit','action','site','hair','hair_color','camera','coinve','tripvvt']:smoke+=choose([t for t in f if t['group']==group],1)
 for group in ['hair','hair_color','background','tripvvt']:smoke+=choose([t for t in v if t['group']==group],1)
 for sub in ['face','accessory','object_addition','clothing_local_attribute']:smoke+=choose([t for t in v if t['group']=='native' and t['subgroup']==sub],1)
 review=[]
 for group in ['identity','single_piece','whole_outfit','action','site','hair','hair_color','camera']:review+=choose([t for t in f if t['group']==group],2)
 review+=choose([t for t in f if t['group'] not in ['coinve','tripvvt'] and t not in review],4)
 review+=choose([t for t in f if t['group']=='coinve'],7)+choose([t for t in f if t['group']=='tripvvt'],3)
 for group in ['hair','hair_color','background','tripvvt','native']:review+=choose([t for t in v if t['group']==group],2)
 rng=random.Random(20261116);repeat=[]
 for group in sorted(set(t['group'] for t in f)):repeat+=choose([t for t in f if t['group']==group],1)
 repeat+=choose([t for t in f if t not in repeat],5)
 for group in ['hair','hair_color','background','tripvvt','native']:repeat+=choose([t for t in v if t['group']==group],1)
 assert len(smoke)==18 and len(review)==40 and len(repeat)==20
 selections={'smoke':[t['key'] for t in smoke],'review':[t['key'] for t in review],'repeat':[t['key'] for t in repeat]}
 selections['checkpoint100']=checkpoint_selection(tasks,selections)
 freeze(ROOT/'metadata/tasks.json',tasks);freeze(ROOT/'metadata/selections.json',selections)
 freeze(ROOT/'metadata/protocol.json',{'parameters':PARAMS,'cp_version':CP_VERSION,'manifests':manifests,'prompt_hashes':{p.name:sha(p) for p in (ROOT/'prompts').glob('*.md')},'fps':12,'sample_rule':'max(16,ceil(4*T)), endpoints, nearest half up','jpeg_quality':95,'subsampling':0,'panel_long_side':1024,'budget':12,'minimum_balance':2,'python':sys.version,'av':av.__version__,'pillow':Image.__version__})
 print('FROZEN 416 tasks /1664 pairs /3328 base +160 repeat',flush=True)

def norm(im):
 im=ImageOps.exif_transpose(im).convert('RGBA');bg=Image.new('RGBA',im.size,(128,128,128,255));bg.alpha_composite(im);im=bg.convert('RGB');im.thumbnail((1024,1024),Image.Resampling.LANCZOS);return im
def decode(path,indices):
 wanted=set(indices);out={}
 with av.open(str(path)) as c:
  s=c.streams.video[0];s.thread_count=1
  for n,f in enumerate(c.decode(s)):
   if n in wanted:out[n]=norm(f.to_image())
   if n==max(wanted):break
 assert set(out)==wanted,(path,'missing frames');return out
def asset(path,label):return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'label':label}
def prepare(t):
 receipt=ROOT/'metadata/prepared'/f'{t["key"]}.json'
 cached=read(receipt) if receipt.exists() else []
 existing={c['model']:c for c in cached}
 for out in t['outputs']:
  if out['model'] in existing:assert existing[out['model']]['output_sha256']==out['sha256'] and existing[out['model']]['source_sha256']==t['source_sha256']
 missing=[out for out in t['outputs'] if out['model'] not in existing]
 if not missing:return [existing[out['model']] for out in t['outputs']]
 indices=t['indices'];src=decode(t['source'],indices);d=ROOT/'frames'/t['key'];d.mkdir(parents=True,exist_ok=True);refs=[]
 for j,r in enumerate(t['references']):
  assert sha(r['path'])==r['sha256']
  if r['type']=='motion':
   native=[r['sample_frame_indices'][i] for i in indices];frames=decode(r['path'],native)
   for k,n in enumerate(native):
    p=d/f'ref{j}_{k:02d}.jpg';frames[n].save(p,quality=95,subsampling=0);refs.append(asset(p,f'R{j+1}-T{k:02d}: motion t={indices[k]/12:.3f}s'))
  else:
   p=d/f'ref{j}.png';norm(Image.open(r['path'])).save(p);refs.append(asset(p,f'R{j+1}: role={r["type"]}, view={r["view"]}'))
 cases=list(cached)
 for out in missing:
  assert sha(out['path'])==out['sha256'],('output_changed',out['path'])
  outframes=decode(out['path'],indices);cards=[]
  for j,n in enumerate(indices):
   a,b=src[n],outframes[n];w,h=max(a.width,b.width),max(a.height,b.height);im=Image.new('RGB',(2*w,h+32),(128,128,128));im.paste(a,((w-a.width)//2,32+(h-a.height)//2));im.paste(b,(w+(w-b.width)//2,32+(h-b.height)//2));draw=ImageDraw.Draw(im);draw.text((8,8),f'P{j:02d} frame={n} t={n/12:.3f}s SOURCE',fill='white');draw.text((w+8,8),'OUTPUT',fill='white');p=d/f'{out["model"]}_{j:02d}.jpg';im.save(p,quality=95,subsampling=0);cards.append(asset(p,f'P{j:02d}, frame={n}, t={n/12:.3f}s'))
  cases.append({'key':t['key']+'__'+out['model'],'task_key':t['key'],'task_id':t['id'],'dataset':t['dataset'],'group':t['group'],'model':out['model'],'source':t['source'],'source_sha256':t['source_sha256'],'output':out['path'],'output_sha256':out['sha256'],'instruction':t['prompt']['en'],'attribute':attribute(t),'references':refs,'cards':cards,'num_frames':t['num_frames']})
 save(receipt,cases);print('PREPARED',t['key'],flush=True);return cases

def parse(raw):
 text=raw.strip();fence=re.fullmatch(r'```(?:text)?\s*\n(.*?)\n```',text,re.S)
 if fence:text=fence[1].strip()
 m=re.fullmatch(r'score:\s*([1-5])[ \t]*\nreason:[ \t]*([^\n]+)',text)
 if not m:raise ValueError('invalid_score_format')
 return {'score':int(m[1]),'reason':m[2],'format_compliant':fence is None}
def messages(c,m):
 prompt=(ROOT/'prompts'/f'{c["attribute"]}_{m}.md').read_text();content=[{'type':'text','text':'EDIT_INSTRUCTION\n'+c['instruction']}]
 def image(a):
  p=ROOT/a['path'];raw=p.read_bytes();assert digest(raw)==a['sha256'];mime='image/png' if p.suffix=='.png' else 'image/jpeg';content.append({'type':'text','text':a['label']});content.append({'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(raw).decode(),'detail':'high'}})
 if m=='edit_achievement':
  for r in c['references']:image(r)
  if not c['references']:content.append({'type':'text','text':'REFERENCES: NONE'})
 for f in c['cards']:image(f)
 content.append({'type':'text','text':f'K={len(c["cards"])}。仅评价'+('全部目标编辑的达成。' if m=='edit_achievement' else '非目标内容保持；编辑未生效或目标方向错误本身不扣CP。非目标内容充分保留（包括可忽略的轻微差异）给5；低于5须指出具体非目标损失。')+'输出score和reason两行。'})
 return [{'role':'system','content':prompt},{'role':'user','content':content}]

class Runner:
 def __init__(self):
  self.cost=sum((read(p).get('response',{}).get('usage',{}).get('cost',0) or 0) for p in attempt_files());self.reserved=0.;self.bytes=0;self.limit=8;self.completed=0;self.errors=0;self.next_send=0.;self.cooldown=0.;self.status={};self.balance_checked=0.;self.balance=0.;self.ramp_after=0
 async def quota(self,client):
  if time.monotonic()-self.balance_checked<300:return
  r=await client.get('https://openrouter.ai/api/v1/credits');r.raise_for_status();d=r.json()['data'];self.balance=d['total_credits']-d['total_usage'];self.balance_checked=time.monotonic()
 async def request(self,client,c,m,repeat):
  ident=c['key']+'__'+m+f'__r{repeat}';dest=ROOT/'results'/f'{ident}.json'
  if dest.exists() and read(dest).get('status')=='ok':
   old=read(dest)
   assert old['prompt_sha256']==sha(ROOT/'prompts'/f'{c["attribute"]}_{m}.md') and old['parameters']==PARAMS and old['cards']==c['cards'], 'stale score requires versioned migration'
   return
  msg=await asyncio.to_thread(messages,c,m);payload=dict(PARAMS,messages=msg);data=json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode();del payload,msg
  assert len(data)<45*1024**2,('request_body_too_large',ident,len(data))
  signature=digest(data);images=len(c['cards'])+(len(c['references']) if m=='edit_achievement' else 0);reserve=(images*1024+4000)*.04/1e6+800*.6/1e6
  previous=len(list((ROOT/'results').glob(ident+'__attempt*.json')))
  for attempt in range(5):
   await self.quota(client)
   while self.status.get('inflight',0)>=self.limit or self.bytes+len(data)>1024**3 or time.monotonic()<max(self.cooldown,self.next_send):await asyncio.sleep(.1)
   self.next_send=time.monotonic()+.25
   self.reserved+=reserve;self.bytes+=len(data);self.status['inflight']=self.status.get('inflight',0)+1;start=time.monotonic()
   record={'request_id':ident,'case_key':c['key'],'task_key':c['task_key'],'dataset':c['dataset'],'model_editor':c['model'],'metric':m,'repeat':repeat,'signature':signature,'parameters':PARAMS,'prompt_sha256':sha(ROOT/'prompts'/f'{c["attribute"]}_{m}.md'),'cards':c['cards'],'references':c['references'] if m=='edit_achievement' else [],'status':'error','started_unix':time.time()}
   fatal=False;retry=2**(attempt+1)
   try:
    response=await client.post('https://openrouter.ai/api/v1/chat/completions',content=data,headers={'Content-Type':'application/json'});record['http_status']=response.status_code;out=response.json();record['response']=out
    self.cost+=(out.get('usage',{}).get('cost',0) or 0)
    status=response.status_code
    if status==200 and out.get('error'):
     code=str(out['error'].get('code','500'));status=int(code) if code.isdigit() else 500
    record['effective_status']=status
    if status==402:
     reason=out.get('error',{}).get('metadata',{}).get('reason');fatal=reason!='in_flight_budget_exhausted';retry=int(response.headers.get('retry-after','120'));self.cooldown=time.monotonic()+retry;self.limit=max(4,self.limit//2);self.ramp_after=self.completed+32
    elif status in [429,500,502,503,504]:
     self.limit=max(4,self.limit//2);retry=max(retry,int(response.headers.get('retry-after','2')));self.ramp_after=self.completed+32;self.cooldown=time.monotonic()+retry
    elif status in [400,401,403,404]:fatal=True
    if status!=200:raise ValueError('HTTP'+str(status))
    choice=out['choices'][0]
    if choice.get('finish_reason')!='stop':raise ValueError('truncated_or_incomplete')
    record.update(parse(choice['message'].get('content') or ''),status='ok')
   except Exception as e:record['error']=type(e).__name__+': '+str(e)[:160];self.errors+=1
   finally:
    record['seconds']=time.monotonic()-start;record['completed_unix']=time.time();self.reserved-=reserve;self.bytes-=len(data);self.status['inflight']-=1
    save(ROOT/'results'/f'{ident}__attempt{previous+attempt}.json',record);save(dest,record)
   if record['status']=='ok':
    self.completed+=1
    if self.completed>=self.ramp_after:
     if self.completed>=24:
      self.limit=min(16,self.limit+4);self.ramp_after=self.completed+32
    save(ROOT/'metadata/progress.json',{'successful_this_process':self.completed,'cost_usd':self.cost,'inflight':self.status['inflight'],'concurrency':self.limit,'errors':self.errors,'updated_unix':time.time()})
    print('SCORE',ident,record['score'],round(record['seconds'],1),'cost',round(self.cost,4),flush=True);return
   print('RETRY',ident,record.get('error'),flush=True)
   if fatal:raise RuntimeError('fatal_api_error: '+str(record.get('http_status')))
   await asyncio.sleep(retry+random.random())
  raise RuntimeError('retry_exhausted '+ident)

 async def stage(self,client,tasks,repeat_keys,workers=64):
  queue=asyncio.Queue(64);loop=asyncio.get_running_loop();pool=concurrent.futures.ProcessPoolExecutor(4,mp_context=multiprocessing.get_context('spawn'))
  async def producer():
   pending=set();iterator=iter(tasks)
   for _ in range(4):
    t=next(iterator,None)
    if t:pending.add(loop.run_in_executor(pool,prepare,t))
   while pending:
    done,pending=await asyncio.wait(pending,return_when=asyncio.FIRST_COMPLETED)
    for f in done:
     for c in f.result():
      for r in ([0,1] if c['task_key'] in repeat_keys else [0]):
       for m in METRICS:await queue.put((c,m,r))
     t=next(iterator,None)
     if t:pending.add(loop.run_in_executor(pool,prepare,t))
   for _ in range(workers):await queue.put(None)
  async def consumer():
   while True:
    item=await queue.get()
    if item is None:return
    await self.request(client,*item)
  try:
   async with asyncio.TaskGroup() as tg:
    tg.create_task(producer())
    for _ in range(workers):tg.create_task(consumer())
  finally:pool.shutdown(wait=True,cancel_futures=True)
 async def run(self):
  protocol=read(ROOT/'metadata/protocol.json');assert protocol['parameters']==PARAMS
  assert protocol['prompt_hashes']=={p.name:sha(p) for p in (ROOT/'prompts').glob('*.md')}
  tasks=read(ROOT/'metadata/tasks.json');sels=read(ROOT/'metadata/selections.json')
  async with httpx.AsyncClient(headers={'Authorization':'Bearer '+os.environ['LIJUYANG_ICLR_PERSONAL']},timeout=httpx.Timeout(180,connect=20),limits=httpx.Limits(max_connections=64,max_keepalive_connections=32)) as client:
   await self.quota(client)
   await self.stage(client,[t for t in tasks if t['key'] in sels['smoke']],set())
   save(ROOT/'metadata/smoke.json',{'status':'complete','requests':sum(len(t['outputs'])*2 for t in tasks if t['key'] in sels['smoke']),'cost':self.cost})
   self.limit=16
   await self.stage(client,[t for t in tasks if t['key'] in sels['checkpoint100']],set())
   save(ROOT/'metadata/checkpoint100_complete.json',{'status':'complete','tasks':100,'requests':sum(len(t['outputs'])*2 for t in tasks if t['key'] in sels['checkpoint100']),'cost':self.cost})
   review=ROOT/'metadata/checkpoint100_review.json'
   if not review.exists() or read(review).get('status')!='approved_to_continue':
    print('CHECKPOINT100 complete; stopped for agent diagnostic review',flush=True);return
   await self.stage(client,tasks,set(sels['repeat']))
   save(ROOT/'metadata/api_completion.json',{'status':'complete','expected':sum(len(t['outputs'])*2*(2 if t['key'] in sels['repeat'] else 1) for t in tasks),'cost_usd':self.cost})

def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'metadata').mkdir(exist_ok=True)
 lock=(ROOT/'metadata/run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if len(sys.argv)>1 and sys.argv[1]=='setup':setup();return
 if len(sys.argv)>1 and sys.argv[1]=='setup_checkpoint':setup_checkpoint();return
 try:asyncio.run(Runner().run())
 except BaseException as e:save(ROOT/'metadata/failure.json',{'error':repr(e),'time':time.time()});raise
if __name__=='__main__':main()
