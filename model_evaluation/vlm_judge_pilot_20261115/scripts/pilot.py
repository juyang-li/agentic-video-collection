"""CPU-only reproducible frame preparation and independent API judging."""
import argparse, base64, concurrent.futures, hashlib, io, json, os, random, re, sys, time
from pathlib import Path
import requests
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT.parent / 'a849_external_vbench_v3_116'
UPSTREAM = Path('/home/users/astar/cfar/stuchengyou/jcy/xks/DataMake/10_research/model_experiments/evaluation_protocol/prompts')
INDICES = [0,3,6,10,13,16,19,22,26,29,32,35,38,42,45,48]
MODELS = ['ltx_base','a849','vace','h3_pdd8']
JUDGES = {'deepseek':'deepseek/deepseek-v4.1-flash-1017', 'qwen':'qwen/qwen3-vl-235b-a22b-instruct','gpt':'gpt-6.1-sol'}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(p)
def load(p): return json.loads(Path(p).read_text())
def frozen(p,data):
    data=json.loads(json.dumps(data,ensure_ascii=False))
    if Path(p).exists():
        if load(p)!=data: raise ValueError('Frozen artifact differs: '+str(p))
    else: save(p,data)
def attr(item):
    return {'hair':'hairstyle','hair_color':'hair_color','identity':'identity','background':'scene','tripvvt':'clothing'}.get(item['group'],'instruction_conditioned')
def build_prompts():
    suffixes=load(UPSTREAM/'request_suffixes.json')['zh']
    provenance=[]
    for a in ['hairstyle','hair_color','identity','scene','clothing']:
        for metric in ['edit_achievement','content_preservation']:
            src=UPSTREAM/f'{a}_{metric}.md'
            body=src.read_text().split('## 中文提示词\n',1)[1].strip()
            # A local, immutable snapshot is generated before requests.
            if metric=='content_preservation':
                body='\n'.join(l for l in body.splitlines() if not l.startswith('- `VISUAL_REFERENCE`'))
                body+='\n\n本项不提供参考图像像素；只依据原始指令确定允许改变的范围。\n'
                body+='\n'+suffixes[a]
            if a=='identity':
                body=body.replace('将源视频中指定人物替换为参考图中的目标人物','仅将源视频中指定人物的面部身份替换为参考身份')
                body=re.sub(r'^- 允许.*?(体形|体型|身形|衣物合身|衣服合身).*$', '- 本任务仅替换面部身份；身体形状、衣服合身程度、表情、视线及头部姿态仍须按原始指令保留。',body,flags=re.M)
                body+='\n原始指令优先：本数据仅替换面部身份，不能将整个人物替换视为允许范围。\n'
            if a=='clothing':
                body+='\n参考中的人物身份、姿势和背景不属于服装目标；仅评价原始指令指定的衣物部件。\n'
            body+='\n\n媒体适配：两个视频通过P00–P15配对帧展示，按时间递增。每张图左为SOURCE，右为GENERATED，标签在画面外。相同帧号对应同一时间点；综合所有可见帧，不得只选最佳帧。原始EDIT_INSTRUCTION决定目标与允许变化，优先于模板中的概括。reason中引用实际支持判断的P帧号。\n'
            dest=ROOT/'prompts'/f'{a}_{metric}.md'
            if dest.exists() and dest.read_text()!=body:raise ValueError('Prompt changed')
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.write_text(body)
            provenance.append({'source':str(src),'source_sha256':sha(src),'snapshot':str(dest.relative_to(ROOT)),'sha256':sha(dest)})
    frozen(ROOT/'metadata/prompt_provenance.json',provenance)

def normalize(im):
    im=ImageOps.exif_transpose(im)
    if im.mode=='RGBA' or 'transparency' in im.info:
        rgba=im.convert('RGBA');bg=Image.new('RGBA',im.size,(128,128,128,255));bg.alpha_composite(rgba);im=bg.convert('RGB')
    else:im=im.convert('RGB')
    im.thumbnail((1024,1024),Image.Resampling.LANCZOS)
    return im

def prepare():
    import av
    build_prompts()
    items=[json.loads(l) for l in (BENCH/'metadata/inputs.jsonl').read_text().splitlines()]
    rng=random.Random(20261115);selected=[]
    strata=[('hair',None,1),('hair_color',None,1),('identity',None,1),('background',None,1),('tripvvt',None,1),('native','accessory',1),('native','object_addition',1),('native','object_removal',1),('native','face',1),('native','clothing_local_attribute',1)]
    for group,sub,n in strata:
        pool=sorted([i for i in items if i['group']==group and (sub is None or i.get('subgroup')==sub)],key=lambda x:[int(s) if s.isdigit() else s for s in re.split(r'(\d+)',x['id'])])
        rng.shuffle(pool)
        eligible=[i for i in pool if all((BENCH/'outputs'/m/(i['id']+'.mp4')).is_file() for m in MODELS)]
        if len(eligible)<n:raise ValueError('Insufficient complete candidates')
        selected+=eligible[:n]
    frozen(ROOT/'metadata/selection.json',{'seed':20261115,'strata':strata,'tasks':selected,'source_manifest_sha256':sha(BENCH/'metadata/inputs.jsonl')})
    candidates=[]
    for task_no,item in enumerate(selected):
        sid=item['id'];refs=[]
        for j,r in enumerate(item['references']):
            dest=ROOT/'frames'/f't{task_no:02d}_r{j}.png';dest.parent.mkdir(parents=True,exist_ok=True)
            if sha(r['absolute_path'])!=r['sha256']:raise ValueError('Reference hash mismatch')
            normalize(Image.open(r['absolute_path'])).save(dest)
            refs.append({'path':str(dest.relative_to(ROOT)),'sha256':sha(dest),'type':r['type'],'view':r.get('view','')})
        for model_no,m in enumerate(MODELS):
            cid=f't{task_no:02d}_c{model_no}';out=BENCH/'outputs'/m/(sid+'.mp4');frames={}
            with av.open(str(out)) as container:
                stream=container.streams.video[0];stream.thread_count=1
                for k,f in enumerate(container.decode(stream)):
                    if k in INDICES: frames[k]=f.to_image()
                if k!=48:raise ValueError(f'Expected49frames:{out}:{k+1}')
            cards=[]
            for j,k in enumerate(INDICES):
                source=BENCH/'inputs/prepared'/sid/f'{k:03d}.png'
                a=normalize(Image.open(source));b=normalize(frames[k])
                w,h=max(a.width,b.width),max(a.height,b.height)
                card=Image.new('RGB',(2*w,h+32),(128,128,128));card.paste(a,((w-a.width)//2,32+(h-a.height)//2));card.paste(b,(w+(w-b.width)//2,32+(h-b.height)//2))
                d=ImageDraw.Draw(card);d.text((8,8),f'P{j:02d} frame={k} t={k/12:.3f}s SOURCE',fill='white');d.text((w+8,8),'GENERATED',fill='white')
                dest=ROOT/'frames'/f'{cid}_p{j:02d}.jpg';card.save(dest,quality=95,subsampling=0)
                cards.append({'path':str(dest.relative_to(ROOT)),'sha256':sha(dest),'source_frame_sha256':sha(source),'frame_index':k,'time_seconds':k/12})
            candidates.append({'id':cid,'task_id':sid,'model':m,'attribute':attr(item),'instruction':item['prompt']['en'],'output':str(out),'output_sha256':sha(out),'source':str(BENCH/'inputs/prepared'/sid/'source.mp4'),'source_sha256':sha(BENCH/'inputs/prepared'/sid/'source.mp4'),'references':refs,'cards':cards})
            print('prepared',cid,sid,m,flush=True)
    frozen(ROOT/'metadata/candidates.json',candidates)
    frozen(ROOT/'metadata/protocol.json',{'indices':INDICES,'fps':12,'api_jpeg':{'quality':95,'subsampling':0},'cp_reference_pixels':False,'judges':JUDGES,'python':sys.version,'pillow':Image.__version__,'av':av.__version__})

def messages(c,metric):
    p=ROOT/'prompts'/f'{c["attribute"]}_{metric}.md';body=p.read_text()
    content=[{'type':'text','text':'EDIT_INSTRUCTION\n'+c['instruction']}]
    def image(p):
        raw=(ROOT/p).read_bytes();mime='image/png' if p.endswith('.png') else 'image/jpeg'
        content.append({'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(raw).decode(),'detail':'high'}})
    if metric=='edit_achievement':
        for j,r in enumerate(c['references']):
            content.append({'type':'text','text':f'VISUAL_REFERENCE R{j+1}: role={r["type"]}; view={r["view"]}'});image(r['path'])
        if not c['references']:content.append({'type':'text','text':'VISUAL_REFERENCE: NONE'})
    content.append({'type':'text','text':'PAIRED_VIDEO_FRAMES P00–P15: SOURCE left; GENERATED right; chronological.'})
    for f in c['cards']:image(f['path'])
    content.append({'type':'text','text':'请仅评价'+('目标修改的编辑达成。' if metric=='edit_achievement' else '除指令要求修改以外内容的保持，目标修改是否成功不影响本项。')+'输出score和reason两行，reason引用可见帧号。'})
    return [{'role':'system','content':body},{'role':'user','content':content}],sha(p)

def score(c,metric,judge,repeat=0):
    dest=ROOT/'results'/f'{judge}_{c["id"]}_{metric}_r{repeat}.json'
    if dest.exists() and load(dest).get('status')=='ok':return
    msgs,prompt_sha=messages(c,metric);gpt=judge=='gpt'
    gpt_router=gpt and os.environ.get('VLM_GPT_OPENROUTER')=='1'
    body={'model':('openai/'+JUDGES[judge]) if gpt_router else JUDGES[judge],'messages':msgs}
    if gpt:
        body.update(reasoning_effort='high',max_completion_tokens=4096)
        if gpt_router:body['provider']={'only':['openai'],'allow_fallbacks':False,'require_parameters':True}
    else:
        body.update(temperature=0,max_tokens=800,provider={'only':['Wafer' if judge=='deepseek' else 'DeepInfra'],'allow_fallbacks':False,'require_parameters':True})
        if judge=='deepseek':body['reasoning']={'enabled':False}
    key=os.environ['OPENAI_API_KEY' if gpt and not gpt_router else 'LIJUYANG_ICLR_PERSONAL']
    url='https://api.openai.com/v1/chat/completions' if gpt and not gpt_router else 'https://openrouter.ai/api/v1/chat/completions'
    identity=hashlib.sha256(json.dumps(body,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    attempts=[]
    previous_count=len(list(dest.parent.glob(dest.stem+'_attempt*.json'))) if dest.parent.exists() else 0
    for attempt in range(3):
        start=time.monotonic();record={'candidate_id':c['id'],'task_id':c['task_id'],'judge':judge,'metric':metric,'repeat':repeat,'model_requested':JUDGES[judge],'prompt_sha256':prompt_sha,'request_sha256':identity,'input_cards':c['cards'],'reference_inputs':c['references'] if metric=='edit_achievement' else [],'parameters':{k:v for k,v in body.items() if k!='messages'},'status':'error'}
        try:
            r=requests.post(url,headers={'Authorization':'Bearer '+key},json=body,timeout=240);record['http_status']=r.status_code
            data=r.json();record['response']=data
            if not r.ok:raise ValueError(f'HTTP{r.status_code}')
            raw=data['choices'][0]['message'].get('content') or ''
            match=re.fullmatch(r'score:\s*([1-5])[ \t]*\nreason:[ \t]*([^\n]+)',raw.strip())
            if not match:raise ValueError('invalid_score_format')
            record.update(status='ok',score=int(match[1]),reason=match[2],usage=data.get('usage'))
        except Exception as e:record['error']=type(e).__name__+': '+str(e)[:80]
        record['seconds']=time.monotonic()-start;record['completed_unix']=time.time();attempts.append(record)
        save(dest.with_name(dest.stem+f'_attempt{previous_count+attempt}.json'),record)
        record['attempt_count']=previous_count+len(attempts);save(dest,record)
        print(judge,c['id'],metric,record['status'],record.get('score'),round(record['seconds'],1),flush=True)
        if record['status']=='ok':return
        if record.get('http_status') in [401,402,403,404]:return
        time.sleep(2**attempt)

def run(judges,limit,workers,repeats=False):
    candidates=load(ROOT/'metadata/candidates.json')
    if limit:candidates=candidates[:limit]
    repeat_ids=[0]
    if repeats:
        rng=random.Random(20261116);chosen=[];seen=set()
        for model in MODELS:
            pool=[c for c in candidates if c['model']==model and c['task_id'] not in seen];rng.shuffle(pool)
            for c in pool[:2]: chosen.append(c);seen.add(c['task_id'])
        candidates=chosen;repeat_ids=[1,2]
        frozen(ROOT/'metadata/stability_selection.json',{'seed':20261116,'candidate_ids':[c['id'] for c in chosen]})
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures=[ex.submit(score,c,m,j,r) for c in candidates for m in ['edit_achievement','content_preservation'] for j in judges for r in repeat_ids]
        for f in concurrent.futures.as_completed(futures):f.result()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--judges',nargs='+',default=['deepseek','qwen'],choices=list(JUDGES));p.add_argument('--limit',type=int,default=0);p.add_argument('--workers',type=int,default=4);p.add_argument('--repeats',action='store_true');a=p.parse_args()
    if a.command=='prepare':prepare()
    else:run(a.judges,a.limit,a.workers,a.repeats)
