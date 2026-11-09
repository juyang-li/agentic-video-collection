"""Export the frozen benchmark with portable paths and an offline scroll viewer."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import argparse,base64,hashlib,html,importlib.util,io,json,shutil,sys,threading,zipfile

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
NAME='ltx25_frozen_bench60_scroll_20261030'
PACKAGE=ROOT/'review'/NAME

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def write_lines(p,rows):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

def build():
    from fontTools import subset
    from fontTools.ttLib import TTFont
    PACKAGE.mkdir(parents=True,exist_ok=True)
    for d in ['media','evaluation/targets','frozen']:(PACKAGE/d).mkdir(parents=True,exist_ok=True)
    original=lines(ROOT/'inputs.jsonl');ev=lines(ROOT/'evaluation_only.jsonl');summary=json.loads((ROOT/'inventory_summary.json').read_text())
    assert len(original)==60 and sha(ROOT/'inputs.jsonl')==summary['input_manifest_sha256']
    assert sha(ROOT/'evaluation_only.jsonl')==summary['evaluation_manifest_sha256']
    copied={};mapping=[]
    def media(path,expected,evaluation=False):
        if path in copied:return copied[path]
        src=PROJECT/path;assert sha(src)==expected,(path,'frozen media changed')
        folder='evaluation/targets' if evaluation else 'media'
        dest=folder+'/'+expected+src.suffix.lower();out=PACKAGE/dest
        if not out.exists():shutil.copy2(src,out)
        assert sha(out)==expected
        copied[path]=dest;mapping.append(dict(original_path=path,package_path=dest,sha256=expected,bytes=out.stat().st_size))
        return dest
    portable=[];evaluation=[]
    for r in original:
        p=dict(r);p['source_video']=media(r['source_video'],r['source_sha256'])
        if r['reference_image']:p['reference_image']=media(r['reference_image'],r['reference_sha256'])
        portable.append(p)
    for r in ev:
        p=dict(r)
        if r['target_video']:p['target_video']=media(r['target_video'],r['target_sha256'],True)
        evaluation.append(p)
    write_lines(PACKAGE/'inputs.jsonl',portable);write_lines(PACKAGE/'evaluation/evaluation_only.jsonl',evaluation)
    for name in ['inputs.jsonl','evaluation_only.jsonl','inventory_summary.json','reference_controls.json','README.md']:
        shutil.copy2(ROOT/name,PACKAGE/'frozen'/name)
    shutil.copy2(ROOT/'reference_controls.json',PACKAGE/'reference_controls.json')
    save(PACKAGE/'media_mapping.json',mapping)
    readme='''# 已冻结的 LTX-2.5 单轮推理 bench（60 条）

解压整个 ZIP 后打开 index.html，可连续下拉审阅，无评分栏。本包只有测试输入和已有评测 GT，没有模型推理结果。

- 参考图换衣 25：上衣 9、下装 8、官方 dresses/整装类 8。
- 外观 15、文字换衣 10、镜头 10；动作任务和动作参考视频为 0。
- source、25 张服装参考图和指令全部包含；换衣的 25 份配对 GT 单独位于 evaluation/，其它任务没有 GT。
- 参考换衣来自 TripVVT：source 为合成换装视频，GT 为真实视频；部分官方 dresses 类实际为外套/套装，原始提示和注意事项保留。

inputs.jsonl 是可直接读取的便携输入清单，媒体路径相对本包根目录。evaluation/evaluation_only.jsonl 仅供评测，不得传给模型。reference_controls.json 保留另外 4 个参考图替换对照定义，不计入正式 60 条；所需图片已包含。

frozen/ 保存原始冻结清单的逐字副本（其中媒体路径仍相对仓库 video_collection/）；便携清单只重定位媒体路径，不改样本、顺序、指令、seed、分片或配对关系。media_mapping.json 记录映射和 SHA256。视频/图片原样复制，无转码、裁剪、缩放或变速；本包不是后续推理使用的 97 帧派生缓存。

60 条共 49 个不同 source 内容；有 source 复用，按 group_id 分组分析。镜头中 9 条主要测构图/推拉，另 1 条混合角度和距离，不能代表完整换视角能力。
'''
    (PACKAGE/'README.md').write_text(readme)
    evmap={r['sample_id']:r for r in evaluation};esc=html.escape
    groups=[('garment','参考图换衣 · 25'),('appearance','外观编辑 · 15'),('text_outfit','文字换衣 · 10'),('camera','镜头编辑 · 10')]
    def group(r):
        return 'garment' if r['task'].startswith('garment_') else 'appearance' if r['task'].startswith('appearance_') else 'camera' if r['task'].startswith('camera_') else 'text_outfit'
    def panel(path,label,image=False):
        content=f'<a href="{path}" target="_blank"><img loading="lazy" src="{path}" alt="服装参考图"></a>' if image else f'<video controls muted playsinline preload="none" data-src="{path}"></video>'
        return f'<div class="panel"><h4>{esc(label)}</h4>{content}<a href="{path}" target="_blank">单独打开原文件</a></div>'
    parts=[]
    for g,title in groups:
        parts.append(f'<section id="{g}"><h2>{esc(title)}</h2>')
        for r in [r for r in portable if group(r)==g]:
            e=evmap[r['sample_id']]
            parts.append(f'<article id="{r["sample_id"]}"><h3>{r["sample_id"]} · {r["task"]}</h3><p class="prompt">{esc(r["instruction"])}</p><div class="panels">'+panel(r['source_video'],'SOURCE · 输入视频'))
            if r['reference_image']:parts.append(panel(r['reference_image'],'REF IMAGE · 目标服装参考',True))
            if e['target_video']:parts.append(panel(e['target_video'],'GT · 已有配对目标（不是模型输出）'))
            parts.append('</div><button class="play">本条视频从头播放</button><button class="pause">暂停</button>')
            parts.append('<p>'+('本条含独立服装参考图。' if r['reference_image'] else '本条没有独立参考图或参考视频。')+('' if e['target_video'] else ' 没有编辑后 GT。')+'</p>')
            if e.get('notes'):parts.append(f'<details><summary>来源及原始注意事项</summary><p>{esc(e["notes"])}</p></details>')
            parts.append(f'<p class="meta">seed={r["seed"]} · shard={r["shard"]} · group_id={esc(r["group_id"])}</p></article>')
        parts.append('</section>')
    nav=' '.join(f'<a href="#{g}">{esc(t)}</a>' for g,t in groups)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LTX-2.5 冻结 bench · 60 条</title><style>__FONT__
*{box-sizing:border-box}body{background:#101720;color:#eaf0f7;font:16px/1.7 CJK,system-ui;margin:0}header{position:sticky;top:0;background:#192838f5;padding:12px 22px;z-index:5}nav{display:flex;gap:20px;flex-wrap:wrap}a{color:#8bd1ff}main{max-width:1550px;margin:auto;padding:24px}section,article{scroll-margin-top:110px}article{background:#1b2837;border:1px solid #405367;border-radius:10px;margin:24px 0;padding:22px}.prompt{background:#102638;padding:16px;border-left:3px solid #7bc4f0;white-space:pre-wrap}.panels{display:flex;gap:18px;flex-wrap:wrap}.panel{flex:1;min-width:260px}.panel video,.panel img{width:100%;height:470px;object-fit:contain;background:#080e15}.panel a{font-size:14px}button{padding:9px 15px;margin:12px 8px 0 0;background:#285679;color:white;border:1px solid #719ab7;border-radius:5px;cursor:pointer}.meta{font-size:13px;color:#a8bbcd;overflow-wrap:anywhere}h2{color:#a2d8ff}</style><header><strong>LTX-2.5 已冻结 bench · 60 条 · 无动作任务</strong><nav>__NAV__</nav></header><main><h1>待推理输入审核</h1><p>参考图换衣 25、外观 15、文字换衣 10、镜头 10。原始素材与完整英文指令，未运行模型。换衣的 GT 是已有配对目标，不是本次生成结果；其它任务没有 GT。所有卡片直接向下滚动，无评分栏。</p>__CONTENT__</main><script>
const ob=new IntersectionObserver(es=>es.forEach(e=>{let v=e.target;if(e.isIntersecting){if(!v.src){v.preload='metadata';v.src=v.dataset.src;v.load()}}else v.pause()}),{rootMargin:'400px'});document.querySelectorAll('video').forEach(v=>ob.observe(v));document.querySelectorAll('article').forEach(a=>{const vs=[...a.querySelectorAll('video')];a.querySelector('.play').onclick=()=>{vs.forEach(v=>{if(!v.src)v.src=v.dataset.src;v.currentTime=0});Promise.allSettled(vs.map(v=>v.play()))};a.querySelector('.pause').onclick=()=>vs.forEach(v=>v.pause())});
</script></html>'''.replace('__NAV__',nav).replace('__CONTENT__',''.join(parts))
    fontpath=PROJECT/'dataset_research/ref_edit_review_20261030/assets/font/NotoSansCJKsc-Regular.otf'
    font=TTFont(fontpath);sub=subset.Subsetter();sub.populate(text=page);sub.subset(font);buf=io.BytesIO();font.save(buf)
    face="@font-face{font-family:CJK;src:url(data:font/otf;base64,"+base64.b64encode(buf.getvalue()).decode()+") format('opentype');font-display:block}"
    (PACKAGE/'index.html').write_text(page.replace('__FONT__',face));shutil.copy2(fontpath.parent/'LICENSE.txt',PACKAGE/'FONT_LICENSE.txt')
    save(PACKAGE/'package_validation.json',dict(status='passed',cards=60,reference_images=25,evaluation_targets=25,
      unique_media=len(mapping),frozen_manifest_sha256=sha(ROOT/'inputs.jsonl'),media_byte_identity_verified=True,
      portable_changes='media paths only',task_counts=summary['task_counts'],model_inference_run=False))
    print('package built',len(mapping),'media files',flush=True)

def finalize():
    check=json.loads((PACKAGE/'browser_validation.json').read_text());assert check['cards']==60 and check['status']=='passed'
    files=[p for p in sorted(PACKAGE.rglob('*')) if p.is_file() and p.name!='SHA256SUMS']
    (PACKAGE/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(PACKAGE)}\n' for p in files))
    archive=PACKAGE.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(PACKAGE.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(PACKAGE.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    receipt=dict(file=str(archive.resolve()),bytes=archive.stat().st_size,sha256=sha(archive),cards=60,
      remote_path='/apps/bypy/lance_review/'+archive.name,status='ready_for_upload')
    save(archive.with_suffix('.sha256.json'),receipt);print(json.dumps(receipt),flush=True)

def upload(receipt_name='bench_delivery_receipt.json'):
    import requests
    sys.path.insert(0,str(PROJECT/'lance_general_edit_finetune/1_specialist_filtering/scripts'))
    import upload_baidu_xpan as base
    helperpath=PROJECT/'dataset_research/recent_motion_review_20261030/deliver.py'
    spec=importlib.util.spec_from_file_location('delivery_listing',helperpath);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    archive=PACKAGE.with_suffix('.zip');side=archive.with_suffix('.sha256.json');d=json.loads(side.read_text())
    token=base.load_token(helper.CONFIG);statepath=archive.with_suffix('.baidu_upload.json')
    session=requests.Session();session.headers['User-Agent']='lance-review-uploader/1.0'
    hashes=base.block_hashes(archive,4*1024*1024)
    if statepath.exists():state=json.loads(statepath.read_text());assert state['sha256']==d['sha256']
    else:
        assert not any(r.get('path')==d['remote_path'] for r in helper.listing(token)),'remote name already exists'
        result=base.retry_call('precreate',5,lambda:base.precreate(session,token,d['remote_path'],d['bytes'],hashes))
        state=dict(sha256=d['sha256'],uploadid=result['uploadid'],needed=result.get('block_list',list(range(len(hashes)))),completed=[],status='uploading');save(statepath,state)
    completed=set(state['completed']);thread=threading.local()
    def work(index):
        if not hasattr(thread,'session'):thread.session=requests.Session()
        with archive.open('rb') as f:f.seek(index*4*1024*1024);block=f.read(4*1024*1024)
        result=base.retry_call('upload block',5,lambda:base.upload_block(thread.session,token,d['remote_path'],state['uploadid'],index,block))
        assert not result.get('md5') or result['md5']==hashes[index]
        return index
    if state['status']!='complete':
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(work,int(i)) for i in state['needed'] if int(i) not in completed]):
                completed.add(future.result());state['completed']=sorted(completed);save(statepath,state)
                if len(completed)%20==0:print('uploaded blocks',len(completed),'/',len(state['needed']),flush=True)
        result=base.retry_call('create',5,lambda:base.create_file(session,token,d['remote_path'],d['bytes'],state['uploadid'],hashes))
        state.update(status='complete',fs_id=result['fs_id']);save(statepath,state)
    rows=[r for r in helper.listing(token) if r.get('path')==d['remote_path']]
    assert len(rows)==1 and rows[0]['size']==d['bytes'] and rows[0]['fs_id']==state['fs_id']
    d.update(status='uploaded',remote_fs_id=state['fs_id'],remote_listing_verified=True,remote_full_download_verification=False)
    save(side,d);helper.upload(side,str(Path(d['remote_path']).with_suffix('.sha256.json')))
    save(ROOT/'review'/receipt_name,d);print(json.dumps(d),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--finalize',action='store_true');p.add_argument('--upload',action='store_true');a=p.parse_args()
    if a.upload:upload()
    elif a.finalize:finalize()
    else:build()
