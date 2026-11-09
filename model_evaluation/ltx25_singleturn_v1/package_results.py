"""Export all frozen benchmark outputs and reference controls without scoring."""
from pathlib import Path
from collections import Counter
import argparse,base64,hashlib,html,io,json,os,re,shutil,statistics,zipfile
from fontTools import subset
from fontTools.ttLib import TTFont
import package_bench as bench

ROOT=Path(__file__).resolve().parent
NAME='ltx25_bench60_results_scroll_20261030'
PACKAGE=ROOT/'review'/NAME
SOURCE=ROOT/'review/ltx25_frozen_bench60_scroll_20261030'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def text(p,s):
    tmp=p.with_name(p.name+'.new');tmp.write_text(s);tmp.replace(p)
def save(p,d):text(p,json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def build():
    inputs=[json.loads(s) for s in (SOURCE/'inputs.jsonl').read_text().splitlines()]
    controls=json.loads((ROOT/'reference_controls.json').read_text())
    ids=[r['sample_id'] for r in inputs]+[r['sample_id'] for r in controls]
    records={sid:json.loads((ROOT/f'outputs/records/{sid}.json').read_text()) for sid in ids}
    assert len(records)==64 and all(r['status'] in ['generated','recovered_tmp'] and Path(r['output']).is_file() for r in records.values())
    assert len({r['config_sha256'] for r in records.values()})==1,'mixed inference configurations'
    if not PACKAGE.exists():shutil.copytree(SOURCE,PACKAGE,copy_function=os.link)
    (PACKAGE/'generated').mkdir(exist_ok=True);(PACKAGE/'provenance').mkdir(exist_ok=True)
    for sid,r in records.items():
        out=PACKAGE/f'generated/{sid}.mp4'
        if not out.exists():shutil.copy2(r['output'],out)
        assert sha(out)==sha(Path(r['output']))
    if (ROOT/'outputs/generated/official_reference_only.mp4').is_file():
        shutil.copy2(ROOT/'outputs/generated/official_reference_only.mp4',PACKAGE/'provenance/official_reference_only.mp4')
    for name in ['model_lock.json','run_config.json','smoke_status.json','supervisor_status.json','final_inference_audit.json','resource_accounting.json','environment_freeze.txt','comfyui_revision.txt','ltx_nodes_revision.txt']:
        if (ROOT/'metadata'/name).exists():shutil.copy2(ROOT/'metadata'/name,PACKAGE/'provenance'/name)
    shutil.copytree(ROOT/'outputs/workflows',PACKAGE/'provenance/workflows',dirs_exist_ok=True)
    shutil.copytree(ROOT/'outputs/records',PACKAGE/'provenance/records',dirs_exist_ok=True)
    for name in ['workflow.py','bench_nodes.py','run_worker.py','infer.sbatch','EXECUTION.md']:shutil.copy2(ROOT/name,PACKAGE/'provenance'/name)
    byid={r['sample_id']:r for r in inputs};esc=html.escape
    def panel(sid,label):return f'<div class="panel"><h4>{esc(label)}</h4><video controls muted playsinline preload="none" data-src="generated/{sid}.mp4"></video><a href="generated/{sid}.mp4" target="_blank">打开模型输出</a></div>'
    page=(SOURCE/'index.html').read_text()
    page=page.replace('LTX-2.5 冻结 bench · 60 条','LTX-2.5 推理结果 · 60 条＋4 个对照').replace('待推理输入审核','LTX-2.5 单轮编辑推理结果')
    page=page.replace('LTX-2.5 已冻结 bench · 60 条 · 无动作任务','LTX-2.5＋Ingredients 实验编辑 · 60 条＋4 个对照')
    page=page.replace('原始素材与完整英文指令，未运行模型。','原始素材与完整英文指令；新增 LTX-2.5＋Ingredients 实验接线输出。模型使用最多97帧、24 FPS、约4秒的原速片段，输入素材保留完整原文件。')
    page=page.replace('</nav>','<a href="#controls">参考图替换对照 · 4</a><a href="report.html">运行报告</a></nav>',1)
    for row in inputs:
        sid=row['sample_id'];begin=page.index(f'<article id="{sid}">');end=page.index('</article>',begin)+len('</article>')
        block=page[begin:end]
        block=block.replace('</div><button class="play">',panel(sid,'LTX-2.5 输出 · 实验编辑路径')+'</div><button class="play">',1)
        spec=records[sid]['media']
        block=block.replace('</article>',f'<p class="meta">模型输出：{spec["width"]}×{spec["height"]} · {spec["frames"]}帧 · {spec["fps"]:g} FPS · seed=42</p></article>')
        page=page[:begin]+block+page[end:]
    extra=['<section id="controls"><h2>参考图替换对照 · 4 条（不计入正式60条）</h2><p>固定同一source、prompt、seed和模型，仅替换服装参考图。输出有变化不等于正确跟随衣服，需结合参考图判断。</p>']
    for c in controls:
        a=byid[c['base_sample_id']];b=byid[c['replacement_reference_from']];sid=c['sample_id']
        extra.append(f'<article id="{sid}"><h3>{sid}</h3><p class="prompt">{esc(a["instruction"])}</p><div class="panels"><div class="panel"><h4>原参考图</h4><img loading="lazy" src="{a["reference_image"]}"></div>'+panel(a['sample_id'],'原参考图对应输出')+f'<div class="panel"><h4>替换后的参考图</h4><img loading="lazy" src="{b["reference_image"]}"></div>'+panel(sid,'替换参考图后输出')+'</div><button class="play">本条视频从头播放</button><button class="pause">暂停</button></article>')
    extra.append('</section>');page=page.replace('</main>',''.join(extra)+'</main>')
    normal=[records[r['sample_id']] for r in inputs]
    times=[r['seconds'] for r in normal if 'seconds' in r]
    report=f'''# LTX-2.5 单轮编辑首测结果

已完成60条正式输入与4条参考图替换对照。另有1个参考图生成的技术烟测示例，放在provenance/，不计入正式结果。

模型为LTX-2.5 BF16 distilled + 官方Ingredients 0.9，全部样本使用同一起点。此处不是经过通用指令编辑训练的裸基模结果：我们按用户同意的实验方案，用逐帧RGB source guide（frame_idx=0）和可选独立图像guide（frame_idx=-1）组合条件。

输入是冻结bench的原始英文指令，seed=42，8步、CFG=1、Euler ancestral、adapter强度1，无提示增强、无超分、未使用GT或目标caption。原速最多97帧@24 FPS；横屏768×448、竖屏448×768，等比例padding，短视频取不超过时长的合法8k+1帧数。

正式任务：参考图换衣25、外观15、文字换衣10、镜头10；动作未加入。换衣的SOURCE是TripVVT合成换装视频，GT是已有真实配对目标；其它任务没有编辑后GT。页面中的模型输出与GT明确分列。

本次正式样本记录的单条用时中位数为{statistics.median(times):.1f}秒，范围{min(times):.1f}–{max(times):.1f}秒。该数字不包含Slurm排队、模型下载和节点stage时间。两次正式作业分别运行4分55秒和4分27秒；第一次的两个分片因状态文件写入竞争退出，其余结果在第二次复用。未采集精确GPU峰值显存，Slurm记录也未提供该指标，不将CPU MaxRSS冒充显存，更不为补这个字段重新推理。模型、代码revision及依赖版本见provenance/。60条正式样本和4条对照采用同一配置标识：{normal[0]['config_sha256']}。

初步观察仅限烟测中抽看的两条换衣样本：原视频内容较好保留，但没有明显按参考图换成目标衣服。这不是全部60条的人工通过率，也不能据此直接判断SFT有效或无效。参考图替换对照可帮助观察条件影响；输出不同同样不能单独证明正确迁移目标服装。请以完整视频审阅为准。

技术过程：srun六条烟测先通过，再用sbatch正式运行。修复了loopback代理、非连续帧编码和并发状态文件临时路径竞争；已完成输出在续跑时复用。推理过程中没有按效果筛掉样本或逐条调参，所有正式结果都包含在包内。

打开index.html即可连续下拉，无评分栏。generated/有64份模型输出，generation_manifest.jsonl记录对应关系，inputs.jsonl为可移植输入；evaluation/仅供评测，不应输入模型。frozen/保留原始冻结清单，provenance/保留完整运行证据。该包只包含模型推理结果及审阅素材，不包含大型模型权重或凭据。
'''
    text(PACKAGE/'REPORT.md',report)
    fontpath=bench.PROJECT/'dataset_research/ref_edit_review_20261030/assets/font/NotoSansCJKsc-Regular.otf'
    font=TTFont(fontpath);sub=subset.Subsetter();sub.populate(text=page+report);sub.subset(font);buf=io.BytesIO();font.save(buf)
    face="@font-face{font-family:CJK;src:url(data:font/otf;base64,"+base64.b64encode(buf.getvalue()).decode()+") format('opentype');font-display:block}"
    page=re.sub(r'@font-face\{[^}]+\}',lambda m:face,page,count=1);text(PACKAGE/'index.html',page)
    text(PACKAGE/'report.html','<!doctype html><meta charset="utf-8"><title>LTX 推理报告</title><style>'+face+'body{max-width:1100px;margin:30px auto;padding:20px;background:#101720;color:#eaf0f7;font:16px/1.8 CJK,system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}a{color:#8bd1ff}</style><a href="index.html">返回结果</a><pre>'+esc(report)+'</pre>')
    text(PACKAGE/'README.md','# LTX-2.5 推理结果审核包\n\n解压后打开index.html，连续下拉，无评分栏。正式60条+参考图替换对照4条，所有素材与完整prompt均包含。模型使用LTX-2.5+Ingredients实验接线，具体配置和局限见REPORT.md。GT与模型输出分列，输入原素材长度与模型约4秒片段可能不同。\n')
    manifest=[]
    control_map={c['sample_id']:c for c in controls}
    for sid in ids:
        r=records[sid]
        base=byid[control_map[sid]['base_sample_id']] if sid in control_map else byid[sid]
        ref=byid[control_map[sid]['replacement_reference_from']] if sid in control_map else base
        manifest.append({**r,'original_source_video':r['source_video'],'original_reference_image':r.get('reference_image'),
          'source_video':base['source_video'],'source_sha256':base['source_sha256'],
          'reference_image':ref['reference_image'],'reference_sha256':ref['reference_sha256'],
          'output':f'generated/{sid}.mp4','output_sha256':sha(PACKAGE/f'generated/{sid}.mp4')})
    text(PACKAGE/'generation_manifest.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest))
    save(PACKAGE/'package_validation.json',dict(status='passed',formal_outputs=60,diagnostic_outputs=4,model_configurations=1,
        output_byte_identity_verified=True,frozen_manifest_sha256=sha(ROOT/'inputs.jsonl'),model_inference_run=True))
    (PACKAGE/'browser_validation.json').unlink(missing_ok=True);(PACKAGE/'SHA256SUMS').unlink(missing_ok=True)
    print('Complete result package built',flush=True)

def finalize():
    check=json.loads((PACKAGE/'browser_validation.json').read_text());assert check['status']=='passed' and check['cards']==64
    assert check['html_sha256']==sha(PACKAGE/'index.html')
    files=[p for p in sorted(PACKAGE.rglob('*')) if p.is_file() and p.name!='SHA256SUMS']
    text(PACKAGE/'SHA256SUMS',''.join(f'{sha(p)}  {p.relative_to(PACKAGE)}\n' for p in files))
    archive=PACKAGE.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(PACKAGE.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(PACKAGE.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    d=dict(file=str(archive.resolve()),bytes=archive.stat().st_size,sha256=sha(archive),cards=64,
           formal_outputs=60,diagnostic_outputs=4,remote_path='/apps/bypy/lance_review/'+archive.name,status='ready_for_upload')
    save(archive.with_suffix('.sha256.json'),d);print(json.dumps(d),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--finalize',action='store_true');p.add_argument('--upload',action='store_true');a=p.parse_args()
    if a.upload:bench.PACKAGE=PACKAGE;bench.upload(receipt_name='results_delivery_receipt.json')
    elif a.finalize:finalize()
    else:build()
