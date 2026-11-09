from pathlib import Path
import json,shutil,hashlib,html,zipfile
import av
from workflow import PROMPT,CASES
R=Path(__file__).resolve().parent;P=R/'review/ltx25_v2v_nomask_scroll_20261031'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    for d in ['media','provenance','provenance/records','provenance/workflows']:(P/d).mkdir(parents=True,exist_ok=True)
    manifest=[]
    def copy(p,rel):
        p=Path(p);dst=P/rel;shutil.copy2(p,dst);assert sha(p)==sha(dst)
        if p.suffix=='.mp4':
            with av.open(str(dst)) as c:
                s=c.streams.video[0];n=sum(1 for _ in c.decode(s));assert n>0
                manifest.append(dict(file=rel,sha256=sha(dst),frames=n,width=s.width,height=s.height,fps=float(s.average_rate)))
        return rel
    def panel(rel,label):return '<div><h3>'+html.escape(label)+'</h3><video controls muted playsinline preload="none" src="'+rel+'"></video><a href="'+rel+'">打开视频</a></div>'
    src=copy(R/'assets/source49.mp4','media/source49.mp4');original=json.loads((R/'metadata/source_row.json').read_text())
    orig=copy(R.parents[1]/original['source_video'],'media/original_source.mp4')
    cards=['<article><h2>输入视频</h2><div class="panels">'+panel(orig,'原始 bench 素材')+panel(src,'实际推理输入：768×448，49帧，24FPS')+'</div></article>']
    labels={'v2v_deblur':'官方 V2V 默认 Deblur adapter（非加墨镜专用）','v2v_no_adapter':'对照：V2V 全 RGB 参考，无 adapter','v2v_inpaint_fullrgb':'消融：Inpainting adapter＋完整 RGB 参考，不提供 mask/绿色提示'}
    blocked=[]
    for sid,adapter in CASES:
        rec=R/f'outputs/records/{sid}.json'
        if not rec.exists():continue
        d=json.loads(rec.read_text())
        if d.get('status')!='generated':blocked.append(d);continue
        p=Path(d['output']);rel=copy(p,'media/'+p.name)
        cards.append('<article><h2>'+html.escape(labels[sid])+'</h2><p>无 mask、无绿色预处理、无 Laplacian 融合；8步，CFG1，seed43。adapter：'+html.escape(str(adapter))+'</p><div class="panels">'+panel(src,'原视频：实际输入')+panel(rel,'第一阶段原始输出')+'</div></article>')
    old=R.parent/'ltx25_official_edit_routes_20261031';rel=copy(old/'outputs/generated/inpaint_stage1_raw.mp4','media/previous_masked_stage1_raw.mp4')
    cards.append('<article><h2>此前结果：带人工眼部 mask 的 Inpainting 第一阶段原始输出</h2><p>仅作对照。该视频有额外 mask 条件，且未进行 Laplacian 融合；其 silent audio 路径与本次不同，不是严格仅改变 mask 的单变量实验。</p><div class="panels">'+panel(src,'原视频')+panel(rel,'此前：带 mask，第一阶段融合前')+'</div></article>')
    report=(R/'metadata/findings.txt').read_text()
    if blocked:report+='\n未完成的配置：\n'+json.dumps(blocked,ensure_ascii=False,indent=2)
    css='body{background:#101720;color:#eee;max-width:1500px;margin:20px auto;padding:18px;font:16px/1.7 system-ui}article{border:1px solid #465464;margin:22px 0;padding:18px}.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}video{width:100%;max-height:640px;background:black}a{color:#8bd1ff}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:750px){.panels{grid-template-columns:1fr}}'
    intro='<h1>LTX-2.5 无 mask V2V · 加黑色墨镜</h1><p>输入：原视频＋下方目标描述。没有目标参考图、没有 mask、没有绿色填充、没有融合。只取单阶段原始模型输出。最后一项为此前带 mask 的对照，明确分开。</p><pre>'+html.escape(PROMPT)+'</pre><pre>'+html.escape(report)+'</pre>'
    (P/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>无 mask V2V 测试</title><style>'+css+'</style>'+intro+''.join(cards)+'</html>')
    (P/'REPORT.txt').write_text(report);(P/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    for p in (R/'metadata').iterdir():
        if p.is_file() and p.suffix in ['.json','.txt','.jpg']:shutil.copy2(p,P/'provenance'/p.name)
    for folder in ['records','workflows']:
        for p in (R/f'outputs/{folder}').glob('*.json'):shutil.copy2(p,P/f'provenance/{folder}'/p.name)
    for name in ['workflow.py','route_io.py','run_worker.py']:shutil.copy2(R/name,P/'provenance'/name)
    (P/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(P))+'\n' for p in sorted(P.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    archive=P.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(P.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(P.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    receipt=dict(status='ready_local',file=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),videos=len(manifest),cards=len(cards),new_outputs=len(cards)-2,blocked_cases=len(blocked))
    (R/'review/package_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
