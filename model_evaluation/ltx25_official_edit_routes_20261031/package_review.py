from pathlib import Path
import json,shutil,hashlib,html,zipfile
import av
from workflow import PROMPT
R=Path(__file__).resolve().parent;P=R/'review/ltx25_official_edit_routes_scroll_20261031'
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
    src=copy(R/'assets/appearance_002_source49.mp4','media/source49.mp4')
    original=json.loads((R/'metadata/source_row.json').read_text());orig=copy(R.parents[1]/original['source_video'],'media/original_source.mp4')
    mask=copy(R/'assets/eye_mask.mp4','media/eye_mask.mp4');overlay=copy(R/'assets/eye_mask_overlay.mp4','media/eye_mask_overlay.mp4')
    cards=['<article><h2>输入与额外条件</h2><div class="panels">'+panel(orig,'原始 bench 视频')+panel(src,'实际模型输入：49帧，24FPS，768×448')+panel(overlay,'人工眼部 mask 叠加：仅表示待编辑区域')+panel(mask,'mask：白色区域允许编辑')+'</div></article>']
    for name,label in [('retake_distilled','原生 Retake · distilled'),('retake_dev','原生 Retake · dev')]:
        p=R/f'outputs/generated/{name}.mp4';rec=R/f'outputs/records/{name}.json'
        if not p.exists() or json.loads(rec.read_text()).get('status')!='generated':continue
        rel=copy(p,'media/'+p.name)
        cards.append('<article><h2>'+label+'</h2><p>无额外 adapter，无空间 mask。重生成请求窗口为 1/3–5/3 秒；时间边界受 latent 分块影响。输出全片静音。</p><div class="panels">'+panel(src,'输入')+panel(rel,label)+'</div></article>')
    for name,label in [('inpaint_final','Inpainting · 官方两阶段最终融合'),('inpaint_stage1_raw','诊断：第一阶段原始模型输出'),('inpaint_stage1_blended','诊断：第一阶段融合'),('inpaint_stage2_raw','诊断：第二阶段原始模型输出')]:
        p=R/f'outputs/generated/{name}.mp4'
        if not p.exists():continue
        rel=copy(p,'media/'+p.name)
        cards.append('<article><h2>'+label+'</h2><p>额外输入人工眼部 mask，使用官方 In-Outpainting adapter；区域外保持部分来自官方 Laplacian 融合，不能全部归因于模型。第一阶段 768×448，第二阶段 1536×896。</p><div class="panels">'+panel(src,'输入')+panel(rel,label)+'</div></article>')
    report=(R/'metadata/findings.txt').read_text() if (R/'metadata/findings.txt').exists() else '结果待人工审阅。'
    css='body{background:#101720;color:#eee;max-width:1500px;margin:20px auto;padding:18px;font:16px/1.7 system-ui}article{border:1px solid #465464;margin:22px 0;padding:18px}.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}video{width:100%;max-height:640px;background:black}a{color:#8bd1ff}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:750px){.panels{grid-template-columns:1fr}}'
    intro='<h1>LTX-2.5 官方视频编辑路线 · appearance_002</h1><p>原 bench 指令：'+html.escape(original.get('instruction','添加黑色墨镜'))+'</p><p>实际送入模型的完整目标描述：</p><pre>'+html.escape(PROMPT)+'</pre><p>无参考图片或参考视频；只有 Inpainting 增加了人工眼部 mask，因此不能作为原纯文本协议的成功结果。连续下拉，无评分控件。</p><pre>'+html.escape(report)+'</pre>'
    (P/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LTX 官方编辑测试</title><style>'+css+'</style>'+intro+''.join(cards)+'</html>')
    (P/'REPORT.txt').write_text(report)
    (P/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    for p in (R/'metadata').iterdir():
        if p.is_file() and p.suffix in ['.json','.txt','.jpg']:shutil.copy2(p,P/'provenance'/p.name)
    for folder in ['records','workflows']:
        for p in (R/f'outputs/{folder}').glob('*.json'):shutil.copy2(p,P/f'provenance/{folder}'/p.name)
    for name in ['workflow.py','route_io.py','run_retake.py','run_inpaint.py']:shutil.copy2(R/name,P/'provenance'/name)
    copy(R/'assets/eye_mask.npy','provenance/eye_mask.npy')
    files=[p for p in P.rglob('*') if p.is_file() and p.name!='SHA256SUMS']
    (P/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(P))+'\n' for p in sorted(files)))
    archive=P.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(P.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(P.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    receipt=dict(status='ready_local',file=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),videos=len(manifest),cards=len(cards))
    (R/'review/package_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
