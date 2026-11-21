"""Build self-contained offline review and paired historical timing report; no upload."""
from common import *
import html, shutil, zipfile, time

def main():
    dest=ROOT/'review';dest.mkdir(exist_ok=True);media=dest/'media';media.mkdir(exist_ok=True)
    inputs=rows(ROOT/'metadata/inputs.jsonl');gt={r['id']:r for r in read(BASE/'metadata/review_targets.json')}
    timing=[];sections=[];inventory={};known=[]
    def asset(path,expected=None):
        path=Path(path);digest=sha(path)
        if expected: assert digest==expected
        name=digest+path.suffix;target=media/name
        if not target.exists(): shutil.copy2(path,target)
        assert sha(target)==digest
        inventory['media/'+name]={'sha256':digest,'bytes':target.stat().st_size}
        return 'media/'+name
    def video(label,url):
        return f'<figure><figcaption>{html.escape(label)}</figcaption><video controls muted playsinline preload="metadata" src="{url}"></video><a href="{url}">完整视频</a></figure>'
    for item in inputs:
        sid=item['id'];old=read(BASE/f'outputs/ltx_base/{sid}.json');p=ROOT/f'outputs/{sid}.json';new=read(p) if p.exists() else {'status':'pending'}
        oldsec=old['denoise_seconds']+old['decode_seconds'];done=new['status']=='complete'
        record={'id':sid,'status':new['status'],'historical_denoise_seconds':old['denoise_seconds'],'historical_decode_seconds':old['decode_seconds'],'historical_stage_seconds':oldsec}
        if done:
            assert new['protocol_audit']['actual_updates']==8 and new['full_decode_frames']==121
            # Early worker was already loaded before timing field refinement.
            # Normalize the report only; preserve all original per-output records.
            if 'vae_write_seconds' not in new:
                new['vae_write_seconds']=new['vae_decode_seconds']+new['write_seconds']
                new['decode_seconds']=new['vae_write_seconds']+new['validation_seconds']
            newsec=new['denoise_seconds']+new['decode_seconds'];record.update({k:new[k] for k in ['model_load_seconds','cache_load_seconds','denoise_seconds','vae_decode_seconds','write_seconds','vae_write_seconds','decode_seconds','validation_seconds','elapsed_seconds','gpu_model','denoise_peak_allocated_bytes','decode_peak_allocated_bytes']});record.update(distilled_stage_seconds=newsec,speedup=oldsec/newsec,actual_updates=8)
        else: record['failure_reason']=new.get('failure_reason',new.get('error','pending'))
        timing.append(record)
        title=html.escape(sid+' / '+item['uid'])
        body=video('SOURCE',asset(item['source']['absolute_path'],item['source']['sha256']))
        for i,r in enumerate(item['references']):
            url=asset(r['absolute_path'],r['sha256']);label=f'参考 {i+1} / {r["type"]} / {r.get("view", "")}'
            body+=video(label,url) if Path(url).suffix=='.mp4' else f'<figure><figcaption>{html.escape(label)}</figcaption><img src="{url}"></figure>'
        body+=video('GT（仅审核）',asset(gt[sid]['frozen_gt_path'],gt[sid]['gt_sha256']))
        body+=video('旧 dev 复用 / 40步 CFG3 STG1 rescale0.7',asset(BASE/f'outputs/ltx_base/{sid}.mp4',old['output_sha256']))
        if done: body+=video('distilled 新运行 / 8步 CFG1 STG0 rescale0',asset(ROOT/f'outputs/{sid}.mp4',new['output_sha256']))
        else: body+=f'<p>distilled: {html.escape(new["status"])} — {html.escape(record["failure_reason"])}</p>'
        sections.append(f'<section id="{sid}"><h2>{title}</h2><p>{html.escape(item["prompt"]["en"])}</p><p>{html.escape(item["prompt"]["zh"])}</p><button class="play">同步播放 0–10秒</button> <button class="pause">暂停</button> <input class="seek" type="range" min="0" max="10" step="0.083333333" value="0"><div class="grid">{body}</div><p>人工观察：身份/服装迁移、动作、内容保持、时序、画质，待审核；本页不自动判定通过。</p><textarea placeholder="本地审核笔记（需手动另存；刷新不会保存）"></textarea></section>')
    paired=[r for r in timing if r['status']=='complete']
    def summary(data):
        return {'paired_count':len(data),'historical_stage_sum_seconds':sum(r['historical_stage_seconds'] for r in data),'distilled_stage_sum_seconds':sum(r['distilled_stage_seconds'] for r in data),'speedup':sum(r['historical_stage_seconds'] for r in data)/sum(r['distilled_stage_seconds'] for r in data) if data else None}
    report={'experiment':'LTX2.5 distilled＋DataMake V7 单阶段实验','rows':timing,'all_10_speedup':summary(paired)['speedup'] if len(paired)==10 else None,'available_pairs':summary(paired),'excluding_first_run':summary([r for r in paired if r['id']!=ORDER[0]]),'limitations':'历史运行比较；软件版本、缓存、资源竞争可能不同，非严格同场基准。两边decode_seconds均包含VAE、编码写盘、完整视频解码验证；新报告另列VAE、编码写盘和验证时间。主要比较denoise_seconds + decode_seconds；另列不含验证的辅助口径。首条保留。四卡仅可理想外推四个独立worker，未实测。仅UE，不外推CoinVE/TripVVT。','failures':[r for r in timing if r['status']!='complete']}
    for r in paired:
        r['stage_without_validation_seconds']=r['denoise_seconds']+r['vae_write_seconds']
        r['speedup_excluding_new_validation']=r['historical_stage_seconds']/r['stage_without_validation_seconds']
    atomic(ROOT/'metadata/speed_report.json',report);atomic(dest/'speed_report.json',report)
    lines=['# LTX2.5 distilled＋DataMake V7 单阶段实验','',report['limitations'],'','| 样本 | 状态 | 旧dev阶段(s) | 蒸馏去噪(s) | VAE(s) | 写盘(s) | 蒸馏阶段(s) | 配对加速 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in timing:
        fmt=lambda k:f'{r[k]:.3f}' if k in r else '—'
        lines.append('| '+' | '.join([r['id'],r['status'],fmt('historical_stage_seconds'),fmt('denoise_seconds'),fmt('vae_decode_seconds'),fmt('write_seconds'),fmt('distilled_stage_seconds'),fmt('speedup')])+' |')
    lines+=['',f'完成配对：{len(paired)}/10；10条总体加速比：{report["all_10_speedup"]}。',f'现有配对汇总：{json.dumps(report["available_pairs"])}',f'排除首条辅助统计：{json.dumps(report["excluding_first_run"])}','', '模型加载时间、逐条总耗时、显存峰值及失败原因见 speed_report.json；软件版本见 runtime.json。','', '可见差异尚待人工查看；不将技术检查通过等同编辑效果通过。']
    (ROOT/'SPEED_REPORT.md').write_text('\n'.join(lines)+'\n');shutil.copy2(ROOT/'SPEED_REPORT.md',dest/'SPEED_REPORT.md')
    nav=' '.join(f'<a href="#{i}">{i}</a>' for i in IDS)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>LTX2.5 distilled＋DataMake V7 单阶段实验</title><style>body{font:16px system-ui;margin:0;background:#11151b;color:#e8eef6}header,section{padding:24px}nav{position:sticky;top:0;padding:12px;background:#27303f;z-index:5}a{color:#90caff;margin:5px}section{border-bottom:2px solid #667;scroll-margin-top:70px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}figure{margin:0}video,img{width:100%;max-height:420px;object-fit:contain;background:#000}textarea{width:95%;height:90px}input{width:40%}button{padding:10px}</style><header><h1>LTX2.5 distilled＋DataMake V7 单阶段实验</h1><p>旧 dev 复用 / 蒸馏新运行。121帧，12fps，seed42，原画幅，参考 → SOURCE → 生成区。GT仅审核。</p><p>旧权重：dev BF16 / 792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584；新权重：distilled BF16 / 31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4。两者revision 49b093a39c19169a8a81721dcc541e07e945232f。</p><p>新：单阶段 ancestral Euler，eta=1，s_noise=1，noise seed10042，CFG1/STG0/rescale0/ref_cfg1，无LoRA。sigma=[1,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0]。旧：40步 LTX2Scheduler，CFG3/STG1/rescale0.7/ref_cfg1，无训练LoRA。</p><p>历史比较非严格同场基准；仅10条UE。不自动评分。<a href="SPEED_REPORT.md">速度报告</a> <a href="speed_report.json">完整数据</a></p></header>'''+f'<nav>{nav}</nav>'+''.join(sections)+'''<script>document.querySelectorAll('section').forEach(s=>{const vs=[...s.querySelectorAll('video')],seek=s.querySelector('.seek');s.querySelector('.play').onclick=()=>{vs.forEach(v=>{v.currentTime=+seek.value;v.play().catch(()=>{})})};s.querySelector('.pause').onclick=()=>vs.forEach(v=>v.pause());seek.oninput=()=>vs.forEach(v=>v.currentTime=+seek.value);if(vs[0])vs[0].ontimeupdate=()=>{if(vs[0].currentTime>=10){vs.forEach(v=>v.pause());seek.value=10}else{seek.value=vs[0].currentTime;vs.slice(1).forEach(v=>{if(!vs[0].paused&&Math.abs(v.currentTime-vs[0].currentTime)>.15)v.currentTime=vs[0].currentTime})}}});</script></html>'''
    (dest/'index.html').write_text(page)
    for name in ['runtime.json','pairing.json','cpu_tests.json']:
        if (ROOT/'metadata'/name).exists():shutil.copy2(ROOT/'metadata'/name,dest/name)
    atomic(dest/'media_inventory.json',inventory)
    tmp=ROOT/'review.zip.tmp'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_STORED) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file(): z.write(p,Path('review')/p.relative_to(dest))
    tmp.replace(ROOT/'review.zip')
    with zipfile.ZipFile(ROOT/'review.zip') as z: assert z.testzip() is None
    atomic(ROOT/'metadata/package_status.json',{'status':'complete','successful_outputs':len(paired),'total':10,'zip_sha256':sha(ROOT/'review.zip'),'media_count':len(inventory),'all_media_hashes_verified':True,'zip_crc_verified':True,'created_unix':time.time()})
    print('Review package ready',len(paired),'/10',flush=True)
if __name__=='__main__':main()
