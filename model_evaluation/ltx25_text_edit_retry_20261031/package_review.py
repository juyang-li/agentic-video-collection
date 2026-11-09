from pathlib import Path
import json,html,hashlib,shutil,zipfile
import av
R=Path(__file__).resolve().parent;PROJECT=R.parents[1];OLD=R.parent/'ltx25_base_text10_20261031'
P=R/'review/ltx25_text_edit_retry_scroll_20261031';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return [json.loads(s) for s in p.read_text().splitlines()]
def main():
 for d in ['media','generated','previous','provenance']:(P/d).mkdir(exist_ok=True,parents=True)
 files={};manifest=[];cards=[]
 def copy(src,rel):
  src=Path(src);dst=P/rel
  if not dst.exists():shutil.copy2(src,dst)
  assert sha(src)==sha(dst);files[rel]=sha(src);return rel
 def media(row):return copy(PROJECT/row['source_video'],'media/'+row.get('base_sample_id','appearance_002')+'.mp4')
 def output(row):
  sid=row['sample_id'];rec=json.loads((R/f'outputs/records/{sid}.json').read_text());assert rec['status'] in ['generated','recovered_tmp']
  rel=copy(rec['output'],f'generated/{sid}.mp4')
  with av.open(str(P/rel)) as c:
   s=c.streams.video[0];n=sum(1 for _ in c.decode(s));assert n==rec['media']['frames']
  manifest.append({**row,'source_video':media(row),'output':rel,'output_sha256':files[rel],'media':rec['media'],'config_sha256':rec['config_sha256']});return rel
 def panel(rel,label):return f'<div><h4>{label}</h4><video controls muted playsinline preload="none" src="{rel}"></video><a href="{rel}">打开视频</a></div>'
 probes=load(R/'probe_inputs.jsonl');rounds=[]
 for attempt in range(1,5):
  pair=[d for d in probes if d['attempt_round']==attempt]
  if not all((R/f"outputs/records/{d['sample_id']}.json").exists() for d in pair):continue
  try:outs=[output(d) for d in pair]
  except AssertionError:continue
  target=next(d for d in pair if not d['control']);rounds.append(attempt)
  review=json.loads((R/f'metadata/attempt{attempt}_review.json').read_text()) if (R/f'metadata/attempt{attempt}_review.json').exists() else {}
  explanation=review.get('evidence','待审阅')
  cards.append(f'<article id="attempt{attempt}"><h2>先行单例 · 第{attempt}次尝试</h2><p>原指令：{html.escape(target["bench_instruction"])}</p><p>实际目标描述：{html.escape(target["instruction"])}</p><p>route={target["route"]} · attention={target["attention_strength"]} · CFG={target["cfg"]} · steps={target["steps"]} · denoise={target.get("denoise","full")}</p><div class="panels">'+panel(media(target),'原bench视频')+panel(outs[0],'本次：目标prompt')+panel(outs[1],'本次：空prompt对照')+f'</div><p>{html.escape(explanation)}</p></article>')
 expansion=[]
 if (R/'expansion_inputs.jsonl').exists():
  for row in load(R/'expansion_inputs.jsonl'):
   sid=row['sample_id'];p=R/f'outputs/records/{sid}.json'
   if not p.exists():continue
   rec=json.loads(p.read_text())
   if rec.get('status') not in ['generated','recovered_tmp']:continue
   rel=output(row);expansion.append(sid);prev=copy(OLD/f"outputs/generated/{row['base_sample_id']}.mp4",f"previous/{row['base_sample_id']}.mp4")
   cards.append(f'<article id="{sid}"><h2>扩展测试 · {row["base_sample_id"]}</h2><p>原指令：{html.escape(row["bench_instruction"])}</p><p>本次目标描述：{html.escape(row["instruction"])}</p><div class="panels">'+panel(media(row),'原bench视频')+panel(rel,'本次：dev无adapter')+panel(prev,'上一轮：distilled无adapter')+'</div></article>')
 css='body{background:#101720;color:#eee;max-width:1600px;margin:30px auto;padding:20px;font:16px/1.7 system-ui}article{border:1px solid #465464;margin:20px 0;padding:20px}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}video{width:100%;max-height:600px;background:black}a{color:#8bd1ff}@media(max-width:800px){.panels{grid-template-columns:1fr}}'
 intro='<h1>原bench纯文本编辑复测 · 先跑通一条再扩展</h1><p>无任何LoRA。使用LTX-2.5 dev BF16与30步CFG，而非上一轮8步distilled。保留完整source视频。先试appearance_002加墨镜，目标描述与空prompt配对；最多四次尝试。只有确认目标编辑迹象才扩展十条。画面漂移、随机重绘不单独算成功。</p><p>原始指令与本次改写的目标描述分别显示；没有GT caption或额外参考图输入。连续下拉，无评分栏。<a href="REPORT.md">查看报告</a></p>'
 (P/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>文字编辑复测</title><style>'+css+'</style>'+intro+''.join(cards)+'</html>')
 (P/'manifest.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in manifest))
 report=(R/'metadata/final_findings.md').read_text() if (R/'metadata/final_findings.md').exists() else '# 复测记录\n\n语义结果待审阅。\n'
 (P/'REPORT.md').write_text(report)
 for p in (R/'metadata').glob('*'):
  if p.is_file() and p.suffix in ['.json','.txt','.md']:shutil.copy2(p,P/'provenance'/p.name)
 for image in (R/'metadata').glob('attempt*_*.jpg'):
  shutil.copy2(image,P/'provenance'/image.name)
 shutil.copytree(R/'outputs/records',P/'provenance/records',dirs_exist_ok=True);shutil.copytree(R/'outputs/workflows',P/'provenance/workflows',dirs_exist_ok=True)
 for name in ['workflow.py','run_worker.py']:shutil.copy2(R/name,P/'provenance'/name)
 (P/'SHA256SUMS').write_text(''.join(f'{v}  {k}\n' for k,v in sorted(files.items())))
 validation=dict(status='passed_media_mapping_and_decode',probe_rounds=rounds,expanded_cases=len(expansion),our_outputs=len(manifest),cards=len(cards),rating_controls=0)
 (P/'package_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
 archive=P.with_suffix('.zip')
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
  for p in sorted(P.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(P.parent))
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 receipt={**validation,'status':'ready_local','file':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)}
 receipt['status']='ready_local';(R/'review/package_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
