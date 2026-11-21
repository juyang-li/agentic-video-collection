"""CPU current-result review; main96 plus the completed4 Identity cases."""
import csv,subprocess,sys,zipfile,re,time
from html.parser import HTMLParser
from common import *
from score import DIAG


def main():
 subprocess.run([sys.executable,str(ROOT/'scripts/aggregate.py')],check=True)
 for filename in ['per_video_metrics.csv','per_task_report.csv']:
  with (ROOT/'metadata'/filename).open() as f:
   reader=csv.DictReader(f);fields=reader.fieldnames;records=[r for r in reader if r['cohort']=='identity_diagnostic']
  with (DIAG/'metadata'/filename).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
 (DIAG/'metadata/report.md').write_text('# Identity完整头部参考诊断 · 运行中指标快照\n\n4条任务、12条输出已生成并直接审核。02/03/04在三个模型中均未呈现清晰参考面部身份迁移；全部结果保留，按用户要求加入补充案例供直接审核。六项质量指标及静态筛选尚未全部完成，缺失留空，不能视为最终指标报告。\n\n逐模型判断见identity_review.json，逐视频已算分数见per_video_metrics.csv，编辑效果与指标联表见per_task_report.csv。\n')
 subprocess.run([sys.executable,str(ROOT/'scripts/review.py'),'--stage','all'],check=True)
 subprocess.run([sys.executable,str(DIAG/'scripts/review.py'),'--stage','all','--no-helpers'],check=True)
 count=read(ROOT/'metadata/all_media_validation.json')['complete_generations']
 diag_count=read(DIAG/'metadata/all_media_validation.json')['complete_generations'];assert diag_count==12
 label=time.strftime('%Y%m%d_%H%M');name=f'a849_v3_results_snapshot_{label}';output=ROOT/'review'/f'{name}.zip'
 helper=re.compile(r'<figure><figcaption>VACE (?:mask overlay|prepared input)</figcaption>.*?</figure>',re.S)
 contents={};pages={}
 for root,prefix in [(ROOT,Path()),(DIAG,Path('identity_diagnostic'))]:
  directory=root/'review/all'
  for p in directory.rglob('*'):
   if not p.is_file():continue
   relative=p.relative_to(directory)
   if root==DIAG and relative.parts[0]=='media' and ('overlay' in p.name or 'conditioning_input' in p.name):continue
   if p.suffix=='.html':
    text=p.read_text()
    if root==DIAG:text=helper.sub('',text)
    else:
     notice=f'<p><strong>当前结果快照：主测已完成{count}/288条输出，另含4条Identity补充案例的12/12条输出。尚缺主测{288-count}条输出及部分评分；新增4条非总统/副总统Celeb hair案例将随后追加，当前不算作完成。</strong> <a href="identity_diagnostic/index.html">查看4条Identity补充案例及三模型结果</a></p>'
     text=text.replace('<h1>',notice+'<h1>',1)
    pages[prefix/relative]=text
   else:contents[prefix/relative]=p
 pages[Path('README.txt')]=f'当前已推理结果审核快照\n\n解压后打开index.html。主测96条，已完成{count}/288条模型输出；另附4条Identity补充案例、全部12条结果。新加4条非总统/副总统Celeb hair案例尚待准备和生成，不包含在此快照。\n\n固定导航、完整source/参考/指令/模型输出，TripVVT含审核用GT。缺失结果逐条标注；评分尚未全部完成，缺失不填零；有效低质量结果不隐藏。VACE helper仅保留整个包前两个主测样例。后续提供104条案例完整包，正式96条主测与8条补充案例分别报告。\n'
 atomic(ROOT/'metadata/current_results_snapshot.json',dict(status='snapshot_not_final',created_unix=time.time(),main_tasks=96,main_completed_outputs=count,main_expected_outputs=288,identity_supplement_tasks=4,identity_completed_outputs=12,new_hair_supplement_tasks_pending=4,planned_total_cases=104,local=str(output),remote=f'/apps/bypy/lance_review/{output.name}',no_vlm_api=True))
 contents[Path('snapshot_status.json')]=ROOT/'metadata/current_results_snapshot.json'
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.links=[]
  def handle_starttag(self,tag,attrs):
   for key,value in attrs:
    if key in ['src','href'] and value and not value.startswith('#'):
     assert not value.startswith(('http:','https:','/'));self.links.append(value)
 members=set(contents)|set(pages)
 for path,page in pages.items():
  if path.suffix!='.html':continue
  parser=Links();parser.feed(page)
  for relative in parser.links:assert path.parent/relative in members,(path,relative)
 with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
  for path,source in sorted(contents.items()):archive.write(source,Path(name)/path)
  for path,page in pages.items():archive.writestr(str(Path(name)/path),page)
 with zipfile.ZipFile(output) as archive:assert archive.testzip() is None
 status=read(ROOT/'metadata/current_results_snapshot.json');status.update(bytes=output.stat().st_size,local_link_and_crc_validation='passed');atomic(ROOT/'metadata/current_results_snapshot.json',status)
 print('Current result ZIP ready',output,count,'main outputs +12 Identity outputs',output.stat().st_size,flush=True)
if __name__=='__main__':main()
