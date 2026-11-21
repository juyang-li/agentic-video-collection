"""CPU: portable frozen-input review ZIP; no generated model videos."""
import zipfile,html
from html.parser import HTMLParser
from common import *
from score import DIAG


def main():
 assert read(ROOT/'metadata/identity_protocol_correction.json')['status']=='validated'
 items=rows(ROOT/'metadata/inputs.jsonl');assert len(items)==96
 name='a849_v3_inputs96_20261114';output=ROOT/'review'/f'{name}.zip';files={};pages={}
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.links=[];self.ids=set();self.videos=0
  def handle_starttag(self,tag,attrs):
   attrs=dict(attrs)
   if 'id' in attrs:self.ids.add(attrs['id'])
   if tag=='video':self.videos+=1
   for key in ['src','href']:
    value=attrs.get(key)
    if value and not value.startswith('#'):
     assert not value.startswith(('http:','https:','/')),value
     self.links.append(value)
 def page(root,prefix):
  directory=root/'review/inputs';content=(directory/'index.html').read_text()
  if not prefix:
   notice='<p><strong>正式主测96条：发色16、发型16、背景16、TripVVT24、通用文字编辑24。</strong> 本包只包含冻结输入及审核预览，三模型生成结果另行交付。<a href="identity_diagnostic/index.html">四条Identity独立诊断的输入与完整头部参考</a> · <a href="README.txt">说明</a> · <a href="review_inventory.jsonl">可移植输入目录</a></p>'
   content=content.replace('<h1>',notice+'<h1>',1)
  parser=Links();parser.feed(content)
  expected=rows(root/'metadata/inputs.jsonl');assert {r['id'] for r in expected}<=parser.ids
  for relative in parser.links:
   if not prefix and relative in ['identity_diagnostic/index.html','README.txt','review_inventory.jsonl']:continue
   source=directory/relative;assert source.is_file(),source
   files[Path(prefix)/relative]=source
  pages[Path(prefix)/'index.html']=content
  return parser.videos
 main_videos=page(ROOT,'');diag_videos=page(DIAG,'identity_diagnostic')
 assert main_videos==122 and diag_videos==4,(main_videos,diag_videos)
 inventory=[]
 for cohort,root,prefix in [('main',ROOT,''),('identity_diagnostic',DIAG,'identity_diagnostic')]:
  for item in rows(root/'metadata/inputs.jsonl'):
   sid=item['id'];inventory.append(dict(cohort=cohort,id=sid,group=item['group'],subgroup=item.get('subgroup'),axis=item.get('axis'),mode=item.get('mode'),prompt=item['prompt'],source_video=str(Path(prefix)/'media'/f'{sid}_source.mp4'),references=[dict(path=str(Path(prefix)/'media'/f"{sid}_ref{k}{Path(r['absolute_path']).suffix}"),type=r['type'],sha256=r['sha256']) for k,r in enumerate(item['references'])],mask_preview=str(Path(prefix)/f'{sid}_mask.jpg'),old_id=item.get('old_id'),uid=item['uid'],task_origin=item.get('task_origin'),revision_reason=item.get('revision_reason'),canvas_wh=item['canvas_wh'],fps=item['fps'],num_frames=item['num_frames']))
 pages[Path('review_inventory.jsonl')]=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in inventory)
 pages[Path('README.txt')]='''A849 v3 冻结测试集输入审核包\n\n解压后打开 index.html；保持同目录的 media 和 identity_diagnostic 文件夹。\n正式主测96条：Hair color16、Hair16、Background16、TripVVT24、通用文字编辑24。另附4条Identity诊断输入，未计入96条。\n每条含source、完整编辑指令、参考图（如有）、0/1/2/3/4秒mask叠加预览；红色表示VACE可编辑范围。TripVVT含24条审核用GT及官方caption，GT不进入模型。VACE helper仅提供整个包前两条主测样例。\n取样时刻0–4秒，共49帧、12fps，编码长度49/12秒。照片不裁头，Identity自然头发保留，人物保留前景像素不重绘，背景RGB128。分割局限见 metadata/identity_input_review.json。\nIdentity完整头部参考的12条三模型复测已直接审核，02/03/04三模型均无清晰面部身份迁移，按预设规则移出主测。旧裁顶参考诊断保留在集群，未用于正式去留。\n本包供查看选样和输入，不含尚在生成/评分的三模型输出，也不是包含全部PNG帧/latent的推理工作目录。可移植目录见review_inventory.jsonl；原始冻结清单及来源、旧ID、修改原因位于metadata。\n没有评分控件；自建配饰任务无GT并明确标记。未调用Gemini或其他VLM API。\n'''
 for src_name,dst_name in [('inputs.jsonl','inputs.jsonl'),('freeze.json','freeze.json'),('revision_log.json','revision_log.json'),('source_clusters.json','source_clusters.json'),('input_visual_review.json','input_visual_review.json'),('task_limitations.json','task_limitations.json'),('hair_color_templates.json','hair_color_templates.json'),('reference_processing.json','reference_processing.json'),('identity_protocol_correction.json','identity_protocol_correction.json')]:files[Path('metadata')/dst_name]=ROOT/'metadata'/src_name
 files[Path('metadata/identity_inputs.jsonl')]=DIAG/'metadata/inputs.jsonl'
 files[Path('metadata/identity_review.json')]=DIAG/'metadata/identity_review.json'
 files[Path('metadata/identity_input_review.json')]=DIAG/'metadata/input_review.json'
 with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
  for path,source in sorted(files.items()):assert source.is_file();archive.write(source,Path(name)/path)
  for path,content in pages.items():archive.writestr(str(Path(name)/path),content)
 with zipfile.ZipFile(output) as archive:
  assert archive.testzip() is None
  members=set(archive.namelist())
  for path,content in pages.items():
   if path.suffix!='.html':continue
   parser=Links();parser.feed(content)
   for relative in parser.links:assert str(Path(name)/path.parent/relative) in members,(path,relative)
  assert not any('/outputs/' in path for path in members)
 atomic(ROOT/'metadata/input_package.json',dict(path=str(output),bytes=output.stat().st_size,main_tasks=96,identity_diagnostic_inputs=4,main_video_elements=main_videos,diagnostic_video_elements=diag_videos,model_outputs_included=0,offline_page=f'{name}/index.html',source_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),local_link_and_crc_validation='passed'))
 print('Input ZIP ready',output,output.stat().st_size,flush=True)
if __name__=='__main__':main()
