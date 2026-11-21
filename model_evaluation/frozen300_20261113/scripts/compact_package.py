"""Ship a small, self-contained review only; the full frozen300 stays local."""
from build import *
import re,subprocess,zipfile
from html.parser import HTMLParser

class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=[];self.articles=[];self.videos=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.append(a['id'])
  if tag=='article':self.articles.append(a['id'])
  if tag=='video':self.videos.append(a['src'])
  for k in ['src','poster','href']:
   if k in a:self.links.append(a[k])

def main():
 p=ROOT/'package';out=ROOT/'review_subset';out.mkdir(exist_ok=True)
 selection=read(ROOT/'metadata/compact_selection_proposed.json');review=read(p/'review_manifest.json');assert review['samples']==30 and len(review['clips'])==63
 page=(p/'index.html').read_text()
 page=re.sub(r'<a href="media/[^\"]+\.mp4">完整原件</a>','原件保留本地',page)
 page=page.replace('<title>300对冻结测评集 · 审核</title>','<title>30对精简审核 · 冻结300对</title>')
 page=page.replace('完整原件保留。','完整原视频保留本地；本精简包只携带浏览副本和所需图片参考。').replace('每个区域保留完整原件入口。','')
 (out/'index.html').write_text(page)
 parser=Links();parser.feed(page);assert len(parser.articles)==30 and set(parser.articles)==set(selection['ids']) and len(parser.videos)==63
 assert len(parser.ids)==len(set(parser.ids))
 files=set()
 for path in parser.links:
  if path.startswith('#'):assert path[1:] in parser.ids;continue
  rel=Path(path);assert not rel.is_absolute() and '..' not in rel.parts
  if path.startswith(('media/','review_media/')):
   assert (p/rel).is_file();dest=out/rel;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p/rel,dest);files.add(path)
 for x in review['clips']:
  assert sha(out/x['path'])==x['sha256'] and x['full_decode_frames']>=49
 clean=[]
 for x in review['clips']:
  r={k:v for k,v in x.items() if k!='original_path'};r['original_video_included']=False;clean.append(r)
 write(out/'review_manifest.json',{**review,'clips':clean})
 write(out/'selection.json',{**selection,'scope':'final compact review only; full frozen300 retained locally','no_output_based_selection':True})
 shutil.copyfile(p/'sampling.json',out/'sampling.json')
 shutil.copyfile(p/'model_interfaces.json',out/'model_interfaces.json')
 (out/'MODEL_HANDOFF.md').write_text('本包是30对审核子集，下面描述的完整300对数据包保留在本机，不在此次上传中。\n\n'+(ROOT/'HANDOFF.md').read_text())
 (out/'README.md').write_text('''# 30对精简HTML审核包

解压后打开 index.html，离线审阅完整编辑prompt、SOURCE、有序参考、GT和官方caption。

合成8个人物ID（12、13、14、15、17、18、19、20）各3对，共24对，覆盖全部8类编辑任务；另取CoinVE3对和TripVVT3对，共30对。真实集没有官方人物ID，真实样例不宣称逐人物覆盖。selection.json记录稳定样本ID与固定抽样seed，不根据模型输出挑样。

只携带63段共同区间H.264浏览视频（30 SOURCE＋30 GT＋3动作参考）、首帧海报和有序图片参考。原视频及完整冻结300对留在本机；本包用于审核，不是可独立执行全量测评的输入包。sampling.json是完整300对的原冻结统计。GT和caption仅供审核，不进入推理；本包无评分控件，不代表人工质量通过。

页面支持任务导航、ID/英文指令筛选、同步播放、共同时间滑条。模型说明见MODEL_HANDOFF.md及model_interfaces.json，实际300对输入不会随这个精简审核包上传。本机A849继续使用完整300对清单。
''')
 # Text and relationships are checked against the frozen source, never a model output.
 import html
 items=[a for a in rows(p/'inputs.jsonl') if a['id'] in selection['ids']];gts={a['id']:a for a in rows(p/'ground_truth.jsonl')}
 identities=Counter(a['source']['attributes']['identity_id'] for a in items if a['dataset']=='ue');assert len(identities)==8 and set(identities.values())=={3}
 assert len({a['group'] for a in items if a['dataset']=='ue'})==8
 for a in items:
  assert html.escape(a['prompt']['en'],quote=True) in page
  if gts[a['id']].get('caption'):assert html.escape(gts[a['id']]['caption'],quote=True) in page
  for i,m in enumerate(a['references']):
   if m['type']!='motion':assert m['path'] in files
 # Use actual page JavaScript; the harness checks relative counts for this subset.
 test=(ROOT/'scripts/test_review_controls.js').read_text().replace(' / 300',' / 30');(ROOT/'metadata/compact_controls_test.js').write_text(test)
 subprocess.run(['node',str(ROOT/'metadata/compact_controls_test.js'),str(out/'index.html')],check=True)
 validation={'status':'passed','samples':30,'synthetic_identities':8,'pairs_per_synthetic_identity':3,'counts':{'ue':24,'coinve':3,'tripvvt':3},'browser_videos':63,'all_clips_full_decoded':True,'browser_video_hashes_verified':True,'complete_prompts_and_captions':True,'ordered_image_reference_links':True,'javascript_controls':'passed','full_frozen300_uploaded':False,'human_quality_review':'not_performed'}
 write(out/'validation.json',validation)
 for link in parser.links:
  if not link.startswith('#'):assert (out/link).is_file(),link
 dst=ROOT/'frozen300_people30_review_20261113.zip';temp=dst.with_suffix('.part.zip')
 with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
  for f in sorted(out.rglob('*')):
   if f.is_file():z.write(f,'frozen300_people30_review_20261113/'+f.relative_to(out).as_posix())
 with zipfile.ZipFile(temp) as z:assert z.testzip() is None
 temp.replace(dst);write(ROOT/'metadata/compact_package.json',{'status':'complete','zip':str(dst),'bytes':dst.stat().st_size,'validation':validation})
 print('COMPACT_REVIEW_READY',dst,dst.stat().st_size,flush=True)
if __name__=='__main__':main()
