"""Review-only checks: all samples, exact text, local links, clip identities, JS behavior."""
from build import *
from html.parser import HTMLParser
import html,re,subprocess

class Page(HTMLParser):
 def __init__(self):super().__init__();self.ids=[];self.links=[];self.articles=[];self.videos=[];self.scoring=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.append(a['id'])
  if tag=='article':self.articles.append(a['id'])
  if tag=='video':self.videos.append(a['src'])
  for attr in ['href','src','poster']:
   if attr in a:self.links.append(a[attr])
  if tag in ['input','select'] and a.get('type') not in ['search','range']:self.scoring.append(a)

def main():
 p=ROOT/'package';page=(p/'index.html').read_text();parsed=Page();parsed.feed(page)
 review=read(p/'review_manifest.json');selected={x['id'] for x in review['clips']}
 items=[a for a in rows(p/'inputs.jsonl') if a['id'] in selected];gt={x['id']:x for x in rows(p/'ground_truth.jsonl')};review=read(p/'review_manifest.json')
 assert set(parsed.articles)=={x['id'] for x in items} and len(parsed.articles)==review['samples']
 assert len(parsed.ids)==len(set(parsed.ids)) and not parsed.scoring
 assert review['input_manifest_sha256']==sha(p/'inputs.jsonl') and review['full_decode_passed']
 assert len(parsed.videos)==len(review['clips'])==sum(2+sum(m['type']=='motion' for m in a['references']) for a in items)
 for link in parsed.links:
  if link.startswith('#'):assert link[1:] in parsed.ids
  else:
   rel=Path(link);assert not rel.is_absolute() and '..' not in rel.parts and (p/rel).is_file(),link
 for a in items:
  assert html.escape(a['prompt']['en'],quote=True) in page
  if gt[a['id']].get('caption'):assert html.escape(gt[a['id']]['caption'],quote=True) in page
 for r in review['clips']:
  assert sha(p/r['path'])==r['sha256'] and r['full_decode_frames']==next(a['num_frames'] for a in items if a['id']==r['id'])
 js=re.search(r'<script>([\s\S]*?)</script>',page).group(1);jsfile=ROOT/'metadata/review.js';jsfile.write_text(js)
 subprocess.run(['node','--check',str(jsfile)],check=True)
 test=(ROOT/'scripts/test_review_controls.js').read_text().replace(' / 300',f' / {len(items)}')
 testfile=ROOT/'metadata/review_controls_current.js';testfile.write_text(test)
 subprocess.run(['node',str(testfile),str(p/'index.html')],check=True)
 result={'status':'passed','sample_sections':len(items),'browser_videos':len(review['clips']),'relative_media_and_navigation_links_checked':len(parsed.links),'complete_prompts_and_captions':True,'browser_clips_sha256_checked':True,'browser_clips_full_decoded':True,'javascript_syntax':'passed','javascript_controls':'play/pause, shared seek, drift correction, endpoint clamp, reset and filter passed','checked_at_utc':datetime.now(timezone.utc).isoformat(),'scoring_ui':False,'human_quality_review':'not_performed'}
 write(ROOT/'metadata/review_validation.json',result)
 v=read(p/'validation.json');v['html_review']=result;write(p/'validation.json',v)
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
