"""CPU ZIP of the116 frozen inputs only; relative links and local CRC validation."""
import zipfile,shutil,html.parser,time
from common import *
from review import check_links

def main():
 src=ROOT/'review/all';dest=ROOT/'review/inputs116';dest.mkdir(parents=True,exist_ok=True);page=(src/'input_review.html').read_text();assets=set()
 class Assets(html.parser.HTMLParser):
  def handle_starttag(self,tag,attrs):
   for key,value in attrs:
    if key in['src','href']and not value.startswith('#'):
     assert not value.startswith(('http:','https:','/'));assets.add(value)
 Assets().feed(page)
 for name in assets:
  p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists():shutil.copy2(src/name,p)
 (dest/'index.html').write_text(page)
 for name in['inputs.jsonl','freeze.json','input_origins.json','input_review.json','generation_gate.json','h3_preflight.json','h3_runtime_contract.json']:
  shutil.copy2(ROOT/'metadata'/name,dest/name)
 check_links(dest);output=ROOT/'review/a849_v3_inputs116_20261114.zip';tmp=output.with_suffix('.part.zip')
 with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1)as z:
  for p in sorted(dest.rglob('*')):
   if p.is_file():z.write(p,Path('a849_v3_inputs116')/p.relative_to(dest))
 with zipfile.ZipFile(tmp)as z:assert z.testzip()is None
 tmp.replace(output);atomic(ROOT/'metadata/inputs_package.json',dict(status='complete',tasks=116,path=str(output),bytes=output.stat().st_size,zip_crc='passed',offline_page='a849_v3_inputs116/index.html',no_model_outputs=True,created_unix=time.time()));print('Input ZIP116',output.stat().st_size,flush=True)
if __name__=='__main__':main()
