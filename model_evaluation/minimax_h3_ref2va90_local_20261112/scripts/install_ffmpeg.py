from common import *
import urllib.request,tarfile,shutil,subprocess
require_job()
receipt=ROOT/'metadata/ffmpeg_release.json'
if receipt.exists():asset=read(receipt)['asset']
else:
 with urllib.request.urlopen('https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest',timeout=60) as response:release=json.load(response)
 asset=next(a for a in release['assets'] if a['name']=='ffmpeg-master-latest-linux64-gpl.tar.xz')
 atomic(receipt,{'release_id':release['id'],'tag':release['tag_name'],'asset':asset})
archive=ROOT/'runtime'/asset['name']
if not archive.exists():
 with urllib.request.urlopen(asset['browser_download_url'],timeout=120) as response,archive.with_suffix('.part').open('wb') as target:shutil.copyfileobj(response,target)
 archive.with_suffix('.part').replace(archive)
(rootbin:=ROOT/'runtime/bin').mkdir(exist_ok=True)
with tarfile.open(archive) as t:
 for binary in ('ffmpeg','ffprobe'):
  member=next(m for m in t.getmembers() if m.isfile() and m.name.endswith('/bin/'+binary))
  with t.extractfile(member) as source,(rootbin/binary).open('wb') as target:shutil.copyfileobj(source,target)
  (rootbin/binary).chmod(0o755)
  print(subprocess.check_output([str(rootbin/binary),'-version'],text=True).splitlines()[0],flush=True)
