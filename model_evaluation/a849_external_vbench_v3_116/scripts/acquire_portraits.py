"""CPU: retrieve the first800 members of the pinned CelebA-HQ1024 ZIP."""
import json, struct, zlib, hashlib
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
s=requests.Session();s.trust_env=False
repo="Iceclear/CelebA-HQ1024"
info=json.loads((ROOT/"metadata/celeba_revision.json").read_text())
pin=ROOT/"metadata/celeba_revision.json"
if pin.exists():info=json.loads(pin.read_text())
else:pin.write_text(json.dumps(info,indent=2))
url=f"https://huggingface.co/datasets/{repo}/resolve/{info['sha']}/CelebA-HQ-1024.zip"
r=s.get(url,stream=True,timeout=120);r.raise_for_status();stream=r.raw
records=[]
def read(n):
 b=stream.read(n)
 if len(b)!=n:raise RuntimeError("incomplete ZIP stream")
 return b
while len(records)<800:
 head=read(30)
 sig,ver,flag,method,mt,md,crc,cs,us,nl,xl=struct.unpack("<IHHHHHIIIHH",head)
 assert sig==0x04034b50 and not flag&8,(hex(sig),flag)
 name=read(nl).decode();extra=read(xl)
 data=read(cs)
 if not name.lower().endswith((".jpg",".png")):continue
 raw=zlib.decompress(data,-15) if method==8 else data
 assert len(raw)==us and zlib.crc32(raw)&0xffffffff==crc
 out=ROOT/"references"/Path(name).name;out.write_bytes(raw)
 records.append(dict(member=name,path=str(out),sha256=hashlib.sha256(raw).hexdigest(),url=url))
 print(name,len(raw),flush=True)
r.close()
(ROOT/"metadata/portrait_candidates.json").write_text(json.dumps(records,indent=2))
from PIL import Image,ImageOps,ImageDraw
im=Image.new("RGB",(1440,((len(records)+5)//6)*260),"#20242a");d=ImageDraw.Draw(im)
for i,r in enumerate(records):
 x=i%6*240;y=i//6*260
 pic=Image.open(r["path"]);assert pic.size==(1024,1024),pic.size
 im.paste(ImageOps.contain(pic,(236,236)),(x,y+22));d.text((x+3,y+3),Path(r["path"]).name,fill="white")
im.save(ROOT/"review/portrait_candidates.jpg",quality=95)
