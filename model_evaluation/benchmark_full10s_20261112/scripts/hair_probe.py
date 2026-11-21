import sys
sys.path.insert(0,'/scratch/prj0000000275/jcy/ljy/video_collection/model_evaluation/benchmark_full10s_20261112/scripts')
from common import *
from media import video_frames
from PIL import Image,ImageDraw
require_job()
items=[r for r in rows(ROOT/'metadata/inputs.jsonl') if r['id'] in ['hair_05','hair_07']]
canvas=Image.new('RGB',(6*256,2*448),'white');draw=ImageDraw.Draw(canvas)
for row,item in enumerate(items):
 frames,_=video_frames(item,item['source'])
 for col,i in enumerate([0,24,48,60,90,120]):
  im=frames[i].copy();im.thumbnail((256,420));canvas.paste(im,(col*256,row*448+25));draw.text((col*256,row*448),item['id']+' t='+str(i/12),fill='black')
canvas.save(ROOT/'metadata/hair_source_probe.jpg')
