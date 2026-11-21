"""Exact, source-only nearest-PTS sampling and aspect-preserving letterbox."""
import sys
from common import *
sys.path.insert(0,str(NATIVE/'scripts/common'))
from project_layout import add_script_paths
add_script_paths()
from coinve_cache_contract import load_video_frames
from native_image_cache_contract import prepare_image

def video_frames(item,record):
    profile=read(NATIVE/'configs/data/mixed720p/tripvvt.json')
    profile.update(duration_seconds=4,num_frames=49)
    window=dict(fps=12,num_frames=49,timestamps_seconds=item['timestamps_seconds'],start_seconds=item['timestamps_seconds'][0],span_seconds=4)
    return load_video_frames(record['absolute_path'],item['canvas_wh'],profile,window)

def write_preview(frames,path,fps=12):
    import av
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with av.open(str(path),'w') as c:
        s=c.add_stream('libx264',rate=fps);s.width,s.height=frames[0].size;s.pix_fmt='yuv420p';s.thread_count=2
        s.options={'crf':'20','preset':'fast'}
        for f in frames:
            for packet in s.encode(av.VideoFrame.from_image(f)):c.mux(packet)
        for packet in s.encode():c.mux(packet)

if __name__=='__main__':
    require_job()
    from PIL import Image,ImageOps,ImageDraw
    inputs=rows(ROOT/'metadata/inputs.jsonl');thumbs=[]
    for item in inputs:
        frames,audit=video_frames(item,item['source'])
        out=ROOT/'inputs/prepared'/item['id'];out.mkdir(parents=True,exist_ok=True)
        for i,f in enumerate(frames):f.save(out/f'{i:03d}.png')
        write_preview(frames,out/'source.mp4')
        atomic(out/'pixels.json',dict(source_sha256=sha(item['source']['absolute_path']),input_manifest_sha256=sha(ROOT/'metadata/inputs.jsonl'),pixel_audit=audit,frame_sha256=[sha(out/f'{i:03d}.png') for i in range(49)]))
        if item['group']=='tripvvt':
            thumbs.append((item['id']+' source',frames[0]))
            for r in item['references']:thumbs.append((item['id']+' garment',Image.open(r['absolute_path']).convert('RGB')))
    canvas=Image.new('RGB',(1200,((len(thumbs)+3)//4)*350),'#20242a');d=ImageDraw.Draw(canvas)
    for i,(label,img) in enumerate(thumbs):
        x=i%4*300;y=i//4*350;canvas.paste(ImageOps.contain(img,(295,320)),(x,y+25));d.text((x,y+4),label,fill='white')
    canvas.save(ROOT/'review/tripvvt_references.jpg',quality=92)
