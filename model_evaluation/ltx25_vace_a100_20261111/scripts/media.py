"""Shared exact pixel preparation; inference reads originals, never browser MP4s."""
import sys
from copy import deepcopy
from common import *
sys.path.insert(0,str(NATIVE/'scripts/common'))
from project_layout import add_script_paths
add_script_paths()
from prepare_datamake_media import load_video_frames as load_ue
from coinve_cache_contract import load_video_frames as load_real
from native_image_cache_contract import prepare_image

def profile(dataset):
    s=next(s for s in read(BUNDLE)['sources'] if s['name']==dataset)
    p=deepcopy(read(s['media_profile']))
    p.update(duration_seconds=4,num_frames=49)
    return p

def video_frames(item, record):
    p=profile(item['dataset'])
    if item['dataset']=='ue':
        return load_ue(record['absolute_path'],item['canvas_wh'],p)
    window={'fps':12,'num_frames':49,'timestamps_seconds':[i/12 for i in range(49)],'start_seconds':0,'span_seconds':4}
    return load_real(record['absolute_path'],item['canvas_wh'],p,window)


def write_preview(frames, path, fps):
    """H264 browser copy; cap encoder threads on the shared login node."""
    import av
    with av.open(str(path),'w') as c:
        stream=c.add_stream('libx264',rate=fps)
        stream.width,stream.height=frames[0].size
        stream.pix_fmt='yuv420p'
        stream.thread_count=2
        stream.options={'crf':'20','preset':'fast'}
        for image in frames:
            for packet in stream.encode(av.VideoFrame.from_image(image)):c.mux(packet)
        for packet in stream.encode():c.mux(packet)
