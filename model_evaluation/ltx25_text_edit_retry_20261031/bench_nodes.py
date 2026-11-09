"""Benchmark I/O only. Conditioning and sampling remain official LTX nodes."""
from pathlib import Path
import os,math
import av,numpy as np,torch
from PIL import Image
import imageio_ffmpeg

ROOT=Path(os.environ['LTX_BENCH_ROOT']).resolve()

def fit(image,width,height):
    image=image.convert('RGB');s=min(width/image.width,height/image.height)
    image=image.resize((max(1,round(image.width*s)),max(1,round(image.height*s))),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(width,height));canvas.paste(image,((width-image.width)//2,(height-image.height)//2))
    return np.asarray(canvas,dtype=np.float32)/255

class BenchSource:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'path':('STRING',),'max_frames':('INT',{'default':97}),'fps':('FLOAT',{'default':24.0})}}
    RETURN_TYPES=('IMAGE','INT','INT','INT')
    RETURN_NAMES=('images','width','height','frames')
    FUNCTION='load';CATEGORY='LTX benchmark'
    def load(self,path,max_frames,fps):
        with av.open(path) as container:
            stream=container.streams.video[0];rate=float(stream.average_rate or fps)
            duration=float(stream.duration*stream.time_base) if stream.duration else (container.duration or 0)/1e6
            available=min(max_frames,max(1,int(math.floor(duration*fps+1e-5)))) if duration else max_frames
            n=1+8*((available-1)//8)
            width,height=(768,448) if stream.width>=stream.height else (448,768)
            wanted=[int(math.floor(k*rate/fps+1e-6)) for k in range(n)]
            result=[];j=0
            for i,frame in enumerate(container.decode(stream)):
                if j>=n:break
                if i<wanted[j]:continue
                arr=fit(frame.to_image(),width,height)
                while j<n and wanted[j]<=i:result.append(arr);j+=1
            if not result:raise ValueError('source video has no decodable frames')
            n=1+8*((len(result)-1)//8)
            images=torch.from_numpy(np.stack(result[:n]))
        return images,width,height,n

class BenchRef:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'path':('STRING',),'width':('INT',),'height':('INT',),'repeat':('INT',{'default':1})}}
    RETURN_TYPES=('IMAGE',);FUNCTION='load';CATEGORY='LTX benchmark'
    def load(self,path,width,height,repeat):
        with Image.open(path) as im:array=fit(im,width,height)
        return (torch.from_numpy(array).unsqueeze(0).repeat(repeat,1,1,1),)

class BenchSave:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'images':('IMAGE',),'path':('STRING',),'fps':('FLOAT',{'default':24.0})}}
    RETURN_TYPES=();FUNCTION='save';OUTPUT_NODE=True;CATEGORY='LTX benchmark'
    def save(self,images,path,fps):
        out=Path(path).resolve()
        if not out.is_relative_to(ROOT/'outputs'):raise ValueError('output must be inside benchmark outputs/')
        out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.mp4')
        h,w=images.shape[1:3]
        writer=imageio_ffmpeg.write_frames(str(tmp),(w,h),fps=fps,codec='libx264',pix_fmt_out='yuv420p',
            macro_block_size=1,output_params=['-crf','18','-threads','2','-movflags','+faststart'])
        writer.send(None)
        try:
            for frame in images:
                array=(frame.detach().float().cpu().numpy().clip(0,1)*255).round().astype(np.uint8)
                writer.send(np.ascontiguousarray(array))
        except Exception:
            # Preserve expensive decoded frames if only the final encoder fails.
            torch.save(images.detach().cpu(),out.with_suffix('.decoded.pt'))
            raise
        finally:writer.close()
        tmp.replace(out)
        out.with_suffix('.decoded.pt').unlink(missing_ok=True)
        return {'ui':{'bench_videos':[{'path':str(out),'frames':len(images),'width':w,'height':h,'fps':fps}]}}

NODE_CLASS_MAPPINGS={'LTXBenchSource':BenchSource,'LTXBenchRef':BenchRef,'LTXBenchSave':BenchSave}
