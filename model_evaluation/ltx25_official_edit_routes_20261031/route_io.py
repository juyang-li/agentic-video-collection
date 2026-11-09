"""Source and manually tracked mask I/O; model operations use official nodes."""
from pathlib import Path
import os
import av,numpy as np,torch
import torch.nn.functional as F

ROOT=Path(os.environ['LTX_BENCH_ROOT'])
class RouteInputs:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'unused':('INT',{'default':0})}}
    RETURN_TYPES=('IMAGE','MASK','IMAGE','MASK','INT');FUNCTION='load';CATEGORY='LTX routes I/O'
    def load(self,unused=0):
        with av.open(str(ROOT/'assets/appearance_002_source49.mp4')) as c:
            images=torch.from_numpy(np.stack([f.to_ndarray(format='rgb24') for f in c.decode(video=0)])).float()/255
        mask=torch.from_numpy(np.load(ROOT/'assets/eye_mask.npy'))
        high=F.interpolate(images.permute(0,3,1,2),scale_factor=2,mode='bicubic',align_corners=False).permute(0,2,3,1).clamp(0,1)
        highmask=F.interpolate(mask[:,None],scale_factor=2,mode='nearest')[:,0]
        assert len(images)==len(mask)==49
        return images,mask,high,highmask,len(images)
NODE_CLASS_MAPPINGS={'LTXRouteInputs':RouteInputs}
