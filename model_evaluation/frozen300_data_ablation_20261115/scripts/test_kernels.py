"""Small CPU invariants; the 12-video smoke additionally validates actual GPUs."""
import numpy as np
from gt_kernels import geometry,frame_metrics
assert geometry(1920,1080,1280,736)==[0,8,1280,720]
assert geometry(1080,1920,736,1280)==[8,0,720,1280]
rng=np.random.default_rng(42);a=rng.integers(0,256,(32,48,3),dtype=np.uint8)
s,p,inf=frame_metrics((a,a));assert abs(s-1)<1e-12 and p is None and inf
b=np.full_like(a,128);z=np.zeros_like(a)
s,p,inf=frame_metrics((z,b));assert not inf
expected=-20*np.log10(float(np.float32(128)/255))
assert abs(p-expected)<1e-6 and 0<s<.01
print('CPU_KERNEL_INVARIANTS_PASSED')
