import torch
class SilentAudio:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'frames':('INT',),'fps':('FLOAT',)}}
    RETURN_TYPES=('AUDIO',);FUNCTION='load';CATEGORY='LTX routes I/O'
    def load(self,frames,fps):return ({'waveform':torch.zeros(1,2,round(frames/fps*48000)), 'sample_rate':48000},)
NODE_CLASS_MAPPINGS={'LTXRouteSilentAudio':SilentAudio}
