from common import *
class HumanParser:
    def __init__(self,record,device='cuda'):
        from transformers import SegformerImageProcessor,SegformerForSemanticSegmentation
        self.processor=SegformerImageProcessor.from_pretrained(record['path'],local_files_only=True)
        self.device=device
        self.model=SegformerForSemanticSegmentation.from_pretrained(record['path'],local_files_only=True).to(device).eval()
        self.record=record
    def segment(self,image,labels,main_person=False):
        import torch
        import numpy as np
        with torch.inference_mode():
            inputs=self.processor(images=image,return_tensors='pt').to(self.device)
            logits=self.model(**inputs).logits
            logits=torch.nn.functional.interpolate(logits,size=image.size[::-1],mode='bilinear',align_corners=False)
            values=logits.argmax(dim=1)[0].cpu().numpy()
        mask=np.isin(values,labels)
        if main_person:
            import cv2
            count,components,stats,_=cv2.connectedComponentsWithStats((values!=0).astype(np.uint8),connectivity=8)
            if count<2:raise RuntimeError('No foreground person parsed')
            largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
            mask &= components==largest
        return mask.astype(np.uint8)
