"""Font-bank and mild capture augmentation, independent of model/loss choices."""
import random
import numpy as np
from PIL import Image,ImageFilter
import torch
import train


class PracticalDataset(train.FormulaDataset):
    def __init__(self,records,vocab,augment=True,preserve_bounds=False,max_width=384):
        super().__init__(records,vocab,augment)
        self.preserve_bounds=preserve_bounds;self.max_width=max_width
        banks=[[base]+[np.array(Image.open(train.DATA/path),dtype=np.uint8) for path in row.get("image_variants",[])]
               for base,row in zip(self.images,records)]
        class Bank:
            def __getitem__(self,index):return random.choice(banks[index])
        self.images=Bank()

    def __getitem__(self,idx):
        image,tokens,index=super().__getitem__(idx)
        if random.random()<.7:
            arr=np.clip((1-image[0].numpy())*255,0,255).astype(np.uint8)
            transformed=Image.fromarray(arr).rotate(random.uniform(-1,1),resample=Image.Resampling.BICUBIC,fillcolor=255,expand=self.preserve_bounds)
            if self.preserve_bounds:
                from prepare_data import preprocess
                transformed=preprocess(transformed,max_width=self.max_width)
            transformed=transformed.filter(ImageFilter.GaussianBlur(random.uniform(0,.45)))
            ink=(255-np.array(transformed).astype(np.float32))/255
            ink=np.clip(ink+np.random.normal(0,random.uniform(0,.015),ink.shape),0,1).astype(np.float32)
            image=torch.from_numpy(ink).unsqueeze(0)
        return image,tokens,index
