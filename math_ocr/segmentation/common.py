"""Image-only symbol operations. No stroke order or truth at inference."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import numpy as np
from PIL import Image,ImageOps

RELATIONS=['None','Right','Sup','Sub','Above','Below','Inside']

def crop_symbol(image,box=None,size=64):
    if box is not None:
        x0,y0,x1,y1=box
        image=image.crop((max(0,int(x0)-1),max(0,int(y0)-1),min(image.width,int(np.ceil(x1))+1),min(image.height,int(np.ceil(y1))+1)))
    image=image.convert('L');bounds=ImageOps.invert(image).point(lambda p:255 if p>35 else 0).getbbox()
    if bounds:image=image.crop(bounds)
    image.thumbnail((size-10,size-10),Image.Resampling.LANCZOS)
    canvas=Image.new('L',(size,size),255);canvas.paste(image,((size-image.width)//2,(size-image.height)//2))
    return np.array(canvas,dtype=np.uint8)

def geometry(a,b):
    a=np.asarray(a,dtype=np.float32);b=np.asarray(b,dtype=np.float32)
    aw,ah=np.maximum(a[2:]-a[:2],1);bw,bh=np.maximum(b[2:]-b[:2],1)
    ac=(a[:2]+a[2:])/2;bc=(b[:2]+b[2:])/2;s=max(ah,bh,8)
    ix=max(0,min(a[2],b[2])-max(a[0],b[0]));iy=max(0,min(a[3],b[3])-max(a[1],b[1]))
    return np.clip(np.array([(bc[0]-ac[0])/s,(bc[1]-ac[1])/s,(b[0]-a[2])/s,(b[1]-a[3])/s,
        np.log(bw/aw),np.log(bh/ah),aw/ah,bw/bh,ix/min(aw,bw),iy/min(ah,bh),
        (b[1]-a[1])/s,(b[3]-a[3])/s,aw/s,bw/s,ah/s,bh/s],dtype=np.float32),-10,10)

def iou(a,b):
    a=np.asarray(a);b=np.asarray(b);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(inter/max(1,np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter))
