"""Rasterize MathWriting InkML and normalize paper illumination for offline OCR."""
import math
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image,ImageDraw,ImageFilter,ImageOps
from prepare_data import preprocess


def annotations(raw):
    root=ET.fromstring(raw)
    return root,{a.attrib.get('type'):a.text or '' for a in root.findall('{*}annotation')}


def rasterize(root,width=1.5,height=64,max_width=512):
    strokes=[]
    for trace in root.findall('{*}trace'):
        values=np.fromstring((trace.text or '').replace(',',' '),sep=' ')
        if len(values)%3:raise ValueError('Unexpected InkML channels')
        points=values.reshape(-1,3)[:,:2]
        if len(points):strokes.append(points)
    if not strokes:raise ValueError('Empty ink')
    points=np.concatenate(strokes);low=points.min(0);extent=np.maximum(points.max(0)-low,1.)
    if not np.isfinite(points).all():raise ValueError('Non-finite coordinates')
    scale=min((height-10)/extent[1],(max_width-10)/extent[0])
    w=max(32,int(math.ceil((extent[0]*scale+10)/4))*4);factor=3
    canvas=Image.new('L',(w*factor,height*factor),255);draw=ImageDraw.Draw(canvas)
    radius=width*factor/2
    for stroke in strokes:
        coordinates=(stroke-low)*scale+[5,(height-extent[1]*scale)/2]
        xy=[(float(x*factor),float(y*factor)) for x,y in coordinates]
        if len(xy)>1:draw.line(xy,fill=0,width=max(1,round(width*factor)),joint='curve')
        for x,y in [xy[0],xy[-1]]:draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=0)
    return canvas.resize((w,height),Image.Resampling.LANCZOS)


def preprocess_handwriting(source,height=64,max_width=512):
    """Continuous local background normalization; keep faint decimal points."""
    source=ImageOps.exif_transpose(source).convert('RGBA')
    white=Image.new('RGBA',source.size,'white');white.alpha_composite(source)
    gray=white.convert('L')
    # First reduce exceptionally large camera crops, before local filtering.
    gray.thumbnail((2400,2400),Image.Resampling.LANCZOS)
    background=gray.filter(ImageFilter.GaussianBlur(max(8,min(gray.size)*.065)))
    values=np.asarray(gray,dtype=np.float32);bg=np.asarray(background,dtype=np.float32)
    corrected=np.clip(values*255/np.maximum(bg,1),0,255).astype(np.uint8)
    return preprocess(Image.fromarray(corrected),height=height,max_width=max_width)


def capture(image,seed):
    rng=np.random.default_rng(seed)
    image=image.resize((image.width*3,image.height*3),Image.Resampling.BICUBIC)
    ink=np.asarray(image,dtype=np.float32)/255
    gradient=np.linspace(185,250,image.width)[None,:]+np.linspace(-6,6,image.height)[:,None]
    pixels=np.clip(ink*gradient+rng.normal(0,1.8,ink.shape),0,255).astype(np.uint8)
    source=Image.fromarray(pixels).rotate(.6 if seed%2 else -.6,fillcolor=225,resample=Image.Resampling.BICUBIC)
    return preprocess_handwriting(source)
