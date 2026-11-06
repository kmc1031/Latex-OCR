"""AI-assisted single-study image-only CNN/BiLSTM/LSTM implementation.

No checkpoint/data annotation is used by inference other than the class vocab.
Hard segmentation is outside autograd; CNN probabilities remain in the graph.
"""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image,ImageOps,ImageFilter
from scipy import ndimage
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
CLASSES=CFG['class_labels'];TOKENS=CFG['output_tokens']
sys.path.insert(0,str(ROOT.parent/'planned_experiment'))
from run import components,crop

def preprocess(source):
    source=ImageOps.exif_transpose(source).convert('RGBA');out=Image.new('RGBA',source.size,'white');out.alpha_composite(source);im=out.convert('L');im.thumbnail((1600,1600))
    a=np.asarray(im,dtype=np.float32);bg=np.asarray(im.filter(ImageFilter.GaussianBlur(max(8,.05*min(im.size)))),dtype=np.float32);a=np.clip(255*a/(bg+1),0,255).astype('uint8')
    def otsu(a):
        hist=np.bincount(a.ravel(),minlength=256).astype(np.float64);w=hist.cumsum();s=(hist*np.arange(256)).cumsum();between=(s[-1]*w-s*w[-1])**2/np.maximum(1,w*(w[-1]-w));t=int(between.argmax());return a<=t
    mask=otsu(a);labs,_=ndimage.label(mask,np.ones((3,3)));heights=[]
    for s in ndimage.find_objects(labs):
        if s and s[0].stop-s[0].start>3 and (s[1].stop-s[1].start)<4*(s[0].stop-s[0].start):heights.append(s[0].stop-s[0].start)
    h=float(np.percentile(heights,70)) if heights else max(1,im.height);scale=min(48/max(h,1),1024/im.width,512/im.height)
    im=Image.fromarray(a).resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS);mask=otsu(np.array(im));return im,mask

def segment(source):
    im,mask=preprocess(source);binary=Image.fromarray(np.where(mask,0,255).astype('uint8'));boxes=components(binary,True,(.8,.7,.5));hvals=[b[3]-b[1] for b in boxes if b[3]-b[1]>3 and b[2]-b[0]<4*(b[3]-b[1])];h=max(1,float(np.median(hvals))) if hvals else 48
    used=set();merged=[]
    for i,p in enumerate(boxes):
        pw,ph=p[2]-p[0],p[3]-p[1]
        if pw>.25*h or ph>.25*h:continue
        choices=[]
        for j,b in enumerate(boxes):
            bw,bh=b[2]-b[0],b[3]-b[1];dx=abs((p[0]+p[2]-b[0]-b[2])/2);gap=b[1]-p[3]
            if i!=j and bh>=.5*h and bw<=.6*h and pw*ph<=.2*bw*bh and dx<=.25*h and 0<=gap<=.4*h:choices.append((dx/h+gap/h,j))
        choices.sort()
        if choices and (len(choices)==1 or choices[1][0]-choices[0][0]>=.1):
            j=choices[0][1]
            if i not in used and j not in used:
                b=boxes[j];used.update((i,j));merged.append([min(p[0],b[0]),min(p[1],b[1]),max(p[2],b[2]),max(p[3],b[3])])
    boxes=[b for i,b in enumerate(boxes) if i not in used]+merged;boxes=sorted(boxes,key=lambda b:((b[0]+b[2])/2,(b[1]+b[3])/2))
    if not boxes or len(boxes)>64:raise ValueError('empty_or_excess_segments')
    labels,_=ndimage.label(mask,np.ones((3,3)));arrays=[]
    for b in boxes:
        x0,y0,x1,y1=b;ids=np.unique(labels[y0:y1,x0:x1]);ids=ids[ids!=0]
        # Only components wholly assigned to this group are retained in its crop.
        patch=np.array(im.crop(b));pm=np.isin(labels[y0:y1,x0:x1],ids);patch=np.where(pm,patch,255).astype('uint8');arrays.append(crop(Image.fromarray(patch)))
    b=np.array(boxes,dtype=np.float32);centers=(b[:,:2]+b[:,2:])/2;delta=np.vstack((np.zeros((1,2),np.float32),np.diff(centers,axis=0)))/h
    geometry=np.column_stack((centers[:,0]/im.width,centers[:,1]/im.height,(b[:,2]-b[:,0])/im.width,(b[:,3]-b[:,1])/im.height,delta,(b[:,2]-b[:,0])/h,(b[:,3]-b[:,1])/h)).astype('float32');geometry[:,4:]=geometry[:,4:].clip(-4,4)
    return np.stack(arrays),geometry,boxes

class Model(nn.Module):
    def __init__(self):
        super().__init__();self.cnn=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(2048,64),nn.ReLU(),nn.Linear(64,70));self.project=nn.Sequential(nn.Linear(78,64),nn.ReLU());self.encoder=nn.LSTM(64,128,bidirectional=True,batch_first=True);self.embedding=nn.Embedding(79,64,padding_idx=0);self.decoder=nn.LSTM(64,256,batch_first=True);self.output=nn.Linear(256,79)
    def encode(self,images,geometry,lengths):
        p=self.cnn(images).softmax(-1);z=torch.cat((p,geometry),-1);z=nn.utils.rnn.pad_sequence(list(z.split(lengths)),batch_first=True);z=self.project(z);packed=nn.utils.rnn.pack_padded_sequence(z,lengths,batch_first=True,enforce_sorted=False);_,(h,c)=self.encoder(packed);return torch.cat((h[0],h[1]),-1)[None],torch.cat((c[0],c[1]),-1)[None]
    def forward(self,images,geometry,lengths,previous_tokens):
        state=self.encode(images,geometry,lengths);z,_=self.decoder(self.embedding(previous_tokens),state);return self.output(z)
    @torch.inference_mode()
    def recognize(self,image):
        self.eval();device=next(self.parameters()).device;x,g,b=segment(image);state=self.encode(torch.tensor(x,device=device),torch.tensor(g,device=device),[len(x)]);prev=torch.tensor([[1]],device=device);out=[];done=False
        for _ in range(64):
            z,state=self.decoder(self.embedding(prev),state);scores=self.output(z);scores[:,:,[0,1,3]]=-torch.inf;prev=scores.argmax(-1);k=prev.item()
            if k==2:done=True;break
            out.append(TOKENS[k])
        return {'tokens':out,'latex':' '.join(out),'terminated':done,'boxes':b,'status':'ok' if done else 'missing_eos'}
