"""Image -> predicted boxes -> CNN symbols -> learned relations -> LaTeX.

Ground-truth boxes/labels are accepted only by explicit evaluation methods.
Normal recognize() takes an image and does not read dataset records.
"""
import json,math
import numpy as np
from PIL import Image,ImageOps,ImageFilter
from scipy import ndimage
import torch
from common import ROOT,crop_symbol,geometry,RELATIONS
from networks import SymbolCNN,RelationMLP,SymbolDetector
from train_detector import boxes_from_output
from physics_syntax import tokenize,result

def preprocess(source,return_transform=False):
    original_size=ImageOps.exif_transpose(source).size
    source=ImageOps.exif_transpose(source).convert('RGBA');white=Image.new('RGBA',source.size,'white');white.alpha_composite(source);gray=white.convert('L');gray.thumbnail((2400,2400))
    a=np.array(gray,dtype=np.float32);bg=np.array(gray.filter(ImageFilter.GaussianBlur(max(8,min(gray.size)*.06))),dtype=np.float32)
    corrected=Image.fromarray(np.clip(a*255/np.maximum(bg,1),0,255).astype(np.uint8));mask=np.array(corrected)<180
    bbox=Image.fromarray(mask.astype(np.uint8)*255).getbbox()
    if bbox is None:raise ValueError('Blank formula image')
    corrected=corrected.crop(bbox);labels,n=ndimage.label(np.array(corrected)<160,np.ones((3,3)))
    heights=[s[0].stop-s[0].start for s in ndimage.find_objects(labels) if s and s[0].stop-s[0].start>3 and (s[1].stop-s[1].start)/(s[0].stop-s[0].start)<6]
    h=float(np.percentile(heights,70)) if heights else corrected.height
    scale=min(48/max(h,1),1008/corrected.width,240/corrected.height)
    corrected=corrected.resize((max(1,round(corrected.width*scale)),max(1,round(corrected.height*scale))),Image.Resampling.LANCZOS)
    canvas=Image.new('L',(math.ceil((corrected.width+16)/4)*4,math.ceil((corrected.height+16)/4)*4),255);canvas.paste(corrected,(8,8))
    sx=corrected.width/(bbox[2]-bbox[0]);sy=corrected.height/(bbox[3]-bbox[1])
    transform=(sx*gray.width/original_size[0],sy*gray.height/original_size[1],8-bbox[0]*sx,8-bbox[1]*sy)
    return (canvas,transform) if return_transform else canvas

def expression_input(row):
    image,transform=preprocess(Image.open(ROOT/'data'/row['image']),True);sx,sy,dx,dy=transform
    boxes=[[b[0]*sx+dx,b[1]*sy+dy,b[2]*sx+dx,b[3]*sy+dy] for b in [g['box'] for g in row['glyphs']]]
    return image,boxes

def components(image):
    mask=np.array(image)<160;labels,n=ndimage.label(mask,np.ones((3,3)));boxes=[]
    for k,s in enumerate(ndimage.find_objects(labels),1):
        if s is None or (labels[s]==k).sum()<3:continue
        y,x=s;boxes.append([max(0,x.start-1),max(0,y.start-1),min(image.width,x.stop+1),min(image.height,y.stop+1)])
    return sorted(boxes,key=lambda b:-(b[2]-b[0])*(b[3]-b[1]))[:128]

def assemble(boxes,labels,probabilities,features,relation_model,device,script_threshold=.45):
    """Build 2-D structures, then an acyclic graph of baseline/script links."""
    n=len(boxes)
    if not n:return ''
    boxes=np.asarray(boxes,np.float32);med=max(float(np.median(boxes[:,3]-boxes[:,1])),8.)
    def relations(items):
        pairs=[(a,b) for a in items for b in items if a!=b]
        if not pairs:return {}
        p=np.array(pairs);g=torch.from_numpy(np.stack([geometry(boxes[a],boxes[b]) for a,b in pairs])).to(device)
        with torch.inference_mode():scores=relation_model(torch.cat([features[p[:,0]],features[p[:,1]],g],1)).softmax(-1).cpu().numpy()
        return dict(zip(pairs,scores))
    def region(items,depth=0,initial=None):
        if depth>12:return ' '.join(labels[i] for i in sorted(items,key=lambda i:boxes[i,0]))
        initial=initial or {};nodes={i:initial.get(i,labels[i]) for i in items};owned=set()
        # Wide horizontal bars and vertical placement identify fraction groups.
        bars=sorted(items,key=lambda i:-(boxes[i,2]-boxes[i,0]))
        for i in bars:
            if i in owned or i in initial:continue
            x0,y0,x1,y1=boxes[i];w=x1-x0;h=y1-y0
            if w<2.2*h or labels[i] not in ['-','\\frac']:continue
            above=[];below=[]
            for j in items:
                if i==j or j in owned:continue
                b=boxes[j];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
                if not x0-3<=cx<=x1+3:continue
                if cy<y0-.2*med and y0-b[3]<4*med:above.append(j)
                if cy>y1+.2*med and b[1]-y1<4*med:below.append(j)
            if above and below:
                nodes[i]='\\frac{'+region(above,depth+1)+'}{'+region(below,depth+1)+'}';owned.update(above+below)
        for i in sorted(items,key=lambda k:-(boxes[k,2]-boxes[k,0])):
            if i in owned or i in initial or labels[i]!='\\sqrt':continue
            b=boxes[i];inside=[j for j in items if j!=i and j not in owned and boxes[j,0]>b[0]+.12*(b[2]-b[0]) and boxes[j,2]<=b[2]+3 and boxes[j,1]>=b[1]-3 and boxes[j,3]<=b[3]+.45*med]
            if inside:nodes[i]='\\sqrt{'+region(inside,depth+1,{j:nodes[j] for j in inside if nodes[j]!=labels[j]})+'}';owned.update(inside)
        active=[i for i in items if i not in owned];scores=relations(active);parents={};children={i:{'Right':[],'Sup':[],'Sub':[]} for i in active}
        for j in active:
            best=None
            for i in active:
                if i==j or (boxes[i,0]+boxes[i,2])/2>=(boxes[j,0]+boxes[j,2])/2:continue
                dx=boxes[j,0]-boxes[i,2]
                if dx>3*med:continue
                score=scores[(i,j)];kind=int(np.argmax(score[1:4]))+1;value=float(score[kind])
                if kind in [2,3]:
                    if labels[i] in ['+','-','=','\\pm','\\cdot','\\times','(']:continue
                    if value<script_threshold:continue
                elif value<.15:continue
                value-=.035*max(dx,0)/med
                if best is None or value>best[0]:best=(value,i,RELATIONS[kind])
            if best:_,i,r=best;parents[j]=i;children[i][r].append(j)
        def subtree(i):
            text=nodes[i]
            for r,op in [('Sub','_'),('Sup','^')]:
                if children[i][r]:text+=op+'{'+row(children[i][r])+'}'
            return text+' '+row(children[i]['Right'])
        def row(ids):return ' '.join(subtree(i) for i in sorted(ids,key=lambda k:boxes[k,0]))
        return row([i for i in active if i not in parents]).strip()
    text=region(list(range(n)))
    # A connected-component proposal may split the letters of named operators.
    for name in ['sin','cos','tan','log','ln','exp']:
        import re
        text=re.sub(r'(?<![A-Za-z\\])'+r'\s+'.join(name)+r'(?![A-Za-z])',lambda _: '\\'+name,text)
    return text

class Pipeline:
    def __init__(self,device=None):
        torch.set_num_threads(4);self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        package=torch.load(ROOT/'models/symbols.pt',map_location='cpu',weights_only=False);self.vocab=package['vocab'];self.cnn=SymbolCNN(len(self.vocab));self.cnn.load_state_dict(package['model']);self.cnn.to(self.device).eval()
        p=torch.load(ROOT/'models/relations.pt',map_location='cpu',weights_only=False);self.relations=RelationMLP();self.relations.load_state_dict(p['model']);self.relations.to(self.device).eval()
        self.detector=None
        if (ROOT/'models/detector.pt').exists():
            self.detector=SymbolDetector();self.detector.load_state_dict(torch.load(ROOT/'models/detector.pt',map_location='cpu',weights_only=False)['model']);self.detector.to(self.device).eval()
        self.settings=json.loads((ROOT/'models/settings.json').read_text()) if (ROOT/'models/settings.json').exists() else {'method':'components','detector_threshold':.25,'script_threshold':.45}

    @torch.inference_mode()
    def classify(self,image,boxes):
        if not boxes:return [],np.empty((0,len(self.vocab))),torch.empty((0,128),device=self.device)
        x=torch.from_numpy((255-np.stack([crop_symbol(image,b) for b in boxes]).astype(np.float32))/255)[:,None]
        features=torch.cat([self.cnn.features(batch.to(self.device)) for batch in x.split(128)])
        p=self.cnn.output(features).softmax(-1).cpu().numpy();labels=[self.vocab[i] for i in p.argmax(1)];return labels,p,features

    def propose(self,image,method,threshold):
        if method=='detector':
            if self.detector is None:raise ValueError('Detector not trained')
            x=torch.from_numpy((255-np.array(image).astype(np.float32))/255)[None,None].to(self.device)
            return [b for b,s in boxes_from_output(self.detector(x),image.width,image.height,threshold)]
        boxes=components(image)
        # Candidate merging for disconnected '=' and dotted i/j. The CNN must
        # recognize the union; no source strokes or target counts are consulted.
        if not boxes:return boxes
        med=max(float(np.percentile([b[3]-b[1] for b in boxes],65)),8);candidates=[];pairs=[]
        for i,a in enumerate(boxes):
            for j,b in enumerate(boxes[:i]):
                xover=max(0,min(a[2],b[2])-max(a[0],b[0]));gap=max(a[1],b[1])-min(a[3],b[3])
                if xover>0 and gap<.65*med:
                    union=[min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]
                    if union[3]-union[1]<1.7*med:candidates.append(union);pairs.append((i,j))
        if candidates:
            labs,ps,_=self.classify(image,candidates);used=set();merged=[]
            order=np.argsort(-ps.max(1))
            for k in order:
                i,j=pairs[k]
                if i in used or j in used:continue
                if labs[k] in ['=','i','j'] and ps[k].max()>.55:merged.append(candidates[k]);used.update([i,j])
            boxes=[b for i,b in enumerate(boxes) if i not in used]+merged
        return boxes

    @torch.inference_mode()
    def recognize(self,image,prepared=False,method=None,threshold=None,boxes=None,labels=None,script_threshold=None):
        image=image.convert('L') if prepared else preprocess(image)
        detected=self.propose(image,method or self.settings['method'],threshold if threshold is not None else self.settings['detector_threshold']) if boxes is None else boxes
        predicted,prob,features=self.classify(image,detected)
        latex=assemble(detected,labels if labels is not None else predicted,prob,features,self.relations,self.device,script_threshold if script_threshold is not None else self.settings['script_threshold'])
        tokens=tokenize(latex);tree,converted=result(tokens)
        return {'prediction':' '.join(tokens),'tokens':tokens,'latex':converted or latex,'ast':tree,'syntax_valid':tree is not None,
            'boxes':detected,'labels':predicted,'scores':prob.max(1).tolist() if len(prob) else [],'prepared':image}
