"""AI-assisted implementation of the chat experiment plan, 2026-11-05.

Local CROHME glyph composites, NOT MathWriting or photographs. O is unavailable.
Existing source test glyphs have been used in earlier project experiments.
Fresh expression-disjoint composites are diagnostic, not a new benchmark.
References: ../segmentation/data/audit.json and ../segmentation/raw/provenance.json.
"""
import copy, json, random, time, hashlib, argparse
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn

BASE=Path(__file__).resolve().parent
ROOT=BASE/'isolated'
SOURCE=BASE.parent/'segmentation'/'data'
CLASSES=list('0123456789abxyzl')+['+','-','=','\\times','/','(',')']
VOCAB=['<PAD>','<BOS>','<EOS>','<UNK>']+CLASSES+['\\frac','^','_','{','}']
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4)

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def seed(n): random.seed(n);np.random.seed(n);torch.manual_seed(n)

def crop(im,box=None):
    if box is not None: im=im.crop(tuple(map(int,box)))
    im=im.convert('L'); bb=Image.fromarray((np.array(im)<180).astype('uint8')*255).getbbox()
    if bb: im=im.crop(bb)
    im.thumbnail((28,28));out=Image.new('L',(32,32),255)
    out.paste(im,((32-im.width)//2,(32-im.height)//2))
    return (255-np.asarray(out).astype('float32'))[None]/255

def components(im,merge=True,profile=(.8,.7,.5)):
    a=np.asarray(im)<160; labs,n=ndimage.label(a,np.ones((3,3)));boxes=[]
    for k,s in enumerate(ndimage.find_objects(labs),1):
        if s is not None and (labs[s]==k).sum()>=3:
            y,x=s;boxes.append([x.start,y.start,x.stop,y.stop])
    heights=[b[3]-b[1] for b in boxes if b[3]-b[1]>4 and (b[2]-b[0])/(b[3]-b[1])<4]
    h=float(np.median(heights)) if heights else 32
    if merge:
        candidates=[]
        for i,a in enumerate(boxes):
            aw,ah=a[2]-a[0],a[3]-a[1]
            if aw<3*ah:continue
            for j,b in enumerate(boxes[i+1:],i+1):
                bw,bh=b[2]-b[0],b[3]-b[1]
                overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))/min(aw,bw)
                gap=max(a[1],b[1])-min(a[3],b[3])
                if bw>=3*bh and overlap>=profile[0] and min(aw,bw)/max(aw,bw)>=profile[1] and 0<gap<=profile[2]*h:
                    candidates.append((overlap-gap/h,i,j))
        used=set();merged=[]
        for _,i,j in sorted(candidates,reverse=True):
            if i in used or j in used:continue
            a,b=boxes[i],boxes[j]
            between=any(k not in (i,j) and max(a[1],b[1])>c[3] and min(a[3],b[3])<c[1] and min(a[2],b[2])>c[0] and max(a[0],b[0])<c[2] for k,c in enumerate(boxes))
            if between:continue
            used.update((i,j));merged.append([min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])])
        boxes=[b for i,b in enumerate(boxes) if i not in used]+merged
    return sorted(boxes,key=lambda b:((b[0]+b[2])/2,(b[1]+b[3])/2))

def box_counts(pred,true):
    if not pred or not true:return [0,len(pred),len(true)]
    p=np.array(pred);t=np.array(true);lt=np.maximum(p[:,None,:2],t[None,:,:2]);rb=np.minimum(p[:,None,2:],t[None,:,2:])
    inter=np.maximum(0,rb-lt).prod(-1);area=(p[:,2:]-p[:,:2]).prod(-1)[:,None]+(t[:,2:]-t[:,:2]).prod(-1)[None,:]
    iou=inter/np.maximum(1,area-inter)
    # Maximise number of valid matches before their IoU sum.
    r,c=linear_sum_assignment(-((iou>=.5)*1000+iou));tp=int((iou[r,c]>=.5).sum())
    return [tp,len(pred)-tp,len(true)-tp]

def segscore(rows,merge,profile):
    counts=np.zeros(3,dtype=int)
    for r in rows:counts+=box_counts(components(Image.open(ROOT/r['image']),merge,profile),[g['box'] for g in r['glyphs']])
    tp,fp,fn=map(int,counts);return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn)}

def prepare():
    rng=random.Random(20261105); raw=json.loads((SOURCE/'glyphs.json').read_text());arr=np.load(SOURCE/'glyphs.npy',mmap_mode='r')
    pools={s:{c:[] for c in CLASSES} for s in ['train','val','test']}
    for i,r in enumerate(raw):
        if r['label'] in CLASSES:pools[r['split']][r['label']].append(i)
    # Source expressions must not cross pools. Existing official test source stays test.
    origins={s:{raw[i]['expression'] for ids in pool.values() for i in ids} for s,pool in pools.items()}
    assert not any(origins[a]&origins[b] for a,b in [('train','val'),('train','test'),('val','test')])
    xs={};ys={};idsout={}
    for s,cap in [('train',100),('val',30),('test',30)]:
        chosen=[]
        for c in CLASSES:
            ids=pools[s][c].copy();rng.shuffle(ids);chosen+=ids[:cap]
        xs[s]=np.stack([crop(Image.fromarray(arr[i,1])) for i in chosen]);ys[s]=np.array([CLASSES.index(raw[i]['label']) for i in chosen]);idsout[s]=chosen
    np.savez_compressed(ROOT/'glyphs.npz',**{f'x_{s}':xs[s] for s in xs},**{f'y_{s}':ys[s] for s in ys})
    records={s:[] for s in pools};used=set();hashes={s:set() for s in pools}
    atoms=list('0123456789abxyzl')
    def glyph(c,s,w,h):
        idx=rng.choice(pools[s][c]);im=Image.fromarray(arr[idx,1]);bb=Image.fromarray((np.array(im)<180).astype('uint8')*255).getbbox();im=im.crop(bb) if bb else im
        if c=='-' and w==130:im=im.resize((w,h))
        else:im.thumbnail((w,h))
        return im,idx
    for s,basic,script,pairs in [('train',700,700,350),('val',150,150,75),('test',150,150,75)]:
        dest=ROOT/'images'/s;dest.mkdir(parents=True,exist_ok=True)
        for kind,num in [('basic',basic),('sup',script//2),('sub',script-script//2),('fraction',pairs)]:
            for k in range(num):
                while True:
                    a,b,c=[rng.choice(atoms) for _ in range(3)];op=rng.choice(['+','-','\\times']);d=rng.choice(list('123456789'))
                    if kind=='basic':tokens=[a,op,b,'=',c]
                    elif kind in ('sup','sub'):tokens=[a,'^' if kind=='sup' else '_','{',d,'}',op,b]
                    else:tokens=['\\frac','{',a,'+',b,'}','{',c,'}']
                    key=' '.join(tokens)
                    if key not in used:used.add(key);break
                eid=hashlib.sha256(key.encode()).hexdigest()[:16]
                for style in (['vertical','slash'] if kind=='fraction' else ['normal']):
                    im=Image.new('L',(260,120),255);gs=[]
                    def put(label,x,y,w=32,h=44):
                        patch,idx=glyph(label,s,w,h);im.paste(patch,(x,y));pa=np.array(patch)<180;bb=Image.fromarray(pa.astype('uint8')*255).getbbox()
                        gs.append({'label':label,'box':[x+bb[0],y+bb[1],x+bb[2],y+bb[3]],'source_glyph':idx})
                    if kind=='basic':
                        for q,label in enumerate(tokens):put(label,12+q*45+rng.randrange(-2,3),40+rng.randrange(-3,4))
                    elif kind in ('sup','sub'):
                        put(a,12,40);put(d,45,10 if kind=='sup' else 78,20,24);put(op,85,40);put(b,130,40)
                    elif style=='vertical':
                        for q,label in enumerate([a,'+',b]):put(label,22+q*40,6,28,34)
                        # The fraction bar is a visible HLINE class, not a command crop.
                        put('-',16,52,130,5);put(c,62,70,30,36)
                    else:
                        for q,label in enumerate(['(',a,'+',b,')','/',c]):put(label,8+q*35,40,28,44)
                    rel=f'images/{s}/{eid}_{style}.png';im.save(ROOT/rel);digest=hashlib.sha256(im.tobytes()).hexdigest();hashes[s].add(digest)
                    records[s].append({'id':eid+'_'+style,'expression_id':eid,'image':rel,'tokens':tokens,'kind':kind,'style':style,'glyphs':gs,'image_sha256':digest})
        save(ROOT/f'{s}.json',records[s])
    assert not any(hashes[a]&hashes[b] for a,b in [('train','val'),('train','test'),('val','test')])
    save(ROOT/'data_audit.json',{'source':'Local CROHME glyphs; handwritten composites only','missing_classes':['O'],'classes':CLASSES,'counts':{s:len(rs) for s,rs in records.items()},'glyph_counts':{s:dict(Counter(raw[i]['label'] for i in idsout[s])) for s in idsout},'glyph_ids':idsout,'source_expression_disjoint':True,'formula_disjoint':True,'pixel_hash_disjoint':True,'real_photo_count':0,'caveat':'Source test glyphs were used by earlier project experiments. New formula composites are not independent writer/photo benchmarks.'})

class Classifier(nn.Module):
    def __init__(self,mlp=False):
        super().__init__();self.net=nn.Sequential(nn.Flatten(),nn.Linear(1024,128),nn.ReLU(),nn.Linear(128,len(CLASSES))) if mlp else nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(2048,64),nn.ReLU(),nn.Linear(64,len(CLASSES)))
    def forward(self,x):return self.net(x)

def clsmetrics(y,p):
    cm=np.zeros((len(CLASSES),len(CLASSES)),int);np.add.at(cm,(y,p),1);tp=cm.diagonal();f=2*tp/np.maximum(1,cm.sum(0)+cm.sum(1));return {'n':len(y),'accuracy':float((y==p).mean()),'macro_f1':float(f.mean()),'confusion':cm.tolist()}

@torch.inference_mode()
def classify(model,x,y):
    model.eval();p=torch.cat([model(torch.tensor(z,device=DEVICE)).argmax(1).cpu() for z in np.array_split(x,max(1,len(x)//256))]).numpy();return clsmetrics(y,p)

def traincnn(kind,n,epochs):
    seed(n);d=np.load(ROOT/'glyphs.npz');m=Classifier(kind=='C0').to(DEVICE);opt=torch.optim.Adam(m.parameters(),lr=.001);best=-1;wait=0;history=[]
    x=torch.tensor(d['x_train'],device=DEVICE);y=torch.tensor(d['y_train'],device=DEVICE);state=None
    for ep in range(epochs):
        m.train();losses=[]
        for idx in torch.randperm(len(x),device=DEVICE).split(64):
            bx=x[idx]
            if kind=='C2':
                theta=torch.eye(2,3,device=DEVICE).repeat(len(idx),1,1);angle=torch.empty(len(idx),device=DEVICE).uniform_(-5,5)*np.pi/180;scale=torch.empty(len(idx),device=DEVICE).uniform_(.9,1.1)
                theta[:,0,0]=angle.cos()*scale;theta[:,1,1]=angle.cos()*scale;theta[:,0,1]=-angle.sin()*scale;theta[:,1,0]=angle.sin()*scale;theta[:,:,2]=torch.empty(len(idx),2,device=DEVICE).uniform_(-.125,.125)
                bx=nn.functional.grid_sample(bx,nn.functional.affine_grid(theta,bx.shape,align_corners=False),align_corners=False)
            opt.zero_grad();loss=nn.functional.cross_entropy(m(bx),y[idx]);loss.backward();opt.step();losses.append(float(loss.detach()))
        v=classify(m,d['x_val'],d['y_val']);history.append({'epoch':ep+1,'loss':float(np.mean(losses)),'val':v['macro_f1']})
        if v['macro_f1']>best:best=v['macro_f1'];state=copy.deepcopy(m.state_dict());wait=0
        else:wait+=1
        if wait>=5:break
    m.load_state_dict(state);torch.save({'state':state,'kind':kind,'classes':CLASSES},ROOT/f'{kind}_{n}.pt');save(ROOT/f'{kind}_{n}_history.json',history)
    return m,{'validation':classify(m,d['x_val'],d['y_val']),'test':None,'epochs':len(history)}

@torch.inference_mode()
def features(model,rows,mode,profile):
    model.eval();out=[]
    for r in rows:
        im=Image.open(ROOT/r['image']).convert('L')
        if mode=='auto':gs=[];bs=components(im,True,profile)
        else:
            gs=sorted(r['glyphs'],key=lambda g:((g['box'][0]+g['box'][2])/2,(g['box'][1]+g['box'][3])/2));bs=[g['box'] for g in gs]
        if not bs:out.append(np.zeros((1,len(CLASSES)+4),np.float32));continue
        p=model(torch.tensor(np.stack([crop(im,b) for b in bs]),device=DEVICE)).softmax(-1).cpu().numpy()
        if mode=='oracle':p=np.eye(len(CLASSES),dtype=np.float32)[[CLASSES.index(g['label']) for g in gs]]
        b=np.array(bs,dtype=np.float32);geo=np.column_stack(((b[:,0]+b[:,2])/2/im.width,(b[:,1]+b[:,3])/2/im.height,(b[:,2]-b[:,0])/im.width,(b[:,3]-b[:,1])/im.height))
        out.append(np.concatenate((p,geo),1).astype('float32'))
    return out

class Seq(nn.Module):
    def __init__(self,position=True):
        super().__init__();self.position=position;self.proj=nn.Sequential(nn.Linear(len(CLASSES)+4,64),nn.ReLU());self.enc=nn.LSTM(64,128,batch_first=True);self.emb=nn.Embedding(len(VOCAB),32,padding_idx=0);self.dec=nn.LSTM(32,128,batch_first=True);self.out=nn.Linear(128,len(VOCAB))
    def encode(self,x,lens):
        if not self.position:x=x.clone();x[:,:,-4:]=0
        z=nn.utils.rnn.pack_padded_sequence(self.proj(x),lens.cpu(),batch_first=True,enforce_sorted=False);return self.enc(z)[1]
    def forward(self,x,lens,t):
        z,_=self.dec(self.emb(t),self.encode(x,lens));return self.out(z)

def batch(fs,rows,idx):
    x=nn.utils.rnn.pad_sequence([torch.tensor(fs[i]) for i in idx],batch_first=True).to(DEVICE);lens=torch.tensor([len(fs[i]) for i in idx]);ts=[torch.tensor([1]+[VOCAB.index(t) for t in rows[i]['tokens']]+[2]) for i in idx];t=nn.utils.rnn.pad_sequence(ts,batch_first=True).to(DEVICE);return x,lens,t

def edit(a,b):
    v=list(range(len(b)+1))
    for i,x in enumerate(a):
        z=[i+1]
        for j,y in enumerate(b):z.append(min(z[-1]+1,v[j+1]+1,v[j]+(x!=y)))
        v=z
    return v[-1]

@torch.inference_mode()
def evaluate(m,fs,rows,details=False):
    m.eval();predictions=[]
    for start in range(0,len(rows),64):
        ids=list(range(start,min(start+64,len(rows))));x,lens,t=batch(fs,rows,ids);state=m.encode(x,lens);prev=torch.ones(len(ids),1,dtype=torch.long,device=DEVICE);pred=[[] for _ in ids];done=[False]*len(ids)
        for step in range(64):
            z,state=m.dec(m.emb(prev),state);prev=m.out(z).argmax(-1)
            for j,k in enumerate(prev[:,0].tolist()):
                if not done[j]:
                    if k==2:done[j]=True
                    else:pred[j].append(VOCAB[k])
            if all(done):break
        for j,i in enumerate(ids):predictions.append({'id':rows[i]['id'],'kind':rows[i]['kind'],'style':rows[i]['style'],'truth':rows[i]['tokens'],'pred':pred[j],'exact':pred[j]==rows[i]['tokens'],'distance':edit(pred[j],rows[i]['tokens']),'terminated':done[j]})
    def score(rs):return {'n':len(rs),'exact_match':sum(r['exact'] for r in rs)/max(1,len(rs)),'token_error_rate':sum(r['distance'] for r in rs)/max(1,sum(len(r['truth']) for r in rs)),'termination_rate':sum(r['terminated'] for r in rs)/max(1,len(rs))}
    result=score(predictions);result['by_type']={k:score([r for r in predictions if (r['style'] if r['kind']=='fraction' else r['kind'])==k]) for k in ['basic','sup','sub','vertical','slash']}
    if details:result['predictions']=predictions
    return result

def trainseq(name,n,cnn,rows,val,profile,epochs):
    seed(n);position=name!='L0';style={'Fvertical':'vertical','Fslash':'slash'}.get(name)
    if style:
        rows=[r for r in rows if r['kind']!='fraction' or r['style']==style];val=[r for r in val if r['kind']!='fraction' or r['style']==style]
    fa=features(cnn,rows,'auto',profile);fg=features(cnn,rows,'gt',profile);vf=features(cnn,val,'auto',profile);trainrows=rows+rows;fs=fa+fg
    m=Seq(position).to(DEVICE);opt=torch.optim.Adam(m.parameters(),lr=.001);best=(-1,-1e9);wait=0;history=[];state=None
    for ep in range(epochs):
        m.train();order=np.random.permutation(len(fs));losses=[]
        for off in range(0,len(order),32):
            x,l,t=batch(fs,trainrows,order[off:off+32]);opt.zero_grad();logits=m(x,l,t[:,:-1]);loss=nn.functional.cross_entropy(logits.reshape(-1,len(VOCAB)),t[:,1:].reshape(-1),ignore_index=0);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();losses.append(float(loss.detach()))
        v=evaluate(m,vf,val);score=(v['exact_match'],-v['token_error_rate']);history.append({'epoch':ep+1,'loss':float(np.mean(losses)),'validation':v});print(name,n,ep+1,score,flush=True)
        if score>best:best=score;state=copy.deepcopy(m.state_dict());wait=0
        else:wait+=1
        save(ROOT/f'{name}_{n}_history.json',history)
        if wait>=5:break
    m.load_state_dict(state);torch.save({'state':state,'position':position,'vocab':VOCAB,'classes':CLASSES,'profile':profile},ROOT/f'{name}_{n}.pt');return m,{'validation':evaluate(m,vf,val),'epochs':len(history)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cnn-epochs',type=int,default=30);ap.add_argument('--lstm-epochs',type=int,default=50);args=ap.parse_args();ROOT.mkdir(exist_ok=True);seed(20261105)
    if not (ROOT/'data_audit.json').exists():prepare()
    data={s:json.loads((ROOT/f'{s}.json').read_text()) for s in ['train','val','test']}
    profiles=[(.9,.8,.35),(.8,.7,.5),(.7,.6,.65)];scores=[segscore(data['val'],True,p) for p in profiles];profile=profiles[max(range(3),key=lambda i:scores[i]['f1'])];save(ROOT/'segmentation_selection.json',{'profiles':profiles,'validation':scores,'selected':profile})
    results={'scope':'23 classes, CROHME glyph composites; no photographs; reused historical source glyphs','device':DEVICE,'config':vars(args),'runs':{},'segmentation':{}}
    gd=np.load(ROOT/'glyphs.npz');trained={};start=time.time()
    for n in [42,43,44]:
        trained[n]={}
        for kind in ['C0','C1','C2']:
            print('CNN',kind,n,flush=True);m,res=traincnn(kind,n,args.cnn_epochs);trained[n][kind]=m;results['runs'][f'{kind}_{n}']=res;save(ROOT/'progress.json',results)
    selected=max(['C1','C2'],key=lambda k:np.mean([results['runs'][f'{k}_{n}']['validation']['macro_f1'] for n in [42,43,44]]));results['selected_cnn']=selected
    # All architecture/checkpoint decisions use validation only. Test is evaluated after training.
    for n in [42,43,44]:
        cnn=trained[n][selected]
        for name in ['L0','L1','Fvertical','Fslash']:
            m,res=trainseq(name,n,cnn,data['train'],data['val'],profile,args.lstm_epochs);results['runs'][f'{name}_{n}']=res;save(ROOT/'progress.json',results)
    selected_lstm=max(['L0','L1'],key=lambda k:np.mean([results['runs'][f'{k}_{n}']['validation']['exact_match'] for n in [42,43,44]]));results['selected_lstm']=selected_lstm
    save(ROOT/'frozen_selection.json',{'cnn':selected,'lstm':selected_lstm,'profile':profile,'selection_data':'validation only'})
    for n in [42,43,44]:
        for kind in ['C0','C1','C2']:results['runs'][f'{kind}_{n}']['test']=classify(trained[n][kind],gd['x_test'],gd['y_test'])
        cnn=trained[n][selected];auto=features(cnn,data['test'],'auto',profile)
        for name in ['L0','L1','Fvertical','Fslash']:
            ck=torch.load(ROOT/f'{name}_{n}.pt',map_location=DEVICE,weights_only=False);m=Seq(ck['position']).to(DEVICE);m.load_state_dict(ck['state']);test=evaluate(m,auto,data['test'],True);save(ROOT/f'{name}_{n}_predictions.json',test.pop('predictions'));results['runs'][f'{name}_{n}']['test']=test
            if name==selected_lstm:
                results['runs'][f'{name}_{n}']['diagnostics']={mode:evaluate(m,features(cnn,data['test'],mode,profile),data['test']) for mode in ['gt','oracle']}
        save(ROOT/'progress.json',results)
    results['segmentation']={'S0':segscore(data['test'],False,profile),'S1':segscore(data['test'],True,profile)};results['seconds']=time.time()-start;results['torch']=torch.__version__;results['gpu']=torch.cuda.get_device_name(0) if DEVICE=='cuda' else None;save(ROOT/'results.json',results)
    print('COMPLETE',results['seconds'],flush=True)

if __name__=='__main__':main()
