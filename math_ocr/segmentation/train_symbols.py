import argparse,json,random,time
import numpy as np
from PIL import Image,ImageFilter
import torch
from torch.utils.data import Dataset,DataLoader
from common import ROOT
from networks import SymbolCNN

class Glyphs(Dataset):
    def __init__(self,split,augment=False):
        self.images=np.load(ROOT/'data/glyphs.npy',mmap_mode='r');self.rows=json.loads((ROOT/'data/glyphs.json').read_text());self.vocab=json.loads((ROOT/'data/vocab.json').read_text());self.lookup={t:i for i,t in enumerate(self.vocab)}
        self.ids=[i for i,r in enumerate(self.rows) if r['split']==split];self.augment=augment
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        i=self.ids[index];mode=random.randrange(2) if self.augment else 0;arr=self.images[i,mode].copy()
        if self.augment:
            image=Image.fromarray(arr).rotate(random.uniform(-7,7),resample=Image.Resampling.BILINEAR,fillcolor=255)
            if random.random()<.2:image=image.filter(ImageFilter.GaussianBlur(random.uniform(.1,.5)))
            arr=np.asarray(image)
        return torch.from_numpy((255-arr).astype(np.float32)/255)[None],self.lookup[self.rows[i]['label']]

@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval();correct=top3=total=0;per={}
    for x,y in loader:
        logits=model(x.to(device));ids=logits.topk(3,-1).indices.cpu();correct+=int(ids[:,0].eq(y).sum());top3+=int(ids.eq(y[:,None]).any(1).sum());total+=len(y)
        for target,pred in zip(y.tolist(),ids[:,0].tolist()):
            item=per.setdefault(target,[0,0]);item[0]+=1;item[1]+=int(target==pred)
    return {'n':total,'accuracy':correct/total,'top3':top3/total,'macro_accuracy':float(np.mean([c/n for n,c in per.values()]))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=22);ap.add_argument('--batch-size',type=int,default=192);args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(317);np.random.seed(317);random.seed(317)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');amp=device.type=='cuda'
    train=Glyphs('train',True);val=Glyphs('val');model=SymbolCNN(len(train.vocab)).to(device)
    loaders=[DataLoader(ds,batch_size=args.batch_size,shuffle=i==0,num_workers=0) for i,ds in enumerate([train,val])]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);scaler=torch.amp.GradScaler('cuda',enabled=amp);best=0;history=[]
    out=ROOT/'models';out.mkdir(exist_ok=True)
    print(json.dumps({'train':len(train),'val':len(val),'parameters':sum(p.numel() for p in model.parameters())}),flush=True)
    for epoch in range(1,args.epochs+1):
        start=time.perf_counter();model.train();loss_sum=count=0
        for group in optimizer.param_groups:group['lr']=.001*(.15+.85*.5*(1+np.cos(np.pi*(epoch-1)/args.epochs)))
        for x,y in loaders[0]:
            x=x.to(device);y=y.to(device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.float16,enabled=amp):
                loss=torch.nn.functional.cross_entropy(model(x),y,label_smoothing=.02)
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),3);scaler.step(optimizer);scaler.update()
            loss_sum+=float(loss.detach())*len(y);count+=len(y)
        score=evaluate(model,loaders[1],device);entry={'epoch':epoch,'loss':loss_sum/count,'validation':score,'seconds':round(time.perf_counter()-start,2)};history.append(entry);print(json.dumps(entry),flush=True)
        if score['accuracy']>best:
            best=score['accuracy'];torch.save({'model':model.state_dict(),'vocab':train.vocab,'epoch':epoch,'validation':score,'attention':False,'ctc':False},out/'symbols.pt')
        (out/'symbol_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')

if __name__=='__main__':main()
