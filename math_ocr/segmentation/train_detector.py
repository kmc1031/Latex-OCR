"""Class-agnostic CNN center/size detector. Box targets are training-only."""
import argparse,json,random,time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from common import ROOT,iou
from networks import SymbolDetector

class Images(Dataset):
    def __init__(self,split):
        self.rows=[json.loads(s) for s in (ROOT/f'data/{split}.jsonl').read_text().splitlines()]
        self.images=[np.array(Image.open(ROOT/'data'/r['image']),dtype=np.uint8) for r in self.rows]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.images[i],np.array([g['box'] for g in self.rows[i]['glyphs']],np.float32)

def collate(batch):
    height=max(a.shape[0] for a,b in batch);width=max(a.shape[1] for a,b in batch)
    x=np.zeros((len(batch),1,height,width),np.float32);heat=np.zeros((len(batch),1,height//4,width//4),np.float32);wh=np.zeros((len(batch),2,height//4,width//4),np.float32);off=wh.copy();mask=heat.copy()
    for i,(a,boxes) in enumerate(batch):
        x[i,0,:a.shape[0],:a.shape[1]]=(255-a.astype(np.float32))/255
        for box in boxes:
            center=(box[:2]+box[2:])/8;xx,yy=center.astype(int);xx=min(xx,width//4-1);yy=min(yy,height//4-1)
            for dy in range(-2,3):
                for dx in range(-2,3):
                    if 0<=yy+dy<height//4 and 0<=xx+dx<width//4:heat[i,0,yy+dy,xx+dx]=max(heat[i,0,yy+dy,xx+dx],np.exp(-(dx*dx+dy*dy)/2))
            heat[i,0,yy,xx]=1;mask[i,0,yy,xx]=1;wh[i,:,yy,xx]=(box[2:]-box[:2])/4;off[i,:,yy,xx]=center-[xx,yy]
    return [torch.from_numpy(v) for v in [x,heat,wh,off,mask]]

def losses(outputs,targets):
    heat,wh,offset=outputs;target,tw,to,mask=targets;p=heat.sigmoid().clamp(1e-5,1-1e-5);n=mask.sum().clamp(min=1)
    focal=(-(1-p).square()*p.log()*target.eq(1)-(p.square())*(1-p).log()*(1-target).pow(4)*target.lt(1)).sum()/n
    sizes=(torch.nn.functional.smooth_l1_loss(wh,tw,reduction='none')*mask).sum()/n
    offsets=(torch.nn.functional.smooth_l1_loss(offset,to,reduction='none')*mask).sum()/n
    return focal+.1*sizes+offsets

@torch.inference_mode()
def boxes_from_output(outputs,width,height,threshold=.25):
    heat,wh,off=outputs;p=heat.sigmoid();keep=p.eq(torch.nn.functional.max_pool2d(p,3,1,1));scores=(p*keep).flatten();values,indices=scores.topk(min(100,scores.numel()));result=[];w=p.shape[-1]
    for value,index in zip(values.tolist(),indices.tolist()):
        if value<threshold:break
        y,x=divmod(index,w);center=(torch.tensor([x,y],device=wh.device)+off[0,:,y,x])*4;size=wh[0,:,y,x]*4
        box=torch.cat([center-size/2,center+size/2]).cpu().numpy();box[[0,2]]=box[[0,2]].clip(0,width);box[[1,3]]=box[[1,3]].clip(0,height)
        if min(box[2:]-box[:2])>=1:result.append((box.tolist(),value))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--warm-start');args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(129);random.seed(129);np.random.seed(129)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');amp=device.type=='cuda'
    train=Images('train');val=Images('val')
    class Buckets:
        def __iter__(self):
            order=list(range(len(train)));random.shuffle(order);batches=[]
            for start in range(0,len(order),512):
                group=sorted(order[start:start+512],key=lambda i:(train.images[i].shape[0]//32,train.images[i].shape[1]//64))
                batches.extend(group[j:j+8] for j in range(0,len(group),8))
            random.shuffle(batches);yield from batches
        def __len__(self):return (len(train)+7)//8
    loader=DataLoader(train,batch_sampler=Buckets(),collate_fn=collate)
    model=SymbolDetector();package=torch.load(ROOT/'models/symbols.pt',map_location='cpu',weights_only=False);model.backbone.load_state_dict({k.removeprefix('backbone.'):v for k,v in package['model'].items() if k.startswith('backbone.')});model.to(device)
    if args.warm_start:
        old=torch.load(args.warm_start,map_location='cpu',weights_only=False)['model'];weights=model.state_dict()
        for key,value in old.items():
            if key not in weights:continue
            if weights[key].shape==value.shape:weights[key].copy_(value)
            elif key=='head.0.weight':weights[key].zero_();weights[key][:,:128].copy_(value)
        model.load_state_dict(weights)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.0001);scaler=torch.amp.GradScaler('cuda',enabled=amp);history=[];best=1e9
    # Fixed hash-shuffled validation images: no first-formula ordering bias.
    order=np.random.default_rng(17).permutation(len(val))[:200]
    vl=DataLoader(torch.utils.data.Subset(val,order),batch_size=8,collate_fn=collate)
    for epoch in range(1,args.epochs+1):
        start=time.perf_counter();model.train();total=count=0
        for batch in loader:
            batch=[x.to(device) for x in batch];optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.float16,enabled=amp):loss=losses(model(batch[0]),batch[1:])
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),3);scaler.step(optimizer);scaler.update();total+=float(loss.detach());count+=1
        model.eval();scores=[]
        with torch.inference_mode():
            for batch in vl:
                batch=[x.to(device) for x in batch];scores.append(float(losses(model(batch[0]),batch[1:])))
        score=float(np.mean(scores));entry={'epoch':epoch,'train_loss':total/count,'val_loss':score,'seconds':round(time.perf_counter()-start,2)};history.append(entry);print(json.dumps(entry),flush=True)
        if score<best:best=score;torch.save({'model':model.state_dict(),'epoch':epoch,'validation_loss':score,'wide_context':True,'warm_start':args.warm_start},ROOT/'models/detector.pt')
        (ROOT/'models/detector_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')

if __name__=='__main__':main()
