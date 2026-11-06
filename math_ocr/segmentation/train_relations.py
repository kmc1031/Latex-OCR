"""Train visual+geometry relation classification, with frozen symbol features."""
from collections import Counter
import json,random,time
import numpy as np
import torch
from common import ROOT,geometry,RELATIONS
from networks import SymbolCNN,RelationMLP

def main():
    torch.set_num_threads(4);torch.manual_seed(719);rng=random.Random(719)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    package=torch.load(ROOT/'models/symbols.pt',map_location='cpu',weights_only=False)
    cnn=SymbolCNN(len(package['vocab']));cnn.load_state_dict(package['model']);cnn.to(device).eval()
    images=np.load(ROOT/'data/glyphs.npy',mmap_mode='r');features=[]
    with torch.inference_mode():
        for i in range(0,len(images),256):
            x=torch.from_numpy((255-images[i:i+256,0].astype(np.float32))/255)[:,None].to(device)
            features.append(cnn.features(x).cpu())
    features=torch.cat(features).to(device);np.save(ROOT/'models/glyph_features.npy',features.cpu().numpy())
    data={}
    for split in ['train','val']:
        rows=[json.loads(s) for s in (ROOT/f'data/{split}.jsonl').read_text().splitlines()];pairs=[];geos=[];labels=[]
        for row in rows:
            glyphs=row['glyphs'];truth={(a,b):RELATIONS.index(r) for a,b,r in row['edges']}
            for i,g in enumerate(glyphs):
                nearest=sorted([j for j in range(len(glyphs)) if j!=i],key=lambda j:float(np.linalg.norm(np.array(g['box'][:2])-glyphs[j]['box'][:2])))[:8]
                chosen=set(nearest)|{b for a,b in truth if a==i}
                for j in sorted(chosen):
                    target=truth.get((i,j),0)
                    if split=='train' and target==0 and rng.random()>.45:continue
                    pairs.append([row['glyph_indexes'][i],row['glyph_indexes'][j]]);geos.append(geometry(g['box'],glyphs[j]['box']));labels.append(target)
        data[split]=(torch.tensor(pairs,device=device),torch.from_numpy(np.array(geos)).to(device),torch.tensor(labels,device=device))
    model=RelationMLP().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    counts=torch.bincount(data['train'][2],minlength=7).float();weight=(counts.sum()/counts.clamp(min=1)).sqrt();weight/=weight.mean()
    history=[];best=-1
    print('Relation pairs', {s:len(v[0]) for s,v in data.items()},flush=True)
    for epoch in range(1,26):
        began=time.perf_counter();model.train();p,g,y=data['train'];order=torch.randperm(len(p),device=device);total=0
        for ix in order.split(1024):
            inputs=torch.cat([features[p[ix,0]],features[p[ix,1]],g[ix]],1)
            optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(inputs),y[ix],weight=weight);loss.backward();optimizer.step();total+=float(loss.detach())*len(ix)
        model.eval();p,g,y=data['val'];pred=[]
        with torch.inference_mode():
            for ix in torch.arange(len(p),device=device).split(2048):pred.append(model(torch.cat([features[p[ix,0]],features[p[ix,1]],g[ix]],1)).argmax(1))
        pred=torch.cat(pred);conf=torch.bincount(y*7+pred,minlength=49).reshape(7,7).float();recall=conf.diag()/conf.sum(1).clamp(min=1);precision=conf.diag()/conf.sum(0).clamp(min=1);f1=2*recall*precision/(recall+precision).clamp(min=1e-6)
        macro=float(f1[1:].mean());entry={'epoch':epoch,'loss':total/len(data['train'][0]),'accuracy':float(pred.eq(y).float().mean()),'positive_macro_f1':macro,'per_relation_f1':dict(zip(RELATIONS,f1.tolist())),'seconds':round(time.perf_counter()-began,2)}
        history.append(entry);print(json.dumps(entry),flush=True)
        if macro>best:
            best=macro;torch.save({'model':model.state_dict(),'epoch':epoch,'validation':entry,'relations':RELATIONS,'symbol_epoch':package['epoch']},ROOT/'models/relations.pt')
        (ROOT/'models/relation_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')

if __name__=='__main__':main()
