"""Joint training of the ONE digital-ink experiment; photo exclusion authorized."""
import copy,json,random,time,hashlib
import numpy as np
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
import torch
from torch import nn
from engine import ROOT,CFG,Model,TOKENS,segment
from syntax import normalize

def save(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def batch(rows,device):
    xs=[];gs=[];lengths=[];targets=[]
    for row in rows:
        x,g,_=segment(Image.open(ROOT/row['image']));xs.append(torch.tensor(x));gs.append(torch.tensor(g));lengths.append(len(x));targets.append(torch.tensor([1]+[TOKENS.index(t) for t in row['tokens']]+[2]))
    return torch.cat(xs).to(device),torch.cat(gs).to(device),lengths,nn.utils.rnn.pad_sequence(targets,batch_first=True).to(device)

def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=row
    return previous[-1]

def evaluate(model,rows,details=False):
    predictions=[]
    for r in rows:
        try:p=model.recognize(Image.open(ROOT/r['image']));out=p['tokens'];terminated=p['terminated']
        except (ValueError,OSError) as exc:p={'status':str(exc)};out=[];terminated=False
        valid=True
        try:normalized,identity=normalize(out)
        except (ValueError,RecursionError):normalized=out;identity=None;valid=False
        target,truth=normalize(r['tokens']);predictions.append({'sample_id':r['sample_id'],'kind':r['kind'],'truth':target,'pred':out,'normalized_pred':normalized,'valid':valid,'terminated':terminated,'exact':bool(valid and terminated and identity==truth),'raw_exact':bool(terminated and out==r['tokens']),'distance':distance(normalized,target),'status':p['status']})
    n=len(predictions);result={'n':n,'exact_match':sum(p['exact'] for p in predictions)/max(1,n),'raw_exact_match':sum(p['raw_exact'] for p in predictions)/max(1,n),'token_error_rate':sum(p['distance'] for p in predictions)/max(1,sum(len(p['truth']) for p in predictions)),'syntax_error_rate':sum(not p['valid'] for p in predictions)/max(1,n)}
    if details:result['predictions']=predictions
    return result

def prepare_cache(real,synthetic):
    signature=hashlib.sha256(json.dumps(real+synthetic,sort_keys=True).encode()+(ROOT/'engine.py').read_bytes()+(ROOT/'syntax.py').read_bytes()).hexdigest()
    path=ROOT/'input_cache.pt'
    if path.exists():
        old=torch.load(path,weights_only=False)
        if old['signature']==signature:return old
    rng=random.Random(42);cached=[]
    for i,r in enumerate([r for r in real if r['split']!='test']+synthetic):
        variants=[];im=Image.open(ROOT/r['image']).convert('L')
        for view in range(2 if r['split']=='train' else 1):
            image=im
            if view:
                image=ImageOps.expand(image,border=(rng.randrange(2,20),rng.randrange(2,12)),fill=255).rotate(rng.uniform(-5,5),resample=Image.Resampling.BILINEAR,expand=True,fillcolor=255)
                image=ImageEnhance.Brightness(image).enhance(rng.uniform(.9,1.1))
            try:x,g,_=segment(image);variants.append((torch.from_numpy(x),torch.from_numpy(g)))
            except ValueError:
                if not view:raise
        cached.append({'row':r,'views':variants,'target':torch.tensor([1]+[TOKENS.index(t) for t in r['tokens']]+[2])})
        if (i+1)%1000==0:print('cached images',i+1,flush=True)
    out={'signature':signature,'items':cached};torch.save(out,path);return out

def cached_batch(items,device,augment=False):
    xs=[];gs=[];lengths=[]
    for item in items:
        x,g=random.choice(item['views']) if augment else item['views'][0];xs.append(x);gs.append(g);lengths.append(len(x))
    return torch.cat(xs).to(device),torch.cat(gs).to(device),lengths,nn.utils.rnn.pad_sequence([r['target'] for r in items],batch_first=True).to(device)

@torch.inference_mode()
def validation(model,items,device):
    model.eval();correct=raw=dist=length=invalid=0
    for offset in range(0,len(items),32):
        group=items[offset:offset+32];x,g,l,t=cached_batch(group,device);state=model.encode(x,g,l);prev=torch.ones(len(group),1,dtype=torch.long,device=device);pred=[[] for _ in group];done=[False]*len(group)
        for _ in range(64):
            z,state=model.decoder(model.embedding(prev),state);scores=model.output(z);scores[:,:,[0,1,3]]=-torch.inf;prev=scores.argmax(-1)
            for j,k in enumerate(prev[:,0].tolist()):
                if done[j]:continue
                if k==2:done[j]=True
                else:pred[j].append(TOKENS[k])
            if all(done):break
        for j,item in enumerate(group):
            truth,identity=normalize(item['row']['tokens']);length+=len(truth);raw+=done[j] and pred[j]==item['row']['tokens']
            try:ts,pi=normalize(pred[j]);correct+=bool(done[j] and pi==identity)
            except (ValueError,RecursionError):ts=pred[j];invalid+=1
            dist+=distance(ts,truth)
    return {'n':len(items),'exact_match':correct/len(items),'raw_exact_match':raw/len(items),'token_error_rate':dist/length,'syntax_error_rate':invalid/len(items)}

def train(real,synthetic,photos=None):
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';m=Model().to(device)
    cache=prepare_cache(real,synthetic);items=cache['items'];val=[r for r in items if r['row']['split']=='val'];trainitems=[r for r in items if r['row']['split']=='train']
    strata=[{k:[r for r in trainitems if (r['row']['source']=='MathWriting human ink')==isreal and r['row']['kind']==k] for k in ['basic','sup','sub','fraction']} for isreal in [True,False]]
    if any(not group for st in strata for group in st.values()):raise ValueError('Missing source/type stratum')
    aux=np.load(ROOT/'auxiliary_glyphs.npz');ax=torch.tensor(aux['images'],device=device);ay=torch.tensor(aux['labels'],device=device);ban=json.loads((ROOT/'symbol_parent_audit.json').read_text())['excluded_auxiliary_indices'];allowed=torch.ones(len(ay),dtype=torch.bool,device=device);allowed[ban]=False;pools=[torch.where((ay==k)&allowed)[0] for k in range(70)];sizes=torch.tensor([len(p) for p in pools],device=device)
    if int(sizes.min())==0:raise ValueError('Empty auxiliary class after source audit')
    pm=torch.zeros((70,int(sizes.max())),dtype=torch.long,device=device)
    for k,p in enumerate(pools):pm[k,:len(p)]=p
    def glyphids(n):
        cls=torch.randint(70,(n,),device=device);off=(torch.rand(n,device=device)*sizes[cls]).long();return pm[cls,off]
    def glyphaugment(x):
        n=len(x);theta=torch.eye(2,3,device=device).repeat(n,1,1);a=torch.empty(n,device=device).uniform_(-5,5)*np.pi/180;s=torch.empty(n,device=device).uniform_(.94,1.06);theta[:,0,0]=a.cos()*s;theta[:,1,1]=a.cos()*s;theta[:,0,1]=-a.sin()*s;theta[:,1,0]=a.sin()*s;theta[:,:,2]=torch.empty(n,2,device=device).uniform_(-.08,.08)
        return nn.functional.grid_sample(x,nn.functional.affine_grid(theta,x.shape,align_corners=False),align_corners=False)
    lastpath=ROOT/'last_model.pt';resume=torch.load(lastpath,map_location=device,weights_only=False) if lastpath.exists() else None
    if resume and resume['input_signature']!=cache['signature']:raise ValueError('Resume data/code signature changed')
    if not resume:
        opt=torch.optim.Adam(m.cnn.parameters(),lr=.001)
        for epoch in range(10):
            m.train();order=torch.cat([p[torch.randint(len(p),(100,),device=device)] for p in pools]);order=order[torch.randperm(len(order),device=device)]
            for ids in order.split(64):opt.zero_grad();loss=nn.functional.cross_entropy(m.cnn(glyphaugment(ax[ids])),ay[ids]);loss.backward();opt.step()
            print('CNN initialization',epoch+1,float(loss.detach()),flush=True)
    opt=torch.optim.Adam([{'params':m.cnn.parameters(),'lr':.0001},{'params':[p for name,p in m.named_parameters() if not name.startswith('cnn.')],'lr':.001}]);best=(-1,float('-inf'));history=[];wait=0;start=time.time();first=0
    if resume:
        m.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);random.setstate(resume['python_rng']);np.random.set_state(resume['numpy_rng']);torch.set_rng_state(resume['torch_rng'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([t.cpu() for t in resume['cuda_rng']])
        history=json.loads((ROOT/'training_history.json').read_text());best=max(tuple(x['selection']) for x in history);first=resume['epoch'];wait=resume['wait']
    save('training_environment.json',{'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'torch':torch.__version__,'parameters':sum(p.numel() for p in m.parameters()),'input_signature':cache['signature'],'auxiliary_allowed':int(allowed.sum()),'smallest_class_originals':int(sizes.min()),'cache_note':'Cached pixel crops and geometry, NOT detached CNN features; two full-image segmentation views per training image.'})
    for epoch in range(first,60):
        m.train();warm=epoch<3
        for p in m.cnn.parameters():p.requires_grad_(not warm)
        losses=[];epochstart=time.time()
        for step in range(600):
            chosen=[]
            for st,count in zip(strata,[7,3]):
                for _ in range(count):chosen.append(random.choice(st[random.choice(list(st))]))
            x,g,l,t=cached_batch(chosen,device,True);opt.zero_grad();z=m(x,g,l,t[:,:-1]);ce=nn.functional.cross_entropy(z.transpose(1,2),t[:,1:],ignore_index=0,reduction='none');mask=t[:,1:]!=0;loss=((ce*mask).sum(1)/mask.sum(1)).mean()
            if not warm:
                ids=glyphids(64);loss=loss+.3*nn.functional.cross_entropy(m.cnn(glyphaugment(ax[ids])),ay[ids])
            loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();losses.append(float(loss.detach()))
        mv=validation(m,val,device);score=(mv['exact_match'],-mv['token_error_rate']);entry={'epoch':epoch+1,'loss':float(np.mean(losses)),'mathwriting_validation':mv,'selection':score,'seconds':time.time()-epochstart};history.append(entry);save('training_history.json',history);print(entry,flush=True)
        improved=score>best
        if improved:best=score;wait=0
        else:wait+=1
        ck={'model':m.state_dict(),'config':CFG,'epoch':epoch+1,'optimizer':opt.state_dict(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else [],'selection':score,'wait':wait,'input_signature':cache['signature']}
        torch.save(ck,lastpath)
        if improved:torch.save(ck,ROOT/'best_model.pt')
        save('state.json',{'state':'TRAINING','complete':False,'epoch':epoch+1,'best_validation_exact_match':best[0],'training_started':True,'photographs_used':False})
        if epoch+1>=15 and wait>=8:break
    selected=torch.load(ROOT/'best_model.pt',map_location=device,weights_only=False);m.load_state_dict(selected['model']);torch.save(selected,ROOT/'final_model.pt');save('frozen_selection.json',{'epoch':selected['epoch'],'selection':selected['selection'],'test_used_for_selection':False,'input_signature':cache['signature']});save('state.json',{'state':'TESTING','complete':False,'training_seconds_this_session':time.time()-start,'training_started':True})
    test=evaluate(m,[r for r in real if r['split']=='test'],True);save('mathwriting_test.json',test);save('state.json',{'state':'REPORTING','complete':False,'primary_test_n':test['n'],'primary_exact_match':test['exact_match'],'training_started':True,'test_predictions_generated':True})
    return m
