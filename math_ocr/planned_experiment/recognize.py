"""Image-only inference. No ground truth or manifest is accessed here."""
import argparse,json,time
import numpy as np
from PIL import Image
import torch
import run as experiment

class Recognizer:
    def __init__(self,seed=42,device='cpu'):
        self.device=device
        sel=json.loads((experiment.ROOT/'frozen_selection.json').read_text());self.profile=sel['profile']
        c=torch.load(experiment.ROOT/f"{sel['cnn']}_{seed}.pt",map_location=device,weights_only=False)
        self.cnn=experiment.Classifier().to(device);self.cnn.load_state_dict(c['state']);self.cnn.eval()
        s=torch.load(experiment.ROOT/f"{sel['lstm']}_{seed}.pt",map_location=device,weights_only=False)
        self.lstm=experiment.Seq(s['position']).to(device);self.lstm.load_state_dict(s['state']);self.lstm.eval()

    @torch.inference_mode()
    def __call__(self,path):
        start=time.perf_counter();im=Image.open(path).convert('L');t=time.perf_counter();bs=experiment.components(im,True,self.profile);t2=time.perf_counter()
        if not bs:return {'tokens':[],'latex':'','boxes':[],'error':'no components','seconds':time.perf_counter()-start}
        p=self.cnn(torch.tensor(np.stack([experiment.crop(im,b) for b in bs]),device=self.device)).softmax(-1).cpu().numpy();b=np.asarray(bs,np.float32)
        geo=np.column_stack(((b[:,0]+b[:,2])/2/im.width,(b[:,1]+b[:,3])/2/im.height,(b[:,2]-b[:,0])/im.width,(b[:,3]-b[:,1])/im.height)).astype('float32')
        x=torch.tensor(np.concatenate((p,geo),1)[None],device=self.device);t3=time.perf_counter();state=self.lstm.encode(x,torch.tensor([len(bs)]));prev=torch.ones(1,1,dtype=torch.long,device=self.device);tokens=[];terminated=False
        for _ in range(64):
            z,state=self.lstm.dec(self.lstm.emb(prev),state);prev=self.lstm.out(z).argmax(-1);k=prev.item()
            if k==2:terminated=True;break
            tokens.append(experiment.VOCAB[k])
        end=time.perf_counter()
        return {'tokens':tokens,'latex':' '.join(tokens),'boxes':bs,'terminated':terminated,'seconds':end-start,'stage_seconds':{'read':t-start,'segment':t2-t,'classify':t3-t2,'decode':end-t3}}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('image');args=ap.parse_args();torch.set_num_threads(1);print(json.dumps(Recognizer()(args.image),ensure_ascii=False,indent=2))
