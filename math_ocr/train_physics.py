"""Warm-start the CNN/BiLSTM/LSTM with a remapped physics vocabulary.

Token cross-entropy, full-image fixed context, no attention and no CTC.
Checkpoint selection uses validation only; physics laws stay sealed for evaluation.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Sampler

import train
from augmentation import PracticalDataset
from model import FormulaOCR,ModelConfig
from physics_syntax import parse,canonical

ROOT=Path(__file__).resolve().parent


def semantic_metrics(rows):
    correct=0
    for row in rows:
        try:ok=row['ended'] and canonical(parse(row['target_tokens']))==canonical(parse(row['predicted_tokens']))
        except (ValueError,RecursionError):ok=False
        row['semantic_exact']=bool(ok);correct+=ok
    return correct/max(1,len(rows))


class WidthBatches(Sampler):
    def __init__(self,records,batch_size):self.records=records;self.batch_size=batch_size
    def __len__(self):return math.ceil(len(self.records)/self.batch_size)
    def __iter__(self):
        indexes=list(range(len(self.records)));random.shuffle(indexes);batches=[]
        for start in range(0,len(indexes),1024):
            group=sorted(indexes[start:start+1024],key=lambda i:self.records[i]['width'])
            batches += [group[j:j+self.batch_size] for j in range(0,len(group),self.batch_size)]
        random.shuffle(batches)
        yield from batches


def initialize(old,vocab,auxiliary=False,local_features=0,raw_context=False,refinement_blocks=0):
    previous=ModelConfig(**old['config'])
    config=replace(previous,vocab_size=len(vocab),context_bins=8,auxiliary=auxiliary,local_features=local_features,raw_context=raw_context,refinement_blocks=refinement_blocks)
    model=FormulaOCR(config);weights=model.state_dict();lookup={t:i for i,t in enumerate(old['vocab'])}
    for name,value in old['model'].items():
        if name not in weights:continue
        if name not in ['embedding.weight','output.weight','output.bias']:
            if weights[name].shape==value.shape:weights[name].copy_(value.cpu())
            else:
                weights[name].zero_()
                if name=='decoder.weight_ih_l0' and previous.local_features:
                    old_context=previous.encoder_hidden*2*(2 if previous.context_bins else 1)+(previous.encoder_hidden*2*previous.context_bins if previous.raw_context else 0)
                    new_context=config.encoder_hidden*2*2+(config.encoder_hidden*2*config.context_bins if config.raw_context else 0)
                    count=previous.embedding_size+old_context
                    weights[name][:,:count].copy_(value[:,:count].cpu())
                    start=config.embedding_size+new_context
                    weights[name][:,start:start+previous.local_features].copy_(value[:,count:].cpu())
                else:weights[name][:,:value.shape[1]].copy_(value.cpu())
    for i,token in enumerate(vocab):
        if token in lookup:
            j=lookup[token]
            for name in ['embedding.weight','output.weight','output.bias']:weights[name][i].copy_(old['model'][name][j].cpu())
    model.load_state_dict(weights);return model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=50);ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--lr',type=float,default=.0008);ap.add_argument('--run',default='physics_v1');ap.add_argument('--resume',type=Path)
    ap.add_argument('--init',type=Path,default=ROOT/'comparison/best_model.pt');ap.add_argument('--no-amp',action='store_true')
    ap.add_argument('--data-dir',type=Path,default=ROOT/'data_physics');ap.add_argument('--handwriting',action='store_true')
    ap.add_argument('--local-features',type=int,default=0);ap.add_argument('--balanced-ce',action='store_true')
    ap.add_argument('--raw-context',action='store_true');ap.add_argument('--refinement-blocks',type=int,default=0);args=ap.parse_args()
    random.seed(731);np.random.seed(731);torch.manual_seed(731);torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.manual_seed_all(731);torch.backends.cudnn.benchmark=True
    train.DATA=args.data_dir.resolve();vocab=json.loads((train.DATA/'vocab.json').read_text(encoding='utf-8'))
    old=torch.load(args.resume or args.init,map_location='cpu',weights_only=False)
    model=FormulaOCR(ModelConfig(**old['config'])) if args.resume else initialize(old,vocab,auxiliary=args.handwriting,local_features=args.local_features,raw_context=args.raw_context,refinement_blocks=args.refinement_blocks)
    if args.resume:model.load_state_dict(old['model']);assert old['vocab']==vocab
    model.to(device)
    rows=train.read_records('train');dataset=PracticalDataset(rows,vocab,preserve_bounds=args.handwriting,max_width=512)
    loader=DataLoader(dataset,batch_sampler=WidthBatches(rows,args.batch_size),collate_fn=train.collate,num_workers=0,pin_memory=device.type=='cuda')
    splits=['val','val_greek','val_capture','val_print'] if args.handwriting else ['val','val_font','val_capture','real_val']
    validators={s:DataLoader(train.FormulaDataset(train.read_records(s)[:(500 if s=='val' else 200)],vocab),batch_size=32,collate_fn=train.collate) for s in splits}
    if args.handwriting:
        slow=[];fast=[]
        for name,param in model.named_parameters():
            prefixes=('embedding.','output.') if args.local_features else ('decoder.','embedding.','output.')
            (slow if name.startswith(prefixes) else fast).append(param)
        optimizer=torch.optim.AdamW([{'params':fast,'lr_scale':1.},{'params':slow,'lr_scale':.5}],lr=args.lr,weight_decay=.0001)
    else:optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.0001)
    amp=device.type=='cuda' and not args.no_amp
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    criterion=nn.CrossEntropyLoss(ignore_index=0,label_smoothing=.02,reduction='none' if args.balanced_ce else 'mean')
    def token_loss(logits,target):
        values=criterion(logits.flatten(0,1),target.flatten())
        if args.balanced_ce:return (values.reshape(target.shape).sum(1)/target.ne(0).sum(1).clamp(min=1)).mean()
        return values
    out=ROOT/'runs'/args.run;out.mkdir(parents=True,exist_ok=True)
    metadata={'attention':False,'ctc':False,'seed':731,'data':str(train.DATA),'train_samples':len(rows),
              'parameters':sum(p.numel() for p in model.parameters()),'initial_package':str(args.init),
              'selection':('weighted semantic exact: val .5 / greek .25 / capture .15 / print .1' if args.handwriting else 'mean semantic exact on val(500), val_font(200), val_capture(200), real_val(200)'),
              'amp':amp,'max_decode':72,'max_width':512,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    metadata['objective']='token CE + .15 symbol-presence BCE + .03 length CE' if model.config.auxiliary else 'token CE'
    metadata['sequence_balanced_ce']=args.balanced_ce
    (out/'config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata),flush=True)
    history=old.get('history',[]) if args.resume else [];start_epoch=old['epoch']+1 if args.resume else 1
    best=max((x['score'] for x in history),default=-1);best_epoch=max((x['epoch'] for x in history if x['score']==best),default=0)
    if args.resume:
        optimizer.load_state_dict(old['optimizer']);scaler.load_state_dict(old['scaler'])
        if 'rng_torch' in old:
            random.setstate(old['rng_python']);np.random.set_state(old['rng_numpy']);torch.set_rng_state(old['rng_torch'])
            if device.type=='cuda':torch.cuda.set_rng_state_all(old['rng_cuda'])
    started=time.perf_counter()
    for epoch in range(start_epoch,args.epochs+1):
        began=time.perf_counter();model.train();total=0.;count=0
        lr=args.lr*(.15+.85*.5*(1+math.cos(math.pi*(epoch-1)/args.epochs)))
        for group in optimizer.param_groups:group['lr']=lr*group.get('lr_scale',1.)
        for step,(images,widths,tokens,_) in enumerate(loader,1):
            images=images.to(device,non_blocking=True);tokens=tokens.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                if model.config.auxiliary:
                    logits,presence,length=model.forward_with_aux(images,widths,tokens[:,:-1])
                    label_loss=token_loss(logits,tokens[:,1:])
                    wanted=torch.zeros_like(presence);wanted.scatter_(1,tokens,1.)
                    auxiliary=torch.nn.functional.binary_cross_entropy_with_logits(presence[:,4:],wanted[:,4:],pos_weight=torch.full_like(presence[0,4:],4.))
                    length_loss=torch.nn.functional.cross_entropy(length,(tokens[:,1:].ne(0).sum(1)-1).clamp(0,72))
                    loss=label_loss+.15*auxiliary+.03*length_loss
                else:
                    logits=model(images,widths,tokens[:,:-1]);label_loss=token_loss(logits,tokens[:,1:]);loss=label_loss
            if not torch.isfinite(loss):raise ValueError('Non-finite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optimizer);scaler.update()
            n=len(tokens) if args.balanced_ce else int(tokens[:,1:].ne(0).sum());total+=float(label_loss.detach())*n;count+=n
            if step%75==0:print(f'epoch {epoch} step {step}/{len(loader)} CE {total/count:.4f}',flush=True)
        scores={};exact={};validation_rows={}
        for split,validation in validators.items():
            metrics,predictions=train.evaluate(model,validation,device,max_tokens=72)
            scores[split]=semantic_metrics(predictions);exact[split]=metrics['exact_match']
            validation_rows[split]=predictions
        score=(.5*scores['val']+.25*scores['val_greek']+.15*scores['val_capture']+.1*scores['val_print']) if args.handwriting else float(np.mean(list(scores.values())))
        entry={'epoch':epoch,'train_ce':total/count,'score':score,'semantic':scores,'exact':exact,
               'seconds':round(time.perf_counter()-began,2),'lr':lr}
        history.append(entry)
        state={'model_type':'seq2seq','model':model.state_dict(),'config':model.config_dict(),'vocab':vocab,
               'input_notation':'mixed','output_notation':'latex','syntax':'physics-v1','max_width':512,'max_decode':72,
               'selected_variant':('Handwriting-deep-v1' if args.raw_context else 'Handwriting-local-v1' if args.local_features else 'Handwriting-v1') if args.handwriting else 'Physics-v1','metadata':metadata,'epoch':epoch,'history':history,
               'preprocessing':'handwriting-v1' if args.handwriting else 'default',
               'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
               'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state()}
        if device.type=='cuda':state['rng_cuda']=torch.cuda.get_rng_state_all()
        torch.save(state,out/'last.pt')
        if score>best:
            best=score;best_epoch=epoch;torch.save(state,out/'best.pt')
            for split,predictions in validation_rows.items():
                (out/f'validation_{split}.json').write_text(json.dumps(predictions,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps(entry),flush=True)
        if epoch>=35 and epoch-best_epoch>=12:print('Early stop: validation plateau',flush=True);break
    print(f'Training finished in {time.perf_counter()-started:.1f}s; selected epoch {best_epoch}, validation {best:.4f}',flush=True)


if __name__=='__main__':main()
