"""Fine-tune the strongest slash CNN/BiLSTM/LSTM using practical augmentation.

Uses only CNN, LSTM and token cross-entropy; no CTC or attention.
"""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import train
from augmentation import PracticalDataset
from model import FormulaOCR,ModelConfig


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=24)
    parser.add_argument("--run",default="slash_practical")
    parser.add_argument("--data-dir",type=Path,default=Path("comparison/views/P_practical"))
    parser.add_argument("--init",type=Path,default=Path("runs/slash_D/best.pt"))
    parser.add_argument("--lr",type=float,default=.0003)
    args=parser.parse_args()
    train.DATA=args.data_dir.resolve()
    random.seed(44);np.random.seed(44);torch.manual_seed(44);torch.set_num_threads(4)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":torch.cuda.manual_seed_all(44);torch.backends.cudnn.benchmark=True
    checkpoint=torch.load(args.init,map_location=device,weights_only=False)
    vocab=json.loads((train.DATA/"vocab.json").read_text())
    assert vocab==checkpoint["vocab"]
    model=FormulaOCR(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    def loader(split,training=False):
        records=train.read_records(split)
        ds=PracticalDataset(records,vocab) if training else train.FormulaDataset(records,vocab)
        return DataLoader(ds,batch_size=32,shuffle=training,collate_fn=train.collate,num_workers=0,pin_memory=device.type=="cuda")
    training=loader("train",True)
    validators={s:loader(s) for s in ["val","val_font","val_capture"]}
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.0001)
    criterion=nn.CrossEntropyLoss(ignore_index=0)
    output=train.ROOT/"runs"/args.run;output.mkdir(parents=True,exist_ok=True)
    metadata={"model_type":"seq2seq","attention":False,"ctc":False,"config":checkpoint["config"],
        "parameters":sum(p.numel() for p in model.parameters()),"train_samples":len(training.dataset),
        "val_samples":300,"seed":44,"initial_checkpoint":str(args.init),
        "training_fonts":["stix","dejavusans","cm"],"added_nested_formulas":1000,
        "selection":"mean validation original/font/capture exact", "args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    (output/"config.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    history=[];best_key=(-1,-1);started=time.perf_counter()
    for epoch in range(args.epochs+1):
        start=time.perf_counter();total=0.;count=0
        if epoch:
            model.train()
            lr=args.lr*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/args.epochs)))
            for group in optimizer.param_groups:group["lr"]=lr
            for images,widths,tokens,_ in training:
                images,tokens=images.to(device),tokens.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(images,widths,tokens[:,:-1])
                loss=criterion(logits.flatten(0,1),tokens[:,1:].flatten())
                if not torch.isfinite(loss):raise ValueError("Non-finite cross-entropy")
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                n=int(tokens[:,1:].ne(0).sum());total+=float(loss.detach())*n;count+=n
        scores={}
        for split,validation in validators.items():
            metric,_=train.evaluate(model,validation,device,max_tokens=48)
            scores[split]=metric["exact_match"]
        robust=float(np.mean(list(scores.values())))
        entry={"epoch":epoch,"train_ce":total/count if count else None,**scores,
            "val_exact":scores["val"],"robust_val_exact":robust,"seconds":time.perf_counter()-start}
        history.append(entry)
        key=(robust,scores["val"])
        package={"model_type":"seq2seq","model":model.state_dict(),"config":checkpoint["config"],
            "vocab":vocab,"metadata":metadata,"epoch":epoch,"history":history,"optimizer":optimizer.state_dict()}
        torch.save(package,output/"last.pt")
        if key>best_key:best_key=key;torch.save(package,output/"best.pt")
        (output/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        print(json.dumps(entry),flush=True)
    selected=torch.load(output/"best.pt",map_location=device,weights_only=False);model.load_state_dict(selected["model"])
    results={"model":metadata,"checkpoint_epoch":selected["epoch"],"training_seconds":time.perf_counter()-started}
    for split in ["val","val_font","val_capture","test","test_font","test_capture","long_test"]:
        metrics,rows=train.evaluate(model,loader(split),device,max_tokens=48)
        results[split]=metrics
        (output/f"predictions_{split}.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        print(f"FINAL {split}: exact={metrics['exact_match']:.4f}",flush=True)
    (output/"metrics.json").write_text(json.dumps(results,indent=2),encoding="utf-8")


if __name__=="__main__":main()
