"""Evaluate selected physics and legacy packages on exactly the same images."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

import train
from model import FormulaOCR,ModelConfig
from physics_syntax import GREEK,canonical,parse
from prepare_data import preprocess
from decoding import generate

ROOT=Path(__file__).resolve().parent


def metrics(rows):
    if not rows:return {'n':0}
    return {'n':len(rows),'exact':sum(r['exact'] for r in rows)/len(rows),
            'semantic_exact':sum(r['semantic_exact'] for r in rows)/len(rows),
            'token_error_rate':sum(r['edit_distance'] for r in rows)/sum(len(r['target_tokens']) for r in rows),
            'syntax_valid':sum(r['syntax_valid'] for r in rows)/len(rows),
            'eos_rate':sum(r['ended'] for r in rows)/len(rows)}


class Images(train.FormulaDataset):
    def __init__(self,records,vocab,root,max_width,preprocessing='default',height=64):
        self.records=records;self.vocab=vocab;self.lookup={t:i for i,t in enumerate(vocab)};self.augment=False
        self.images=[]
        if preprocessing=='handwriting-v1':
            from handwriting_data import preprocess_handwriting
            processor=preprocess_handwriting
        else:processor=preprocess
        for row in records:
            with Image.open(root/row['image']) as image:
                # Treat the saved source as an upload: use exactly the UI pipeline,
                # including illumination correction and recropping, for every model.
                image=processor(image,height=height,max_width=max_width)
                self.images.append(np.array(image,dtype=np.uint8))


@torch.inference_mode()
def evaluate(model,package,records,root,device,blank=False,batch_size=32):
    data=Images(records,package['vocab'],root,package.get('max_width',384),
                package.get('preprocessing','default'),package['config']['height'])
    loader=DataLoader(data,batch_size=batch_size,collate_fn=train.collate)
    predictions=[];start=time.perf_counter()
    for images,widths,_,indexes in loader:
        images=images.to(device)
        if blank:images.zero_()
        generated=generate(model,package,images,widths).cpu().tolist()
        for index,ids in zip(indexes,generated):
            target=records[index];ended=2 in ids
            ids=ids[:ids.index(2)] if ended else ids;tokens=[package['vocab'][i] for i in ids]
            try:
                tree=parse(tokens);valid=True
                semantic=ended and canonical(tree)==canonical(parse(target['tokens']))
            except (ValueError,RecursionError):valid=False;semantic=False
            distance=train.edit_distance(target['tokens'],tokens)
            predictions.append({'image':target['image'],'target_tokens':target['tokens'],'prediction_tokens':tokens,
                                'target':' '.join(target['tokens']),'prediction':' '.join(tokens),
                                'exact':ended and tokens==target['tokens'],'semantic_exact':bool(semantic),
                                'syntax_valid':valid,'ended':ended,'edit_distance':distance,
                                'family':target.get('family'),'font':target.get('font'),
                                'layout':target.get('layout'),'law_id':target.get('law_id')})
    result=metrics(predictions);result['seconds']=round(time.perf_counter()-start,3)
    result['greek']=metrics([r for r in predictions if any(t.startswith('\\') and t[1:] in GREEK for t in r['target_tokens'])])
    result['groups']={family:metrics([r for r in predictions if r['family']==family]) for family in sorted({r['family'] for r in predictions})}
    result['layouts']={layout:metrics([r for r in predictions if r['layout']==layout]) for layout in sorted({r['layout'] for r in predictions if r['layout']})}
    return result,predictions


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,default=ROOT/'runs/physics_v1/best.pt');args=ap.parse_args()
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=ROOT/'physics';out.mkdir(exist_ok=True);results={}
    for label,path in [('legacy',ROOT/'comparison/best_model.pt'),('physics',args.candidate)]:
        package=torch.load(path,map_location='cpu',weights_only=False)
        model=FormulaOCR(ModelConfig(**package['config']));model.load_state_dict(package['model']);model.to(device).eval()
        results[label]={'checkpoint':str(path),'epoch':package.get('epoch'),'parameters':sum(p.numel() for p in model.parameters())}
        for split in ['val','test','test_font','test_capture','real_test','physics_laws','legacy_test']:
            folder=ROOT/('comparison/views/D_slash_slash' if split=='legacy_test' else 'data_physics')
            file=folder/('test.jsonl' if split=='legacy_test' else split+'.jsonl')
            records=[json.loads(line) for line in file.read_text(encoding='utf-8').splitlines()]
            score,predictions=evaluate(model,package,records,folder,device)
            results[label][split]=score
            (out/f'{label}_{split}.json').write_text(json.dumps(predictions,ensure_ascii=False,indent=2),encoding='utf-8')
            print(f'{label} {split}: semantic={score["semantic_exact"]:.4f} token_exact={score["exact"]:.4f}',flush=True)
        records=[json.loads(line) for line in (ROOT/'data_physics/test.jsonl').read_text(encoding='utf-8').splitlines()][:128]
        results[label]['blank_control'],_=evaluate(model,package,records,ROOT/'data_physics',device,True)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    (out/'metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Comparison saved to physics/metrics.json',flush=True)


if __name__=='__main__':main()
