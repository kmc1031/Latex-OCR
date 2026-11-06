"""Validation-only settings, then separated oracle/automatic test evaluation."""
from collections import Counter
import argparse,json,random,time
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from common import ROOT,iou
from pipeline import Pipeline,assemble,expression_input
from physics_syntax import parse,canonical,tokenize

def read(split):return [json.loads(s) for s in (ROOT/f'data/{split}.jsonl').read_text().splitlines()]
def correct(text,target):
    try:return canonical(parse(tokenize(text)))==canonical(parse(target))
    except (ValueError,RecursionError):return False

def matching(pred,true):
    if not pred or not true:return 0
    matrix=np.array([[iou(a,b) for b in true] for a in pred]);a,b=linear_sum_assignment(-matrix)
    return int(sum(matrix[x,y]>=.5 for x,y in zip(a,b)))

def selected_rows(limit=160):
    rows=read('val');random.Random(733).shuffle(rows);seen=Counter();selected=[]
    for row in rows:
        if seen[row['formula_identity']]>=3:continue
        seen[row['formula_identity']]+=1;selected.append(row)
        if len(selected)==limit:break
    return selected

@torch.inference_mode()
def select(pipeline):
    rows=selected_rows();trials=[];correct_examples={}
    for method,threshold in [('components',0),('detector',.15),('detector',.25),('detector',.4)]:
        if method=='detector' and pipeline.detector is None:continue
        cache=[];matched=ngt=npred=0
        for row in rows:
            image,gt=expression_input(row);boxes=pipeline.propose(image,method,threshold)
            labels,p,features=pipeline.classify(image,boxes);cache.append((row,boxes,labels,p,features))
            matched+=matching(boxes,gt);ngt+=len(gt);npred+=len(boxes)
        for script in [.35,.5,.65]:
            scores=[]
            for row,boxes,labels,p,features in cache:
                text=assemble(boxes,labels,p,features,pipeline.relations,pipeline.device,script)
                scores.append(correct(text,row['tokens']))
            trial={'method':method,'detector_threshold':threshold,'script_threshold':script,'n':len(rows),'expression_accuracy':float(np.mean(scores)),
                   'box_precision_iou50':matched/max(npred,1),'box_recall_iou50':matched/max(ngt,1)}
            trials.append(trial);print(json.dumps(trial),flush=True)
            correct_examples[(method,threshold,script)]=[r for r,ok in zip(rows,scores) if ok]
    best=max(trials,key=lambda r:(r['expression_accuracy'],r['box_recall_iou50'],r['box_precision_iou50']))
    examples=correct_examples[(best['method'],best['detector_threshold'],best['script_threshold'])]
    choices=sorted(examples or rows,key=lambda r:len(r['glyphs']))
    demos=[];used=set()
    for row in choices:
        if row['formula_identity'] in used:continue
        used.add(row['formula_identity']);demos.append(row['id'])
        if len(demos)==3:break
    while len(demos)<3:demos.append(choices[0]['id'])
    best['demos']=demos;best['demos_selected_from_correct_validation']=bool(examples)
    (ROOT/'models/settings.json').write_text(json.dumps(best,indent=2),encoding='utf-8');(ROOT/'models/validation_selection.json').write_text(json.dumps({'trials':trials,'selected':best,'uses_test':False,'selection':'max 3 images per unseen validation formula'},indent=2),encoding='utf-8')
    pipeline.settings=best

@torch.inference_mode()
def evaluate(pipeline,split='test'):
    rows=read(split);predictions=[];counts=Counter();glyph_good=glyph_total=top3=0;confusions=Counter();start=time.perf_counter()
    for k,row in enumerate(rows):
        image,boxes=expression_input(row);truth=[g['label'] for g in row['glyphs']]
        labels,p,features=pipeline.classify(image,boxes)
        for target,pred,probs in zip(truth,labels,p):
            glyph_total+=1;glyph_good+=target==pred;top3+=target in [pipeline.vocab[i] for i in np.argsort(probs)[-3:]]
            if target!=pred:confusions[(target,pred)]+=1
        modes={}
        for name,labs in [('oracle_boxes_and_labels',truth),('oracle_boxes',labels)]:
            text=assemble(boxes,labs,p,features,pipeline.relations,pipeline.device,pipeline.settings['script_threshold']);ok=correct(text,row['tokens']);counts[name]+=ok;modes[name]={'prediction':text,'correct':ok}
        for method in ['components','detector']:
            if method=='detector' and pipeline.detector is None:continue
            # The selected threshold is used for the detector, regardless of
            # whether components won selection. Fallback .25 is validation-free.
            threshold=pipeline.settings.get('detector_threshold') or .25
            result=pipeline.recognize(image,prepared=True,method=method,threshold=threshold);ok=correct(result['prediction'],row['tokens']);counts[method]+=ok
            modes[method]={'prediction':result['prediction'],'correct':ok,'boxes':result['boxes'],'labels':result['labels']}
            counts[method+'_matched']+=matching(result['boxes'],boxes);counts[method+'_predicted']+=len(result['boxes'])
        predictions.append({**{key:row[key] for key in ['id','image','tokens','latex','seen_formula']},'modes':modes})
        if (k+1)%100==0:print('Test processed',k+1,'/',len(rows),flush=True)
    metrics={'n':len(rows),'glyphs':{'n':glyph_total,'accuracy':glyph_good/glyph_total,'top3':top3/glyph_total},'modes':{},'selected_automatic_method':pipeline.settings['method'],'seconds':round(time.perf_counter()-start,2),
             'confusions':[{'target':a,'prediction':b,'count':n} for (a,b),n in confusions.most_common(20)]}
    for name in predictions[0]['modes']:
        subset=[r for r in predictions if not r['seen_formula']]
        metrics['modes'][name]={'accuracy':counts[name]/len(rows),'correct':counts[name],'new_formula_n':len(subset),'new_formula_accuracy':sum(r['modes'][name]['correct'] for r in subset)/max(len(subset),1)}
        if name in ['components','detector']:metrics['modes'][name].update({'box_precision_iou50':counts[name+'_matched']/max(counts[name+'_predicted'],1),'box_recall_iou50':counts[name+'_matched']/glyph_total})
    (ROOT/f'{split}_predictions.json').write_text(json.dumps(predictions,indent=2),encoding='utf-8');(ROOT/f'{split}_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');print(json.dumps(metrics),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--select',action='store_true');ap.add_argument('--probe',action='store_true');args=ap.parse_args();pipeline=Pipeline()
    if args.probe:
        good=[0,0];rows=selected_rows(100)
        for row in rows:
            image,boxes=expression_input(row);pred,p,f=pipeline.classify(image,boxes)
            for i,labs in enumerate([[g['label'] for g in row['glyphs']],pred]):good[i]+=correct(assemble(boxes,labs,p,f,pipeline.relations,pipeline.device),row['tokens'])
        print(json.dumps({'validation_only':True,'n':len(rows),'gt_labels_and_boxes':good[0]/len(rows),'predicted_labels_gt_boxes':good[1]/len(rows)}));return
    if args.select:select(pipeline)
    evaluate(pipeline)

if __name__=='__main__':main()
