"""Choose decoding on validation only, then export a weights-only model package."""
import argparse
import json
from pathlib import Path
import torch
from model import FormulaOCR,ModelConfig
from evaluate_physics import evaluate

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,default=ROOT/'runs/physics_v1/best.pt')
    ap.add_argument('--data-dir',type=Path,default=ROOT/'data_physics');ap.add_argument('--handwriting',action='store_true')
    ap.add_argument('--selection-json',type=Path,help='Reuse joint architecture/decoder validation from the chooser')
    args=ap.parse_args()
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source=torch.load(args.candidate,map_location='cpu',weights_only=False)
    model=FormulaOCR(ModelConfig(**source['config']));model.load_state_dict(source['model']);model.to(device).eval()
    splits=['val','val_greek','val_capture','val_print'] if args.handwriting else ['val','val_font','val_capture','real_val']
    validators={s:[json.loads(x) for x in (args.data_dir/f'{s}.jsonl').read_text(encoding='utf-8').splitlines()][:200] for s in splits}
    trials=[]
    if args.selection_json:
        selection=json.loads(args.selection_json.read_text(encoding='utf-8'))
        chosen=selection['candidates'][selection['selected']]
        assert Path(chosen['checkpoint']).resolve()==args.candidate.resolve()
        trials=chosen['decoder_trials']
    else:
        for beam,alpha in [(1,0),(3,.6),(5,.6)]:
            package={**source,'beam_width':beam,'length_alpha':alpha};scores={}
            for split,records in validators.items():
                score,_=evaluate(model,package,records,args.data_dir,device,batch_size=1 if args.handwriting else 32);scores[split]=score['semantic_exact']
            value=(.5*scores['val']+.25*scores['val_greek']+.15*scores['val_capture']+.1*scores['val_print']) if args.handwriting else sum(scores.values())/len(scores)
            trial={'beam_width':beam,'length_alpha':alpha,'semantic':scores,'score':value}
            trials.append(trial);print(json.dumps(trial),flush=True)
    # Smaller beams win a tie to reduce latency. Test/law data never enters here.
    best=max(trials,key=lambda row:(row['score'],-row['beam_width']))
    package={key:source[key] for key in ['model_type','config','vocab','syntax','input_notation','output_notation','max_width','max_decode','selected_variant','epoch']}
    package['model']={key:value.detach().cpu() for key,value in source['model'].items()}
    package.update({'beam_width':best['beam_width'],'length_alpha':best['length_alpha'],
                    'source_checkpoint':str(args.candidate),'decoder_validation':best,
                    'scope':'Filtered 1-32 token MathWriting handwriting and short physics notation.' if args.handwriting else 'Printed introductory physics notation; no handwriting training.',
                    'preprocessing':source.get('preprocessing','default'),
                    'attention':False,'ctc':False})
    if args.handwriting:
        summary=json.loads((args.data_dir/'summary.json').read_text(encoding='utf-8'))
        package['training_data']={key:summary[key] for key in ['source','license','train_family_counts','max_tokens','official_partitions_preserved']}
        records=[json.loads(x) for x in (args.data_dir/'val.jsonl').read_text(encoding='utf-8').splitlines()]
        _,predictions=evaluate(model,package,records,args.data_dir,device,batch_size=1)
        from physics_syntax import GREEK
        tests=[('그리스 문자','αβ',lambda r:any(t.startswith('\\') and t[1:] in GREEK for t in r['tokens'])),
               ('첨자','xⁿ',lambda r:'^' in r['tokens'] or '_' in r['tokens']),
               ('분수','a / b',lambda r:'\\frac' in r['tokens'])]
        demos=[]
        for label,symbol,test in tests:
            candidates=[r for r,p in zip(records,predictions) if test(r) and p['semantic_exact']]
            if not candidates:candidates=[r for r in records if test(r)]
            chosen=min(candidates,key=lambda r:len(r['tokens']))
            demos.append({'ink_id':chosen['ink_id'],'label':label,'symbol':symbol})
        package['demo_samples']=demos
    out=ROOT/'physics';out.mkdir(exist_ok=True)
    torch.save(package,out/'best_model.pt.tmp');(out/'best_model.pt.tmp').replace(out/'best_model.pt')
    (out/'decoder_selection.json').write_text(json.dumps({'trials':trials,'selected':best},indent=2),encoding='utf-8')
    print(f'Exported {out / "best_model.pt"}',flush=True)


if __name__=='__main__':main()
