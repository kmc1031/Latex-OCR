"""Choose an architecture by validation only, then tune its decoder."""
import json
from pathlib import Path
import subprocess
import sys
import torch
from model import FormulaOCR,ModelConfig
from evaluate_physics import evaluate

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=ROOT/'data_handwriting'
    validators={s:[json.loads(x) for x in (data/f'{s}.jsonl').read_text(encoding='utf-8').splitlines()][:200]
                for s in ['val','val_greek','val_capture','val_print']}
    candidates={}
    for name in ['physics_handwriting_v1','physics_handwriting_local','physics_handwriting_deep']:
        folder=ROOT/'runs'/name
        history=json.loads((folder/'history.json').read_text(encoding='utf-8'))
        best=max(history,key=lambda r:r['score'])
        source=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        model=FormulaOCR(ModelConfig(**source['config']));model.load_state_dict(source['model']);model.to(device).eval()
        trials=[]
        for beam,alpha in [(1,0),(3,.6),(5,.6)]:
            scores={}
            for split,records in validators.items():
                metrics,_=evaluate(model,{**source,'beam_width':beam,'length_alpha':alpha},records,data,device,batch_size=1)
                scores[split]=metrics['semantic_exact']
            score=.5*scores['val']+.25*scores['val_greek']+.15*scores['val_capture']+.1*scores['val_print']
            trial={'beam_width':beam,'length_alpha':alpha,'semantic':scores,'score':score}
            trials.append(trial);print(json.dumps({'candidate':name,**trial}),flush=True)
        selected=max(trials,key=lambda row:(row['score'],-row['beam_width']))
        candidates[name]={'validation_score':selected['score'],'semantic':selected['semantic'],'epoch':best['epoch'],
                          'beam_width':selected['beam_width'],'decoder_trials':trials,
                          'training_validation_score':best['score'],'checkpoint':str(folder/'best.pt'),
                          'parameters':sum(p.numel() for p in model.parameters()),'config':source['config']}
        print(json.dumps({'candidate':name,**candidates[name]}),flush=True)
        del model;torch.cuda.empty_cache()
    name=max(candidates,key=lambda k:candidates[k]['validation_score'])
    out=ROOT/'physics';out.mkdir(exist_ok=True)
    (out/'candidate_selection.json').write_text(json.dumps({'candidates':candidates,'selected':name,'selection_uses_test':False,
        'batch_size':1,'validation_samples_per_split':200,'beam_widths':[1,3,5]},indent=2),encoding='utf-8')
    print(json.dumps({'selected':name,**candidates[name]}),flush=True)
    subprocess.run([sys.executable,str(ROOT/'select_physics_decoder.py'),'--candidate',candidates[name]['checkpoint'],
                    '--data-dir',str(ROOT/'data_handwriting'),'--handwriting',
                    '--selection-json',str(out/'candidate_selection.json')],check=True)


if __name__=='__main__':main()
