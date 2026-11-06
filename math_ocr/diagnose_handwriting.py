"""Post-test diagnostics; never changes checkpoint or decoder selection."""
import json
from pathlib import Path
import torch
from model import FormulaOCR,ModelConfig
from evaluate_physics import evaluate,metrics
from physics_syntax import canonical,parse

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data_handwriting';OUT=ROOT/'physics'


def identity(tokens):return repr(canonical(parse(tokens)))


def records(split):return [json.loads(x) for x in (DATA/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]


def main():
    torch.set_num_threads(4)
    package=torch.load(OUT/'best_model.pt',map_location='cpu',weights_only=False)
    original=torch.load(package['source_checkpoint'],map_location='cpu',weights_only=False)
    identical=package['model'].keys()==original['model'].keys() and all(torch.equal(w,original['model'][k]) for k,w in package['model'].items())
    if not identical:raise AssertionError('Exported weights differ from the selected training checkpoint')
    del original
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=FormulaOCR(ModelConfig(**package['config']));model.load_state_dict(package['model']);model.to(device).eval()
    validation=records('val')
    score,predictions=evaluate(model,package,validation,DATA,device,batch_size=1)
    train_ids={identity(r['tokens']) for r in records('train')}
    result={'post_test_diagnostic_only':True,'changes_selection':False,'exported_weights_identical':identical,
            'full_validation':score,'splits':{}}
    for split,rows,predicted in [('val',validation,predictions),('test',records('test'),json.loads((OUT/'handwriting_test.json').read_text(encoding='utf-8')))]:
        seen=[identity(r['tokens']) in train_ids for r in rows]
        result['splits'][split]={'n':len(rows),'mean_tokens':sum(len(r['tokens']) for r in rows)/len(rows),
                                'formulas_seen_in_training':sum(seen),'formula_overlap_fraction':sum(seen)/len(rows),
                                'seen_formula':metrics([p for p,s in zip(predicted,seen) if s]),
                                'new_formula':metrics([p for p,s in zip(predicted,seen) if not s])}
    (OUT/'generalization_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
