"""Report actual handwriting generalization, including Greek and length groups."""
import argparse
import json
from pathlib import Path
import torch
from model import FormulaOCR,ModelConfig
from evaluate_physics import evaluate,metrics

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'physics'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,default=OUT/'best_model.pt');args=ap.parse_args()
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');results={}
    for label,path in [('legacy',ROOT/'comparison/best_model.pt'),('printed_transfer',ROOT/'runs/physics_v1/best.pt'),('handwriting',args.candidate)]:
        package=torch.load(path,map_location='cpu',weights_only=False)
        model=FormulaOCR(ModelConfig(**package['config']));model.load_state_dict(package['model']);model.to(device).eval()
        results[label]={'checkpoint':str(path),'epoch':package.get('epoch'),'beam_width':package.get('beam_width',1)}
        for split in ['test','test_greek','test_capture','legacy_test','physics_laws']:
            data=ROOT/('comparison/views/D_slash_slash' if split=='legacy_test' else 'data_physics' if split=='physics_laws' else 'data_handwriting')
            name='test' if split=='legacy_test' else split
            rows=[json.loads(line) for line in (data/(name+'.jsonl')).read_text(encoding='utf-8').splitlines()]
            score,predictions=evaluate(model,package,rows,data,device,batch_size=1)
            score['lengths']={f'{lo}-{hi}':metrics([r for r in predictions if lo<=len(r['target_tokens'])<=hi]) for lo,hi in [(1,8),(9,16),(17,24),(25,32),(33,72)]}
            results[label][split]=score
            (OUT/f'{label}_{split}.json').write_text(json.dumps(predictions,ensure_ascii=False,indent=2),encoding='utf-8')
            print(f'{label} {split}: structure={score["semantic_exact"]:.4f} exact={score["exact"]:.4f} TER={score["token_error_rate"]:.4f}',flush=True)
        if label=='handwriting':
            data=ROOT/'data_handwriting';rows=[json.loads(s) for s in (data/'test.jsonl').read_text(encoding='utf-8').splitlines()][:128]
            results[label]['blank_control'],_=evaluate(model,package,rows,data,device,True,batch_size=1)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    (OUT/'handwriting_metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    # Display the measured scope in the UI; this metadata never changes decoding
    # or model selection, and is added only after the sealed test evaluation.
    package=torch.load(args.candidate,map_location='cpu',weights_only=False)
    package['evaluation']={'handwriting_match':results['handwriting']['test']['semantic_exact'],
                           'handwriting_samples':results['handwriting']['test']['n'],
                           'greek_match':results['handwriting']['test_greek']['semantic_exact'],
                           'dataset':'Filtered MathWriting digital-pen rasterization; not real camera photos'}
    temporary=args.candidate.with_suffix('.pt.tmp')
    torch.save(package,temporary);temporary.replace(args.candidate)


if __name__=='__main__':main()
