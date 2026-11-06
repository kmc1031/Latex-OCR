"""Compare the unchanged previous OCR on the exact same held-out source images."""
import json
import torch
from PIL import Image
from common import ROOT
from evaluate_physics import evaluate
from model import FormulaOCR,ModelConfig

def main():
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    package=torch.load(ROOT.parent/'physics/best_model.pt',map_location='cpu',weights_only=False)
    model=FormulaOCR(ModelConfig(**package['config']));model.load_state_dict(package['model']);model.to(device).eval()
    rows=[json.loads(s) for s in (ROOT/'data/test.jsonl').read_text().splitlines()]
    metrics,predictions=evaluate(model,package,rows,ROOT/'data',device,batch_size=1)
    (ROOT/'previous_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');(ROOT/'previous_predictions.json').write_text(json.dumps(predictions,indent=2),encoding='utf-8');print(json.dumps(metrics),flush=True)

if __name__=='__main__':main()
