"""Additional cross-domain diagnostic on the previously used MathWriting suite.

This is not a new blind test, and never changes pipeline settings.
Both models receive the same saved 64px source images from the earlier suite.
"""
import json,time
from PIL import Image
from common import ROOT
from pipeline import Pipeline
from evaluate import correct

def main():
    model=Pipeline();data=ROOT.parent/'data_handwriting';rows=[json.loads(s) for s in (data/'test.jsonl').read_text().splitlines()];out=[];began=time.perf_counter()
    for k,row in enumerate(rows):
        prediction=model.recognize(Image.open(data/row['image']))
        out.append({'image':row['image'],'target':row['tokens'],'prediction':prediction['tokens'],'correct':correct(prediction['prediction'],row['tokens']),'greek':row['greek']})
        if (k+1)%200==0:print('Cross-domain diagnostic',k+1,'/',len(rows),flush=True)
    (ROOT/'transfer_predictions.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    previous=json.loads((ROOT.parent/'physics/handwriting_metrics.json').read_text(encoding='utf-8'))['handwriting']
    greek=[r for r in out if r['greek']]
    result={'diagnostic_only_previously_used_test':True,'settings_changed':False,'n':len(out),'accuracy':sum(r['correct'] for r in out)/len(out),
            'greek_n':len(greek),'greek_accuracy':sum(r['correct'] for r in greek)/len(greek),'previous_accuracy':previous['test']['semantic_exact'],
            'source':'Same saved 64px MathWriting raster images as the previous experiment, not original high-resolution photos.',
            'seconds':round(time.perf_counter()-began,2)}
    (ROOT/'transfer_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(ROOT/'transfer_predictions.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
