"""Adapt the separate image-only symbol pipeline to the existing local editor."""
import base64,io,json,sys,threading,time
from pathlib import Path
import torch
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent;SEG=ROOT/'segmentation'
sys.path.insert(0,str(SEG))
from pipeline import Pipeline
from physics_syntax import GLYPHS,FUNCTIONS

def png(image):
    b=io.BytesIO();image.save(b,'PNG');return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()

class SegmentRecognizer:
    def __init__(self):
        self.pipeline=Pipeline();self.device=self.pipeline.device;self.lock=threading.Lock()
        self.package={'syntax':'physics-v1','preprocessing':'segmentation-v1','selected_variant':'SymbolCNN + RelationMLP'}
    def status(self):
        metrics=json.loads((SEG/'test_metrics.json').read_text())
        method=metrics['selected_automatic_method'];scope=json.loads((SEG/'data/summary.json').read_text())
        architecture='연결요소 분할 + 기호 CNN + 관계 MLP' if method=='components' else '검출 CNN + 기호 CNN + 관계 MLP'
        return {'ready':True,'device':str(self.device),'model':self.package['selected_variant'],'architecture':architecture,
                'input_notation':'mixed','syntax':'physics-v1','preprocessing':'segmentation-v1','vocabulary':self.pipeline.vocab,'glyphs':GLYPHS,'functions':sorted(FUNCTIONS),
                'evaluation':{'handwriting_samples':metrics['n'],'handwriting_match':metrics['modes'][method]['accuracy'],'dataset':'CROHME 2013 supported subset'},
                'scope':'CROHME 2013 기호·구조 부분집합. 분수·첨자·근호를 조합하며 실제 공책 사진의 성능은 검증하지 않았습니다.'}
    def samples(self):
        demos=json.loads((SEG/'models/settings.json').read_text()).get('demos',[])
        rows=[json.loads(s) for s in (SEG/'data/val.jsonl').read_text().splitlines()];lookup={r['id']:r for r in rows}
        if not demos:demos=[r['id'] for r in sorted(rows,key=lambda r:len(r['glyphs']))[:3]]
        return [{'id':f'segments_{i}','label':name,'symbol':symbol,'path':SEG/'data'/lookup[key]['image']} for i,(key,name,symbol) in enumerate(zip(demos,['손글씨','첨자·기호','수식 조합'],['x+y','xⁿ','a/b']))]
    def predict(self,raw):
        start=time.perf_counter()
        with Image.open(io.BytesIO(raw)) as source:
            if source.width*source.height>20_000_000:raise ValueError('Image too large')
            with self.lock:answer=self.pipeline.recognize(source)
        prepared=answer.pop('prepared');overlay=prepared.convert('RGB');draw=ImageDraw.Draw(overlay)
        for i,box in enumerate(answer['boxes']):draw.rectangle(box,outline='#147d53',width=1);draw.text((box[0],max(0,box[1]-10)),str(i+1),fill='#147d53')
        return {**answer,'terminated':True,'prepared_image':png(prepared),'prepared_size':list(prepared.size),
                'segmentation_overlay':png(overlay),'elapsed_ms':round((time.perf_counter()-start)*1000),'model':self.package['selected_variant'],'device':str(self.device)}
