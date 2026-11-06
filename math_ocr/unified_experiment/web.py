"""Local-only UI: image -> the same frozen model -> LaTeX display and copy."""
import argparse,base64,io,json,threading,time
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
import torch
from PIL import Image
from engine import ROOT,Model

HTML='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>손글씨 수식 인식 실험</title>
<style>body{font:16px system-ui;max-width:980px;margin:40px auto;padding:20px;background:#f8fafc;color:#172033}h1{font-size:28px}section{background:white;border:1px solid #dbe1e9;border-radius:16px;padding:24px;margin:20px 0}button{padding:10px 18px;margin:10px 8px 10px 0;cursor:pointer}textarea{width:95%;min-height:90px;font:17px monospace}img{max-width:100%;max-height:250px}small{color:#586779}#status{white-space:pre-wrap}</style>
<h1>손글씨 수식 → LaTeX</h1><p>수식 하나가 보이는 이미지를 선택하세요. 인식된 코드를 수정하고 복사할 수 있습니다.</p>
<small>MathWriting 디지털 필기 실험 모델입니다. 실제 종이 사진의 정확도는 측정하지 않았습니다.</small>
<section><input id="file" type="file" accept="image/*"><button id="run">인식하기</button><p id="status">이미지를 선택하세요.</p><img id="original" alt="입력 이미지"></section>
<section><h2>인식 결과</h2><img id="formula" alt="인식 수식"><p id="renderstatus"></p><textarea id="latex" aria-label="LaTeX 코드"></textarea><br><button id="copy">LaTeX 복사</button><span id="copied"></span></section>
<script>const file=document.getElementById('file'),original=document.getElementById('original'),latex=document.getElementById('latex'),formula=document.getElementById('formula'),renderstatus=document.getElementById('renderstatus'),copy=document.getElementById('copy'),copied=document.getElementById('copied');let url='';file.onchange=()=>{const f=file.files[0];if(!f)return;const r=new FileReader();r.onload=()=>{url=r.result;original.src=url;document.getElementById('status').textContent='인식할 준비가 되었습니다.'};r.readAsDataURL(f)};
document.getElementById('run').onclick=async()=>{const s=document.getElementById('status');if(!url){s.textContent='이미지를 먼저 선택하세요.';return}s.textContent='인식 중…';try{const res=await fetch('/recognize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:url})});const p=await res.json();if(!res.ok)throw Error(p.error);latex.value=p.latex;formula.src=p.render_png||'';renderstatus.textContent=p.render_error?'수식 미리보기를 만들지 못했습니다. 원시 LaTeX를 확인하세요.':'';s.textContent='인식 시간: '+(p.seconds*1000).toFixed(1)+'ms · 결과를 원본과 확인하세요.'}catch(e){s.textContent=e.message}};
copy.onclick=async()=>{try{await navigator.clipboard.writeText(latex.value)}catch(e){latex.focus();latex.select();document.execCommand('copy')}copied.textContent='복사했습니다.'};</script></html>'''

_model=None
def load_model():
    global _model
    if _model is None:
        ck=torch.load(ROOT/'final_model.pt',map_location='cpu',weights_only=False)
        from engine import CFG
        if ck['config']['class_labels']!=CFG['class_labels'] or ck['config']['output_tokens']!=CFG['output_tokens']:raise ValueError('Checkpoint and current token dictionaries differ')
        _model=Model();_model.load_state_dict(ck['model']);_model.eval();torch.set_num_threads(1)
    return _model

def render(latex):
    import os
    os.environ['MPLCONFIGDIR']=str(ROOT/'plot_cache')
    from matplotlib.mathtext import MathTextParser
    from matplotlib.font_manager import FontProperties
    import numpy as np
    result=MathTextParser('agg').parse('$'+latex+'$',dpi=130,prop=FontProperties(size=22));im=Image.fromarray(255-np.asarray(result.image));out=io.BytesIO();im.save(out,format='PNG');return 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()

class Handler(BaseHTTPRequestHandler):
    def send(self,value,code=200):
        payload=json.dumps(value,ensure_ascii=False).encode();self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    def do_GET(self):
        if self.path=='/status':return self.send({'ready':(ROOT/'final_model.pt').exists(),'state':json.loads((ROOT/'state.json').read_text())})
        if self.path!='/':return self.send({'error':'Not found'},404)
        payload=HTML.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    def do_POST(self):
        if self.path!='/recognize':return self.send({'error':'Not found'},404)
        try:
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<=20*1024*1024:raise ValueError('이미지 크기를 확인하세요.')
            body=json.loads(self.rfile.read(size));data=base64.b64decode(body['image'].split(',',1)[-1],validate=True);im=Image.open(io.BytesIO(data));model=load_model();start=time.perf_counter();result=model.recognize(im);result['seconds']=time.perf_counter()-start
            try:result['render_png']=render(result['latex']);result['render_error']=None
            except Exception as exc:result['render_png']=None;result['render_error']=str(exc)
            self.send(result)
        except FileNotFoundError:self.send({'error':'최종 모델이 아직 준비되지 않았습니다.'},503)
        except Exception as exc:self.send({'error':str(exc)},400)
    def log_message(self,*args):pass

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8767);args=ap.parse_args();print(f'http://127.0.0.1:{args.port}',flush=True);HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
