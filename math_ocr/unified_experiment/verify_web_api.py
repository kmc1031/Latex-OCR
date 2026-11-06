"""Loopback API checks and JS mock checks; no browser automation or clipboard reads."""
import argparse,base64,io,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8768);ap.add_argument('--self-host',action='store_true');args=ap.parse_args();base=f'http://127.0.0.1:{args.port}';server=None
    if args.self_host:
        log=open(ROOT/'web_server.log','a',encoding='utf-8');flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        server=subprocess.Popen([sys.executable,str(ROOT/'web.py'),'--port',str(args.port)],stdout=log,stderr=log,creationflags=flags)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(40):
        try:status=json.loads(opener.open(base+'/status',timeout=3).read());break
        except Exception:
            if attempt==39:raise
            time.sleep(.5)
    page=opener.open(base).read().decode('utf-8');assert 'const file=document.getElementById' in page and 'LaTeX' in page
    rows=json.loads((ROOT/'prepared_manifest.json').read_text(encoding='utf-8'));row=next(r for r in rows if r['split']=='val');encoded=base64.b64encode((ROOT/row['image']).read_bytes()).decode();request=urllib.request.Request(base+'/recognize',data=json.dumps({'image':'data:image/png;base64,'+encoded}).encode(),headers={'Content-Type':'application/json'})
    result=json.loads(opener.open(request,timeout=60).read());assert isinstance(result['tokens'],list) and result['latex']==' '.join(result['tokens']);assert 'render_png' in result
    if result['render_png']:
        image=Image.open(io.BytesIO(base64.b64decode(result['render_png'].split(',',1)[1])));assert image.width>0;image.save(ROOT/'ui_render_preview.png')
    (ROOT/'ui_fixture.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');(ROOT/'ui_page_snapshot.html').write_text(page,encoding='utf-8')
    checks={'api_status_ready':status['ready'],'page_served':True,'image_only_inference_response':True,'render_success':bool(result['render_png']),'render_fallback_reason':result.get('render_error'),'url':base,'browser_live_clicks_verified':False,'browser_limitation':'In-app browser localhost timeout; native Chrome control not approved. No browser permissions or firewall settings changed.','os_clipboard_verified':False,'server_pid':server.pid if server else None,'sample_id':row['sample_id']}
    (ROOT/'ui_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8');print(checks)

if __name__=='__main__':main()
