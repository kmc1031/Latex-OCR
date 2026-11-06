"""Start the local editor, run real API/browser checks, and stop our server."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parent


def main():
    log=ROOT/'physics/ui_server_qa.log'
    environment={**os.environ,'PYTHONUTF8':'1'}
    with log.open('w',encoding='utf-8') as output:
        server=subprocess.Popen([sys.executable,'-u',str(ROOT/'ui_server.py')],cwd=ROOT,
                                stdout=output,stderr=subprocess.STDOUT,env=environment,
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        (ROOT/'physics/ui_qa_process.json').write_text(json.dumps({'pid':server.pid,'command':'ui_server.py','port':8765,'stopped':False}),encoding='utf-8')
        try:
            for _ in range(90):
                if server.poll() is not None:raise RuntimeError('UI server exited; see '+str(log))
                try:
                    with urlopen('http://127.0.0.1:8765/api/status',timeout=1) as response:
                        status=json.loads(response.read())
                    if status['profile']=='physics' and status['evaluation']:break
                except OSError:pass
                time.sleep(.5)
            else:raise RuntimeError('UI server did not become ready')
            checks=['test_ui_api','test_handwriting_ui']
            if (ROOT/'segmentation/test_metrics.json').exists():checks.append('test_segmentation_ui')
            subprocess.run([sys.executable,'-m','unittest',*checks,'-v'],check=True,cwd=ROOT,env=environment)
            subprocess.run(['node','test_ui_browser.cjs'],check=True,cwd=ROOT,env=environment)
        finally:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait()
            (ROOT/'physics/ui_qa_process.json').write_text(json.dumps({'pid':server.pid,'command':'ui_server.py','port':8765,'stopped':True}),encoding='utf-8')
    print('Real API/browser checks passed. Owned preview server stopped.',flush=True)


if __name__=='__main__':main()
