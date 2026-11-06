import json,subprocess,sys,time
from common import ROOT
def main():
    previous=0
    while True:
        try:h=json.loads((ROOT/'models/detector_history.json').read_text())
        except (OSError,json.JSONDecodeError):h=[]
        if len(h)!=previous:print('Detector epochs',len(h),'/ 8',flush=True);previous=len(h)
        if len(h)>=8:break
        time.sleep(10)
    for name,args in [('evaluate.py',['--select']),('compare_previous.py',[]),('make_report.py',[])]:
        print('Running',name,flush=True);subprocess.run([sys.executable,str(ROOT/name),*args],check=True,cwd=ROOT)
    print('Final evaluation and report complete.',flush=True)
if __name__=='__main__':main()
