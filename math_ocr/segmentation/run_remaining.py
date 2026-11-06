"""Continue once the symbol classifier's fixed training schedule finishes."""
import json,time,subprocess,sys
from common import ROOT

def main():
    while True:
        try:h=json.loads((ROOT/'models/symbol_history.json').read_text())
        except (OSError,json.JSONDecodeError):h=[]
        if len(h)>=22:break
        time.sleep(10)
    for script,args in [('train_relations.py',[]),('evaluate.py',['--probe']),('train_detector.py',[]),('evaluate.py',['--select'])]:
        print('Running',script,*args,flush=True);subprocess.run([sys.executable,str(ROOT/script),*args],check=True,cwd=ROOT)
    print('CNN segmentation pipeline training and evaluation complete.',flush=True)

if __name__=='__main__':main()
