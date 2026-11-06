"""Wait for the current training schedule, select on validation, then evaluate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--wait-epochs',type=int,default=0)
    args=ap.parse_args()
    previous=0
    while args.wait_epochs:
        try:history=json.loads((ROOT/'runs/physics_handwriting_deep/history.json').read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError):history=[]
        if len(history)!=previous:
            previous=len(history)
            print(f'Training completed epochs: {previous}/{args.wait_epochs}; best validation: '+
                  f'{max((r["score"] for r in history),default=0):.4f}',flush=True)
        if previous>=args.wait_epochs:break
        time.sleep(15)
    for script in ['choose_handwriting_candidate.py','evaluate_handwriting.py','diagnose_handwriting.py','make_handwriting_report.py']:
        print('Running '+script,flush=True)
        subprocess.run([sys.executable,str(ROOT/script)],check=True,cwd=ROOT)
    print('Selection, sealed test evaluation and report complete.',flush=True)


if __name__=='__main__':main()
