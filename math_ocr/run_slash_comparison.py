"""Run notation variants and the practical CNN/LSTM candidate sequentially."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def main():
    logs=ROOT/"comparison/logs";logs.mkdir(parents=True,exist_ok=True)
    jobs=[("slash_C",["train.py","--run","slash_C","--data-dir","comparison/views/C_slash_tex","--epochs","40","--max-decode","48"]),
          ("slash_D",["train.py","--run","slash_D","--data-dir","comparison/views/D_slash_slash","--epochs","40","--max-decode","48"]),
          ("slash_B",["train.py","--run","slash_B","--data-dir","comparison/views/B_stack_slash","--epochs","40","--max-decode","48"]),
          ("slash_practical",["train_practical.py","--run","slash_practical","--epochs","24"])]
    environment=dict(os.environ);environment["PYTHONIOENCODING"]="utf-8"
    started=time.perf_counter()
    for name,args in jobs:
        if (ROOT/"runs"/name/"metrics.json").exists():
            print(f"SKIP completed {name}",flush=True);continue
        print(f"START {name}",flush=True)
        with (logs/f"{name}.log").open("w",encoding="utf-8") as log:
            proc=subprocess.Popen([sys.executable,"-u"]+args,cwd=ROOT,env=environment,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace")
            for line in proc.stdout:
                log.write(line);log.flush()
                if line.startswith("{"):
                    try: data=json.loads(line)
                    except json.JSONDecodeError: data={}
                    if "epoch" in data and (data["epoch"]%5==0 or data["epoch"]==1):
                        print(name+" "+line.strip(),flush=True)
                elif line.startswith(("Initial","Training wall","CONTROL","FINAL test:","Results saved")):
                    print(name+" "+line.strip()[:220],flush=True)
            code=proc.wait()
        if code:
            tail=(logs/f"{name}.log").read_text(encoding="utf-8")[-7000:]
            raise RuntimeError(f"{name} exited {code}\n{tail}")
        print(f"DONE {name}",flush=True)
    print(f"ALL TRAINING DONE: {time.perf_counter()-started:.1f} seconds",flush=True)


if __name__=="__main__":main()
