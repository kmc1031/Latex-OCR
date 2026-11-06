"""Audit paired formulas, conversion correctness and actual saved image hashes."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from slash_semantics import parse,self_check

ROOT=Path(__file__).resolve().parent


def main():
    self_check()
    out=ROOT/"comparison";reports={};views={}
    for folder in sorted((out/"views").iterdir()):
        sets={};data={}
        for split in ["train","val","test","long_test","test_font","test_capture","val_font","val_capture"]:
            rows=[json.loads(x) for x in (folder/f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            data[split]=rows;sets[split]={r["pair_id"] for r in rows}
            for r in rows:
                assert parse(r["tokens"])==parse(r["reference_tex"])
                with Image.open(folder/r["image"]) as image:
                    assert image.height==64 and image.width<=384 and image.width%4==0
                    digest=hashlib.sha256(np.asarray(image).tobytes()+str(image.size).encode()).hexdigest()
                    assert digest==r["image_sha256"]
        core=["train","val","test","long_test"]
        assert all(not sets[a]&sets[b] for i,a in enumerate(core) for b in core[i+1:])
        trees={s:{parse(r["reference_tex"]) for r in data[s]} for s in core}
        assert all(not trees[a]&trees[b] for i,a in enumerate(core) for b in core[i+1:])
        assert sets["test_font"]==sets["test_capture"]==sets["test"]
        views[folder.name]=data
        reports[folder.name]={"counts":{s:len(rs) for s,rs in data.items()},"core_split_pair_overlap":0,"image_hashes_verified":True}
    baseline=views["A_stack_tex"]
    for view in views.values():
        for split in baseline:
            base_ids=[r["pair_id"] for r in baseline[split]]
            view_ids=[r["pair_id"] for r in view[split]]
            if split=="train":
                assert view_ids[:len(base_ids)]==base_ids
            else:
                assert base_ids==view_ids
    # A labeled visual check of eight fractions from the actual test order.
    font=ImageFont.truetype("C:/Windows/Fonts/consola.ttf",16)
    fractions=[r for r in baseline["test"] if r["tags"]["fraction"]][:8]
    slash={r["pair_id"]:r for r in views["D_slash_slash"]["test"]}
    preview=Image.new("RGB",(1000,8*112+50),"white");draw=ImageDraw.Draw(preview)
    draw.text((15,12),"STACKED INPUT                       SLASH INPUT                     SAME EXPRESSION",font=font,fill="black")
    for i,r in enumerate(fractions):
        sr=slash[r["pair_id"]];y=50+i*112
        a=Image.open(out/"views/A_stack_tex"/r["image"]).convert("RGB")
        b=Image.open(out/"views/D_slash_slash"/sr["image"]).convert("RGB")
        preview.paste(a,(15,y));preview.paste(b,(300,y))
        draw.text((700,y+15)," ".join(sr["tokens"]),font=font,fill="black")
    preview.save(out/"paired_examples.png")
    (out/"data_audit.json").write_text(json.dumps(reports,indent=2),encoding="utf-8")
    print("PASS: paired identities, syntax-preserving conversion, dimensions, image checksums and no core split identity overlap.")


if __name__=="__main__":main()
