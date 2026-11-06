"""Build an application-oriented slash candidate with fonts and nested examples.

All new nested training formulas exclude every held-out expression. Font banks
only contain training expressions. Validation shifts use validation identities.
"""
import hashlib
import json
import os
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image,ImageFilter

from prepare_data import preprocess,tags
from slash_semantics import parse,slash_tokens

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"comparison"


def main():
    dest=OUT/"views/P_practical";dest.mkdir(parents=True,exist_ok=True)
    images=OUT/"practical_images";images.mkdir(exist_ok=True)
    renderer=MathTextParser("agg")
    props={f:FontProperties(size=24,math_fontfamily=f) for f in ["stix","dejavusans","cm"]}
    rng=random.Random(918273)
    dview=OUT/"views/D_slash_slash"
    def read(folder,split):return [json.loads(x) for x in (folder/f"{split}.jsonl").read_text().splitlines()]
    def rel(path):return os.path.relpath(path,dest).replace("\\","/")
    def render(tokens,font):
        r=renderer.parse("$"+" ".join(tokens)+"$",dpi=110,prop=props[font])
        return preprocess(Image.fromarray(255-np.asarray(r.image)))
    training=read(dview,"train")
    forbidden={parse(r["reference_tex"]) for split in ["train","val","test","long_test"] for r in read(dview,split)}
    atoms=list("xyzabc123456789")
    def binary():return [rng.choice(atoms),rng.choice(["+","-"]),rng.choice(atoms)]
    for i in range(1000):
        while True:
            # Both numerator and denominator nesting; base training contains neither.
            if i%2:
                tex=["\\frac","{"]+binary()+["}","{","\\frac","{"]+binary()+["}","{",rng.choice(atoms),"}","}","+"]+binary()
            else:
                tex=["\\frac","{","\\frac","{"]+binary()+["}","{",rng.choice(atoms),"}","}","{"]+binary()+["}","-"]+binary()
            tree=parse(tex)
            if tree not in forbidden:forbidden.add(tree);break
        tokens=slash_tokens(tex);assert parse(tokens)==tree
        path=images/f"nested_{i:04d}_stix.png";image=render(tokens,"stix");image.save(path)
        training.append({"image":rel(path),"tokens":tokens,"latex":" ".join(tokens),"reference_tex":tex,
            "reference_slash":tokens,"pair_id":f"nested_train_{i:04d}","family":"nested_fraction",
            "tags":tags(tex),"width":image.width,"source_filename":f"nested_train_{i:04d}",
            "image_sha256":hashlib.sha256(np.asarray(image).tobytes()+str(image.size).encode()).hexdigest()})
    print("Created 1000 disjoint nested training expressions",flush=True)
    for i,row in enumerate(training):
        if i<3000:row["image"]=rel((dview/row["image"]).resolve())
        variants=[]
        for font in ["dejavusans","cm"]:
            path=images/f"{row['pair_id']}_{font}.png";render(row["tokens"],font).save(path);variants.append(rel(path))
        row["image_variants"]=variants
        if (i+1)%1000==0:print(f"Font banks: {i+1}/{len(training)}",flush=True)
    (dest/"train.jsonl").write_text("\n".join(json.dumps(r) for r in training)+"\n",encoding="utf-8")
    (dest/"vocab.json").write_text((dview/"vocab.json").read_text(),encoding="utf-8")
    # Copy only manifests; all existing images remain shared immutable files.
    for split in ["val","test","long_test","test_font","test_capture"]:
        rows=read(dview,split)
        for r in rows:r["image"]=rel((dview/r["image"]).resolve())
        (dest/f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n",encoding="utf-8")
    # Validation font/capture shifts for all views; never use test identities for selection.
    for name in ["A_stack_tex","B_stack_slash","C_slash_tex","D_slash_slash","P_practical"]:
        view=OUT/"views"/name
        rows=read(view,"val");fontrows=[];capturerows=[]
        layout="stack" if name[0] in ["A","B"] else "slash"
        for j,row in enumerate(rows):
            shown=row["reference_tex"] if layout=="stack" else row["reference_slash"]
            fontpath=images/f"val_{layout}_{row['pair_id']}_dejavusans.png"
            render(shown,"dejavusans").save(fontpath)
            orig=(view/row["image"]).resolve()
            cap=Image.open(orig).convert("L").rotate((-1 if j%2 else 1)*.7,resample=Image.Resampling.BICUBIC,fillcolor=255).filter(ImageFilter.GaussianBlur(.35))
            cap_path=images/f"val_{layout}_{row['pair_id']}_capture.png";cap.save(cap_path)
            for path,collection in [(fontpath,fontrows),(cap_path,capturerows)]:
                with Image.open(path) as image:
                    sha=hashlib.sha256(np.asarray(image).tobytes()+str(image.size).encode()).hexdigest()
                collection.append({**row,"image":os.path.relpath(path,view).replace("\\","/"),"image_sha256":sha})
        for split,rs in [("val_font",fontrows),("val_capture",capturerows)]:
            (view/f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rs)+"\n",encoding="utf-8")
    nested_test={parse(r["reference_tex"]) for r in read(dview,"long_test")}
    assert not {parse(r["reference_tex"]) for r in training}&nested_test
    (OUT/"practical_data_summary.json").write_text(json.dumps({"train_formulas":4000,"base_formulas":3000,
        "added_nested_formulas":1000,"training_fonts":["stix","dejavusans","cm"],
        "font_selection":"One random font per formula per epoch", "nested_test_training_overlap":0,
        "validation_shifts_use_validation_identities":True},indent=2),encoding="utf-8")
    print("Practical training data and validation shifts ready",flush=True)


if __name__=="__main__":main()
