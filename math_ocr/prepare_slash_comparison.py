"""Paired views of the SAME previous train/val/test expressions, without leakage."""
import hashlib
import json
import os
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
import numpy as np
from PIL import Image, ImageFilter

from prepare_data import SPECIAL, preprocess
from slash_semantics import parse, self_check, slash_tokens

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "comparison"
SOURCE = ROOT / "data_controlled"
VIEWS = {"A_stack_tex":("stack", "tex"), "B_stack_slash":("stack", "slash"),
         "C_slash_tex":("slash", "tex"), "D_slash_slash":("slash", "slash")}


def digest(image):
    return hashlib.sha256(np.asarray(image).tobytes() + str(image.size).encode()).hexdigest()


def main():
    self_check()
    renderer = MathTextParser("agg")
    props = {name:FontProperties(size=24, math_fontfamily=name) for name in ["stix", "dejavusans"]}
    for layout in ["stack", "slash"]:
        (OUT / "images" / layout).mkdir(parents=True, exist_ok=True)
    records = {s:[json.loads(x) for x in (SOURCE/f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]
               for s in ["train", "val", "test", "long_test"]}
    semantics = {s:{r["latex"] for r in rs} for s,rs in records.items()}
    keys = list(semantics)
    assert all(not semantics[a]&semantics[b] for i,a in enumerate(keys) for b in keys[i+1:])

    def render(tokens, font):
        result = renderer.parse("$"+" ".join(tokens)+"$", dpi=110, prop=props[font])
        return preprocess(Image.fromarray(255-np.asarray(result.image)))

    generated = {layout:{} for layout in ["stack", "slash"]}
    semantic_count = 0
    for split, rows in records.items():
        for layout in generated:
            generated[layout][split] = []
        for row in rows:
            tex = row["tokens"]
            slash = slash_tokens(tex)
            assert parse(tex) == parse(slash), row["latex"]
            semantic_count += 1
            pair_id = row["image"].split("/")[-1].removesuffix(".png")
            for layout in generated:
                target_path = SOURCE / row["image"] if layout == "stack" else OUT / "images/slash" / f"{pair_id}.png"
                if layout == "slash":
                    render(slash, "stix").save(target_path)
                generated[layout][split].append((row, tex, slash, target_path, pair_id))
        print(f"Paired {split}: {len(rows)}", flush=True)
    # Distribution shifts: same held-out formulas, independently rendered/corrupted images.
    for layout in generated:
        generated[layout]["test_font"] = []
        generated[layout]["test_capture"] = []
        for j, (row, tex, slash, original, pair_id) in enumerate(generated[layout]["test"]):
            shown = tex if layout == "stack" else slash
            font_path = OUT / "images" / layout / f"font_{pair_id}.png"
            render(shown, "dejavusans").save(font_path)
            capture_path = OUT / "images" / layout / f"capture_{pair_id}.png"
            rng = np.random.default_rng(1000+j)
            image = Image.open(original).convert("L")
            image = image.rotate(float(rng.uniform(-1, 1)), resample=Image.Resampling.BICUBIC, fillcolor=255)
            image = image.filter(ImageFilter.GaussianBlur(0.35))
            image = Image.fromarray(np.clip(np.array(image).astype(float)+rng.normal(0, 4, (image.height,image.width)),0,255).astype(np.uint8))
            image.save(capture_path)
            generated[layout]["test_font"].append((row,tex,slash,font_path,pair_id))
            generated[layout]["test_capture"].append((row,tex,slash,capture_path,pair_id))
    report = {"source":"data_controlled, unchanged expression splits", "views":{},
              "semantic_conversion_checks": semantic_count, "train_test_formula_overlap":0,
              "font_shift":"STIX -> DejaVu Sans, no training on shifted font",
              "capture_shift":"rotation +/-1 degree, Gaussian blur 0.35px, intensity noise sigma 4/255",
              "paired_shift_tests_share_test_formulas":True}
    for name,(layout,output) in VIEWS.items():
        dest = OUT / "views" / name
        dest.mkdir(parents=True, exist_ok=True)
        manifests = {}
        for split, rows in generated[layout].items():
            manifest=[]
            for row,tex,slash,path,pair_id in rows:
                tokens = tex if output=="tex" else slash
                with Image.open(path) as image:
                    sha = digest(image)
                    width = image.width
                manifest.append({**row,"image":os.path.relpath(path,dest).replace("\\","/"),
                    "tokens":tokens,"latex":" ".join(tokens),"reference_tex":tex,"reference_slash":slash,
                    "pair_id":pair_id,"width":width,"image_sha256":sha})
            manifests[split]=manifest
            (dest/f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in manifest)+"\n",encoding="utf-8")
        vocab = SPECIAL+sorted({t for r in manifests["train"] for t in r["tokens"]})
        (dest/"vocab.json").write_text(json.dumps(vocab,indent=2),encoding="utf-8")
        report["views"][name]={"input":layout,"output":output,"vocab":len(vocab),
            "samples":{s:len(rs) for s,rs in manifests.items()}}
    # The existing A checkpoint sees exactly the same train/val/test images and labels.
    report["baseline_A"]="runs/controlled/best.pt, already trained 40 epochs on these exact expressions"
    (OUT/"data_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("PASS: identical paired expression splits; all fraction conversions preserve AST.",flush=True)


if __name__=="__main__":
    main()
