"""A controlled, formula-disjoint diagnostic dataset, not a real benchmark.

Render independently generated LaTeX with Matplotlib's STIX math font. Five
known template families test symbol recognition and simple 2D relationships.
The long split uses a held-out nested fraction composition.
"""
import hashlib
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image

from prepare_data import SPECIAL, preprocess, tags, tokenize

ROOT = Path(__file__).resolve().parent


def main():
    dest = ROOT / "data_controlled"
    (dest / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261022)
    parser = MathTextParser("agg")
    prop = FontProperties(size=24, math_fontfamily="stix")
    atoms = list("xyzabc123456789")
    letters = list("xyzabc")
    def atom():
        return rng.choice(atoms)
    def binary():
        return atom() + rng.choice(["+", "-"]) + atom()
    def formula(family):
        if family == "linear":
            return binary() + rng.choice(["=", "+", "-"]) + atom()
        if family == "superscript":
            return rng.choice(letters) + "^{" + rng.choice("234") + "}" + rng.choice(["+", "-", "="]) + binary()
        if family == "subscript":
            return rng.choice(letters) + "_{" + rng.choice("123456789") + "}" + rng.choice(["+", "-", "="]) + binary()
        if family == "fraction":
            return "\\frac{" + binary() + "}{" + atom() + "}" + rng.choice(["", "+"+atom(), "="+atom()])
        if family == "radical":
            return "\\sqrt{" + binary() + "}" + rng.choice(["+", "-", "="]) + atom()
        return "\\frac{" + binary() + "}{\\frac{" + binary() + "}{" + atom() + "}}+" + binary()

    families = ["linear", "superscript", "subscript", "fraction", "radical"]
    used, used_images = set(), set()
    all_records = {}
    for split, size in [("train", 3000), ("val", 300), ("test", 500), ("long_test", 200)]:
        records = []
        while len(records) < size:
            family = "nested_fraction" if split == "long_test" else families[len(records) % len(families)]
            tex = formula(family)
            tokens = tokenize(tex)
            key = " ".join(tokens)
            if key in used:
                continue
            rendered = parser.parse("$" + tex + "$", dpi=110, prop=prop)
            image = preprocess(Image.fromarray(255 - np.asarray(rendered.image)))
            digest = hashlib.sha256(np.asarray(image).tobytes() + str(image.size).encode()).hexdigest()
            if digest in used_images:
                continue
            used.add(key)
            used_images.add(digest)
            rel = f"images/{split}_{len(records):05d}.png"
            image.save(dest / rel)
            records.append({"image": rel, "tokens": tokens, "latex": key,
                            "source_filename": f"controlled_{split}_{len(records)}",
                            "image_sha256": digest, "tags": tags(tokens),
                            "width": image.width, "family": family})
        # Remove grouped family order while retaining a reproducible evaluation order.
        rng.shuffle(records)
        all_records[split] = records
        (dest / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in records)+"\n", encoding="utf-8")
        print(split, len(records), flush=True)
    vocab = SPECIAL + sorted({t for r in all_records["train"] for t in r["tokens"]})
    (dest / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    summary = {"source": "Generated here with a fixed five-family grammar and Matplotlib/STIX.",
               "seed": 20261022, "vocab_size": len(vocab), "shared_template_families": True,
               "claim_scope": "Unseen formulas from known templates; not arbitrary formula or handwriting recognition.",
               "splits": {s:{"samples":len(rs), "min_tokens":min(len(r['tokens']) for r in rs),
                              "max_tokens":max(len(r['tokens']) for r in rs),
                              "tags":{k:sum(r['tags'][k] for r in rs) for k in ['linear','fraction','scripts','radical']}}
                          for s,rs in all_records.items()},
               "leakage_checks": {"canonical_formula_disjoint":True,"preprocessed_pixel_hash_disjoint":True}}
    (dest / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
