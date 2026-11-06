"""Prepare disjoint short-formula splits from the original author's HF export."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageChops
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]


def tokenize(latex):
    return re.findall(r"\\[a-zA-Z]+|\\.|[^\s]", latex)


def preprocess(image, height=64, max_width=384):
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    white.alpha_composite(rgba)
    gray = white.convert("L")
    bbox = ImageChops.invert(gray).point(lambda p: 255 if p > 20 else 0).getbbox()
    if not bbox:
        raise ValueError("Blank formula image")
    gray = gray.crop(bbox)
    scale = min((height - 8) / gray.height, (max_width - 8) / gray.width)
    w, h = max(1, round(gray.width * scale)), max(1, round(gray.height * scale))
    gray = gray.resize((w, h), Image.Resampling.LANCZOS)
    width = max(32, ((w + 8 + 3) // 4) * 4)
    canvas = Image.new("L", (width, height), 255)
    canvas.paste(gray, (4, (height - h) // 2))
    return canvas


def tags(tokens):
    return {
        "fraction": "\\frac" in tokens,
        "scripts": "^" in tokens or "_" in tokens,
        "radical": "\\sqrt" in tokens,
        "linear": not any(t in tokens for t in ["\\frac", "^", "_", "\\sqrt"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--long-test", type=int, default=250)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = ROOT / "data"
    (data / "images").mkdir(exist_ok=True)
    used_formulas, used_files, used_pixels = set(), set(), set()
    selected = {}
    summary = {"config": vars(args), "splits": {}}
    counts = Counter()
    for source, out_splits in [("train", ["train"]), ("val", ["val"]), ("test", ["test", "long_test"])]:
        parquet = next((data / "raw").glob(f"{source}-*.parquet"))
        rows = pq.read_table(parquet).to_pylist()
        random.Random(args.seed + len(source)).shuffle(rows)
        print(f"{source}: {len(rows)} raw rows", flush=True)
        for split in out_splits:
            limit = args.long_test if split == "long_test" else getattr(args, split)
            records = []
            rejected = Counter()
            for row in rows:
                tokens = tokenize(row["formula"])
                lower, upper = ((args.max_tokens + 1, 56) if split == "long_test" else (3, args.max_tokens))
                if not lower <= len(tokens) <= upper:
                    rejected["length"] += 1
                    continue
                # Initial scope: a single expression, not arrays/matrices/multiline environments.
                if any(t in tokens for t in ["\\begin", "\\end", "\\\\", "\\vspace"]):
                    rejected["environment"] += 1
                    continue
                key = " ".join(tokens)
                filename = row["filename"]
                if key in used_formulas or filename in used_files:
                    rejected["duplicate_formula_or_filename"] += 1
                    continue
                image = preprocess(Image.open(io.BytesIO(row["image"]["bytes"])))
                digest = hashlib.sha256(np.asarray(image).tobytes() + str(image.size).encode()).hexdigest()
                if digest in used_pixels:
                    rejected["duplicate_pixels"] += 1
                    continue
                used_formulas.add(key)
                used_files.add(filename)
                used_pixels.add(digest)
                relpath = f"images/{split}_{len(records):05d}.png"
                image.save(data / relpath)
                record = {"image": relpath, "tokens": tokens, "latex": key,
                          "source_filename": filename, "image_sha256": digest,
                          "tags": tags(tokens), "width": image.width}
                records.append(record)
                if split == "train":
                    counts.update(tokens)
                if len(records) >= limit:
                    break
            selected[split] = records
            (data / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            summary["splits"][split] = {
                "samples": len(records), "source_split": source, "filter_rejections_until_filled": dict(rejected),
                "min_tokens": min(map(lambda r: len(r["tokens"]), records)),
                "max_tokens": max(map(lambda r: len(r["tokens"]), records)),
                "mean_tokens": float(np.mean([len(r["tokens"]) for r in records])),
                "tags": {tag: sum(r["tags"][tag] for r in records) for tag in ["linear", "fraction", "scripts", "radical"]},
            }
            print(f"{split}: {summary['splits'][split]}", flush=True)
    vocab = SPECIAL + sorted(counts)
    (data / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    for split, records in selected.items():
        unknown = [t for r in records for t in r["tokens"] if t not in counts]
        summary["splits"][split]["oov_tokens"] = len(unknown)
        summary["splits"][split]["oov_samples"] = sum(any(t not in counts for t in r["tokens"]) for r in records)
    summary["vocab_size"] = len(vocab)
    summary["leakage_checks"] = {"canonical_formula_disjoint": True, "source_filename_disjoint": True,
                                  "preprocessed_pixel_hash_disjoint": True}
    (data / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Vocabulary: {len(vocab)} tokens. Disjoint split checks passed.", flush=True)


if __name__ == "__main__":
    main()
