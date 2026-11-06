"""Reproducible pilot training; all reported accuracy uses free-running decoding."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import FormulaOCR, ModelConfig

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def read_records(split):
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]


class FormulaDataset(Dataset):
    def __init__(self, records, vocab, augment=False):
        self.records = records
        self.vocab = vocab
        self.lookup = {t: i for i, t in enumerate(vocab)}
        self.augment = augment
        self.images = [np.array(Image.open(DATA / r["image"]), dtype=np.uint8) for r in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        ink = (255 - self.images[idx]).astype(np.float32) / 255.0
        if self.augment:
            dx, dy = random.randint(-2, 2), random.randint(-2, 2)
            shifted = np.zeros_like(ink)
            srcx, dstx = max(0, -dx), max(0, dx)
            srcy, dsty = max(0, -dy), max(0, dy)
            w, h = ink.shape[1] - abs(dx), ink.shape[0] - abs(dy)
            shifted[dsty:dsty+h, dstx:dstx+w] = ink[srcy:srcy+h, srcx:srcx+w]
            ink = shifted
        r = self.records[idx]
        ids = [1] + [self.lookup.get(t, 3) for t in r["tokens"]] + [2]
        return torch.from_numpy(ink).unsqueeze(0), torch.tensor(ids), idx


def collate(batch):
    images, tokens, indexes = zip(*batch)
    widths = torch.tensor([x.size(-1) for x in images])
    canvas = torch.zeros(len(batch), 1, images[0].size(1), int(widths.max()))
    padded = torch.zeros(len(batch), max(t.numel() for t in tokens), dtype=torch.long)
    for i, (image, target) in enumerate(zip(images, tokens)):
        canvas[i, :, :, :image.size(-1)] = image
        padded[i, :target.numel()] = target
    return canvas, widths, padded, indexes


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


def summarize(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    exact = sum(r["exact"] for r in rows)
    # Wilson interval describes sampling uncertainty for this selected subset only.
    p, z = exact / n, 1.96
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return {"n": n, "exact_count": exact, "exact_match": p,
            "exact_wilson_95": [max(0, center-half), min(1, center+half)],
            "token_error_rate": sum(r["edit_distance"] for r in rows) / sum(len(r["target_tokens"]) for r in rows),
            "normalized_similarity": float(np.mean([r["similarity"] for r in rows])),
            "eos_rate": sum(r["ended"] for r in rows) / n}


@torch.inference_mode()
def evaluate(model, loader, device, mode="normal", max_tokens=72):
    model.eval()
    rows = []
    for images, widths, _, indexes in loader:
        images = images.to(device)
        if mode == "blank":
            images = torch.zeros_like(images)
        elif mode == "shuffled":
            images = images.roll(1, 0)
            widths = widths.roll(1, 0)
        predictions = model.generate(images, widths, max_tokens).cpu().tolist()
        for idx, predicted in zip(indexes, predictions):
            ended = 2 in predicted
            ids = predicted[:predicted.index(2)] if ended else predicted
            words = [loader.dataset.vocab[i] for i in ids]
            record = loader.dataset.records[idx]
            target = record["tokens"]
            distance = edit_distance(target, words)
            rows.append({"image": record["image"], "target": record["latex"],
                         "prediction": " ".join(words), "target_tokens": target,
                         "predicted_tokens": words, "exact": bool(words == target and ended),
                         "edit_distance": distance,
                         "similarity": 1 - distance / max(len(target), len(words), 1),
                         "ended": ended, "tags": record["tags"]})
    metrics = summarize(rows)
    metrics["groups"] = {tag: summarize([r for r in rows if r["tags"][tag]])
                         for tag in ["linear", "fraction", "scripts", "radical"]}
    metrics["length_groups"] = {label: summarize([r for r in rows if lo <= len(r["target_tokens"]) <= hi])
                                for label, lo, hi in [("3-12", 3, 12), ("13-22", 13, 22), ("23-32", 23, 32), ("33-56", 33, 56)]}
    return metrics, rows


def main():
    global DATA
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", default="pilot")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--max-decode", type=int, default=72)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    DATA = args.data_dir.resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = ROOT / "runs" / args.run
    output.mkdir(parents=True, exist_ok=True)
    vocab = json.loads((DATA / "vocab.json").read_text(encoding="utf-8"))
    train_records = read_records("train")
    if args.train_limit:
        train_records = train_records[:args.train_limit]
    val_records = read_records("val")
    if args.val_limit:
        val_records = val_records[:args.val_limit]
    def loader(records, training=False):
        return DataLoader(FormulaDataset(records, vocab, training and not args.no_augment),
                          batch_size=args.batch_size, shuffle=training, collate_fn=collate,
                          num_workers=0, pin_memory=device.type == "cuda")
    training = loader(train_records, True)
    validation = loader(val_records)
    config = ModelConfig(vocab_size=len(vocab))
    model = FormulaOCR(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    history = []
    start_epoch = 1
    best_key = (-1, -1)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if not args.eval_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            # A new command can explicitly change the resumed learning rate.
            for group in optimizer.param_groups:
                group["lr"] = args.lr
            start_epoch = checkpoint["epoch"] + 1
            history = checkpoint["history"]
            best_key = tuple(checkpoint["best_key"])
            random.setstate(checkpoint["rng_python"])
            np.random.set_state(checkpoint["rng_numpy"])
            torch.set_rng_state(checkpoint["rng_torch"].cpu())
            if device.type == "cuda" and "rng_cuda" in checkpoint:
                torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["rng_cuda"]])
            if not (output / "best.pt").exists():
                # Resuming in a new run directory must retain the baseline candidate.
                previous_best = args.resume.parent / "best.pt"
                baseline = torch.load(previous_best, map_location=device, weights_only=False) if previous_best.exists() else checkpoint
                torch.save(baseline, output / "best.pt")
    parameters = sum(p.numel() for p in model.parameters())
    metadata = {"args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                "config": model.config_dict(), "parameters": parameters,
                "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "torch": torch.__version__, "train_samples": len(train_records), "val_samples": len(val_records),
                "attention": False, "decode": "greedy, no reference tokens provided"}
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    if not args.eval_only:
        initial, _ = evaluate(model, validation, device, max_tokens=args.max_decode)
        (output / "untrained_val_metrics.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
        print(f"Initial val: exact={initial['exact_match']:.3f}, similarity={initial['normalized_similarity']:.3f}", flush=True)
        total_start = time.perf_counter()
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            start = time.perf_counter()
            losses, token_total = 0.0, 0
            for step, (images, widths, targets, _) in enumerate(training, 1):
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images, widths, targets[:, :-1])
                loss = criterion(logits.reshape(-1, len(vocab)), targets[:, 1:].reshape(-1))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                num = int(targets[:, 1:].ne(0).sum())
                losses += float(loss.detach()) * num
                token_total += num
                if step % 75 == 0:
                    print(f"epoch {epoch}, step {step}/{len(training)}, CE={losses/token_total:.4f}", flush=True)
            metrics, _ = evaluate(model, validation, device, max_tokens=args.max_decode)
            entry = {"epoch": epoch, "train_ce": losses / token_total,
                     "val_exact": metrics["exact_match"], "val_similarity": metrics["normalized_similarity"],
                     "val_ter": metrics["token_error_rate"], "seconds": time.perf_counter() - start,
                     "lr": optimizer.param_groups[0]["lr"]}
            history.append(entry)
            key = (metrics["exact_match"], metrics["normalized_similarity"])
            improved = key > best_key
            if improved:
                best_key = key
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "config": model.config_dict(), "vocab": vocab, "epoch": epoch,
                     "history": history, "best_key": best_key, "metadata": metadata,
                     "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                     "rng_torch": torch.get_rng_state()}
            if device.type == "cuda":
                state["rng_cuda"] = torch.cuda.get_rng_state_all()
            torch.save(state, output / "last.pt")
            if improved:
                torch.save(state, output / "best.pt")
            (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(json.dumps(entry), flush=True)
        print(f"Training wall time: {time.perf_counter()-total_start:.1f}s", flush=True)
        checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
    results = {"model": metadata, "checkpoint_epoch": checkpoint["epoch"]}
    for split in ["train_probe", "val", "test", "long_test"]:
        records = train_records[:256] if split == "train_probe" else (val_records if split == "val" else read_records(split))
        metrics, predictions = evaluate(model, loader(records), device, max_tokens=args.max_decode)
        results[split] = metrics
        (output / f"predictions_{split}.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
        print(f"FINAL {split}: {json.dumps(metrics)}", flush=True)
    # Controls use an explicit prefix of the test split; selection is independent of predictions.
    control_loader = loader(read_records("test")[:256])
    for mode in ["normal", "blank", "shuffled"]:
        metrics, _ = evaluate(model, control_loader, device, mode=mode, max_tokens=args.max_decode)
        results[f"control_{mode}"] = metrics
        print(f"CONTROL {mode}: exact={metrics['exact_match']:.3f}, similarity={metrics['normalized_similarity']:.3f}", flush=True)
    if device.type == "cuda":
        results["peak_gpu_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved: {output}", flush=True)


if __name__ == "__main__":
    main()
