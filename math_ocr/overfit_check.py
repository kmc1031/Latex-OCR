"""Verify that the architecture can memorize 64 actual formulas in free decoding.

This diagnoses trainability only and never substitutes for held-out evaluation.
"""
import json
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import train
from model import FormulaOCR, ModelConfig


def main():
    random.seed(43)
    np.random.seed(43)
    torch.manual_seed(43)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    vocab = json.loads((train.DATA / "vocab.json").read_text(encoding="utf-8"))
    records = train.read_records("train")[:64]
    dataset = train.FormulaDataset(records, vocab)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=train.collate)
    evaluation = DataLoader(dataset, batch_size=32, collate_fn=train.collate)
    model = FormulaOCR(ModelConfig(len(vocab), dropout=0.0)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    history = []
    start = time.perf_counter()
    for epoch in range(1, 241):
        model.train()
        for images, widths, tokens, _ in loader:
            images, tokens = images.to(device), tokens.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(images, widths, tokens[:, :-1])
            loss = criterion(pred.flatten(0,1), tokens[:, 1:].flatten())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        if epoch % 20 == 0:
            metrics, predictions = train.evaluate(model, evaluation, device)
            history.append({"epoch":epoch,"ce_last_batch":float(loss.detach()), **metrics})
            print(f"Memorization epoch {epoch}: CE={float(loss.detach()):.3f}, exact={metrics['exact_match']:.3f}, sim={metrics['normalized_similarity']:.3f}", flush=True)
            if metrics["exact_match"] >= 0.95:
                break
    output = train.ROOT / "runs" / "overfit_check"
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model":model.state_dict(),"config":model.config_dict(),"vocab":vocab}, output / "last.pt")
    (output / "metrics.json").write_text(json.dumps({"train_only":True,"samples":64,
        "seconds":time.perf_counter()-start,"history":history,"final":metrics},indent=2),encoding="utf-8")
    (output / "predictions_train.json").write_text(json.dumps(predictions,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
