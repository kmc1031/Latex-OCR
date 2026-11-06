"""Predict a cropped formula image using a locally trained checkpoint."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from model import FormulaOCR, ModelConfig
from prepare_data import preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "runs/pilot/best.pt")
    parser.add_argument("--max-tokens", type=int, default=72)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = FormulaOCR(ModelConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    image = preprocess(Image.open(args.image))
    tensor = torch.from_numpy((255-np.array(image)).astype(np.float32)/255).unsqueeze(0).unsqueeze(0).to(device)
    ids = model.generate(tensor, torch.tensor([image.width]), args.max_tokens)[0].tolist()
    if 2 in ids:
        ids = ids[:ids.index(2)]
    print(" ".join(ckpt["vocab"][i] for i in ids))


if __name__ == "__main__":
    main()
