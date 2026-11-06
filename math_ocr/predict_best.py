"""Use the selected notation/model package on one cropped input image."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from model import FormulaOCR,ModelConfig
from prepare_data import preprocess
from slash_semantics import parse

ROOT=Path(__file__).resolve().parent


def default_checkpoint():
    physics=ROOT/'physics/best_model.pt'
    return physics if physics.is_file() else ROOT/'comparison/best_model.pt'


def parse_output(tokens,package):
    if package.get('syntax')=='physics-v1':
        from physics_syntax import result
        return result(tokens)
    try:
        tree=parse(tokens)
        return tree,latex(tree)
    except (ValueError,RecursionError):return None,None


def latex(tree):
    kind=tree[0]
    if kind=="symbol":return tree[1]
    if kind=="div":return r"\frac{"+latex(tree[1])+"}{"+latex(tree[2])+"}"
    if kind=="sqrt":return r"\sqrt{"+latex(tree[1])+"}"
    if kind in ["sup","sub"]:
        return "{"+latex(tree[1])+"}"+("^" if kind=="sup" else "_")+"{"+latex(tree[2])+"}"
    if kind in ["negative","positive"]:return ("-" if kind=="negative" else "+")+"("+latex(tree[1])+")"
    symbols={"add":"+","subtraction":"-","eq":"="}
    return "("+latex(tree[1])+symbols[kind]+latex(tree[2])+")"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image",type=Path)
    parser.add_argument("--checkpoint",type=Path,default=default_checkpoint())
    parser.add_argument("--latex",action="store_true",help="Convert recognized expression structure to LaTeX fractions")
    parser.add_argument("--json",action="store_true",help="Include syntax and expected input notation")
    parser.add_argument("--prepared",action="store_true",help="Input is already the saved 64px preprocessed image")
    args=parser.parse_args()
    torch.set_num_threads(4)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    package=torch.load(args.checkpoint,map_location=device,weights_only=False)
    assert package["model_type"]=="seq2seq", "Only CNN/LSTM encoder-decoder models are supported"
    model=FormulaOCR(ModelConfig(**package["config"])).to(device).eval()
    model.load_state_dict(package["model"])
    with Image.open(args.image) as source:
        if package.get('preprocessing')=='handwriting-v1':
            from handwriting_data import preprocess_handwriting
            processor=preprocess_handwriting
        else:processor=preprocess
        image=source.convert("L") if args.prepared else processor(source,height=package['config']['height'],max_width=package.get('max_width',384))
    if image.height!=package['config']['height'] or image.width>package.get('max_width',384) or image.width%4:
        raise ValueError("Prepared image dimensions do not match this checkpoint")
    tensor=torch.from_numpy((255-np.array(image)).astype(np.float32)/255).unsqueeze(0).unsqueeze(0).to(device)
    from decoding import generate
    ids=generate(model,package,tensor,torch.tensor([image.width]))[0].cpu().tolist()
    if 2 in ids:ids=ids[:ids.index(2)]
    tokens=[package["vocab"][i] for i in ids]
    result=" ".join(tokens)
    tree,converted=parse_output(tokens,package)
    if args.json:
        print(json.dumps({"prediction":result,"latex":converted,"syntax_valid":tree is not None,
            "expected_input":package["input_notation"],"model":package["selected_variant"]},ensure_ascii=False))
    elif args.latex:
        if converted is None:raise ValueError("Prediction could not be parsed into a valid expression")
        print(converted)
    else:print(result)


if __name__=="__main__":main()
