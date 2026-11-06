"""The frozen image-only inference path, shared with the local web UI."""
import argparse,json,torch
from PIL import Image
from web import load_model

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('image');args=ap.parse_args();m=load_model();print(json.dumps(m.recognize(Image.open(args.image)),ensure_ascii=False,indent=2))
