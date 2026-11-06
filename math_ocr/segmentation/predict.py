import argparse,json
from PIL import Image,ImageDraw
from pipeline import Pipeline

def main():
    ap=argparse.ArgumentParser();ap.add_argument('image');ap.add_argument('--method',choices=['detector','components']);ap.add_argument('--prepared',action='store_true');ap.add_argument('--overlay');args=ap.parse_args()
    model=Pipeline();result=model.recognize(Image.open(args.image),prepared=args.prepared,method=args.method)
    image=result.pop('prepared')
    if args.overlay:
        image=image.convert('RGB');draw=ImageDraw.Draw(image)
        for box,label in zip(result['boxes'],result['labels']):draw.rectangle(box,outline='#20875a',width=1);draw.text((box[0],max(0,box[1]-10)),label,fill='#155637')
        image.save(args.overlay)
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
