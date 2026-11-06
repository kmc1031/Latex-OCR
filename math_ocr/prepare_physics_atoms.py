"""Add isolated symbols/short quantities so one-symbol input can end at EOS."""
import hashlib
import json
from pathlib import Path
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image
from physics_syntax import GLYPHS,canonical,parse,tokenize
from prepare_data import preprocess,tags

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data_physics'


def main():
    rows=[json.loads(s) for s in (DATA/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[r for r in rows if r.get('family')!='atomic']
    forbidden=set()
    for split in ['val','test','physics_laws','real_val','real_test']:
        for line in (DATA/(split+'.jsonl')).read_text(encoding='utf-8').splitlines():
            forbidden.add(repr(canonical(parse(json.loads(line)['tokens']))))
    rng=random.Random(814);renderer=MathTextParser('agg')
    props={f:FontProperties(size=26,math_fontfamily=f) for f in ['stix','cm','dejavusans']}
    formulas=list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')+list(GLYPHS)
    formulas += [str(rng.randrange(10,1000)) for _ in range(100)]
    formulas += [rng.choice(list(GLYPHS))+' '+rng.choice(list('rtmvEF')) for _ in range(140)]
    seen=set();added=[]
    for text in formulas:
        tokens=tokenize(text);identity=repr(canonical(parse(tokens)))
        if identity in seen or identity in forbidden:continue
        seen.add(identity);paths=[]
        for font,prop in props.items():
            rendered=renderer.parse('$'+' '.join(tokens)+'$',dpi=110,prop=prop)
            image=preprocess(Image.fromarray(255-np.asarray(rendered.image)),max_width=512)
            rel=f'images/atomic_{len(added):05d}_{font}.png';image.save(DATA/rel);paths.append(rel)
            if font=='stix':
                size=image.width;sha=hashlib.sha256(np.asarray(image).tobytes()+str(image.size).encode()).hexdigest()
        added.append({'image':paths[0],'image_variants':paths[1:],'tokens':tokens,'latex':' '.join(tokens),
                      'tags':tags(tokens),'family':'atomic','width':size,'image_sha256':sha,'source_filename':f'atomic_{len(added)}'})
    (DATA/'train.jsonl').write_text('\n'.join(json.dumps(r) for r in rows+added)+'\n',encoding='utf-8')
    print(json.dumps({'isolated_and_short_training_examples':len(added),'train_total':len(rows)+len(added)}),flush=True)


if __name__=='__main__':main()
