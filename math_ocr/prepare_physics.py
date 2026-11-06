"""Build formula-disjoint synthetic physics notation and a held-out law suite."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

import matplotlib
matplotlib.use('Agg')
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image,ImageFilter

from physics_syntax import GREEK,parse,tokenize,canonical
from prepare_data import SPECIAL,preprocess,tags
from slash_semantics import slash_tokens

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'data_physics'
MAX_WIDTH=512
# All of these exact expressions, in both layouts and all fonts, are held out.
# They are not used to choose checkpoints or tune the decoder.
LAWS=[
 ('mechanics',r'F=ma'),('mechanics',r'p=mv'),('mechanics',r'E=mc^{2}'),
 ('mechanics',r'K=\frac{1}{2}mv^{2}'),('mechanics',r'U=mgh'),
 ('mechanics',r'F=G\frac{m_{1}m_{2}}{r^{2}}'),('mechanics',r'v=v_{0}+at'),
 ('mechanics',r'x=x_{0}+v_{0}t+\frac{1}{2}at^{2}'),('mechanics',r'v^{2}=v_{0}^{2}+2ax'),
 ('mechanics',r'a=\frac{v^{2}}{r}'),('mechanics',r'\tau=I\alpha'),
 ('mechanics',r'\vec{F}=m\vec{a}'),('mechanics',r'\vec{L}=\vec{r}\times\vec{p}'),
 ('mechanics',r'W=\int_{0}^{x}Fdx'),('mechanics',r'\rho=\frac{m}{V}'),
 ('waves',r'\omega=2\pi f'),('waves',r'v=f\lambda'),('waves',r'T=\frac{1}{f}'),
 ('waves',r'T=2\pi\sqrt{\frac{l}{g}}'),('waves',r'v=\omega r'),
 ('waves',r'y=A\sin(\omega t+\phi)'),('waves',r'k=\frac{2\pi}{\lambda}'),
 ('waves',r'\theta=\omega t'),('waves',r'E=\frac{hc}{\lambda}'),
 ('em',r'V=IR'),('em',r'P=IV'),('em',r'C=\frac{Q}{V}'),
 ('em',r'F=\frac{q_{1}q_{2}}{4\pi\epsilon_{0}r^{2}}'),
 ('em',r'B=\frac{\mu_{0}I}{2\pi r}'),('em',r'C=\frac{\epsilon_{0}A}{d}'),
 ('em',r'\Phi=BA\cos\theta'),('em',r'\vec{F}=q\vec{v}\times\vec{B}'),
 ('em',r'\epsilon=-\frac{d\Phi}{dt}'),('em',r'E=\frac{1}{2}CV^{2}'),
 ('em',r'c=\frac{1}{\sqrt{\mu_{0}\epsilon_{0}}}'),('em',r'\nabla\cdot\vec{B}=0'),
 ('thermal',r'PV=nRT'),('thermal',r'Q=mc\Delta T'),('thermal',r'\Delta U=Q-W'),
 ('thermal',r'E=\frac{3}{2}kT'),('thermal',r'\eta=1-\frac{T_{c}}{T_{h}}'),
 ('modern',r'E=hf'),('modern',r'\lambda=\frac{h}{p}'),
 ('modern',r'\gamma=\frac{1}{\sqrt{1-\frac{v^{2}}{c^{2}}}}'),
 ('modern',r'\Delta x\Delta p\geq\frac{\hbar}{2}'),
 ('measurement',r'g=9.81\mathrm{m}/\mathrm{s}^{2}'),
 ('measurement',r'c=3.00\times10^{8}\mathrm{m}/\mathrm{s}'),
 ('measurement',r'R=10\Omega'),('measurement',r'f=50\mathrm{Hz}'),
 ('measurement',r'\mu=1.25\times10^{-3}'),
]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=int,default=9000);ap.add_argument('--val',type=int,default=500);ap.add_argument('--test',type=int,default=600);args=ap.parse_args()
    DEST.mkdir(exist_ok=True);(DEST/'images').mkdir(exist_ok=True)
    rng=random.Random(20261101);renderer=MathTextParser('agg')
    props={f:FontProperties(size=26,math_fontfamily=f) for f in ['stix','dejavusans','cm','dejavuserif']}
    forbidden=set();used_pixels={};used_trees={};render_failures=0
    def key(tokens):return repr(canonical(parse(tokens)))
    for _,s in LAWS:
        forbidden.add(key(tokenize(s)));forbidden.add(key(slash_tokens(tokenize(s))))
    # Exclude all previous evaluation identities; keep the old suite independent.
    for split in ['val','test','long_test']:
        for line in (ROOT/f'comparison/views/D_slash_slash/{split}.jsonl').read_text().splitlines():
            row=json.loads(line);forbidden.add(key(row['tokens']));forbidden.add(key(row['reference_tex']))
    symbols=list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')+['\\'+s for s in GREEK]+['\\hbar','\\ell','\\nabla','\\partial','\\infty']
    functions=['\\sin','\\cos','\\tan','\\ln','\\log','\\exp']
    def number():
        return str(rng.randint(0,99)) if rng.random()<.6 else f'{rng.randint(0,99)}.{rng.randint(0,99):02d}'
    def atom(script=True):
        if rng.random()<.2:return rng.choice(['0','1','2','3','4','5','6','7','8','9',number()])
        value=rng.choice(symbols)
        if script and rng.random()<.28:value+=rng.choice(['_{0}','_{1}','_{2}','_{x}','_{i}','^{2}','^{3}','^{-1}'])
        return value
    def product():return ' '.join(atom() for _ in range(rng.choice([1,1,2,2,3])))
    def binary():return product()+rng.choice(['+','-'])+product()
    families=['symbols','linear','product','scripts','fraction','function','root','vector','units','calculus']
    def generate(family):
        if family=='symbols':return atom(False)+rng.choice(['=','+','-'])+atom(False)
        if family=='linear':return atom()+rng.choice(['=','+','-','\\approx','\\neq','\\leq','\\geq','\\propto','\\pm'])+binary()
        if family=='product':return atom(False)+'='+product()
        if family=='scripts':return atom()+rng.choice(['=','+','-'])+atom()+rng.choice(['','+'+atom()])
        if family=='fraction':return atom(False)+'=\\frac{'+rng.choice([product(),binary()])+'}{'+product()+'}'
        if family=='function':return atom(False)+'='+atom(False)+rng.choice(functions)+'('+rng.choice([product(),binary()])+')'
        if family=='root':return atom(False)+'=\\sqrt{'+rng.choice([product(),binary()])+'}'
        if family=='vector':
            a,b,c=[rng.choice(list('FEABvrpL')) for _ in range(3)]
            return rng.choice(['\\vec{'+a+'}=\\vec{'+b+'}'+rng.choice(['+','\\times','\\cdot'])+'\\vec{'+c+'}',
                               '\\'+rng.choice(['hat','bar','dot','ddot'])+'{'+a+'}='+product()])
        if family=='units':
            n=number()
            unit=rng.choice(['m','s','kg','Hz','N','J','W','V','A','C','Pa','K','mol'])
            return atom(False)+'='+n+rng.choice(['','\\times10^{'+rng.choice(['2','3','6','-2','-3','-6'])+'}'])+'\\mathrm{'+unit+'}'+rng.choice(['','/\\mathrm{s}','^{2}'])
        op=rng.choice(['\\int','\\sum','\\prod','\\oint'])
        return atom(False)+'='+op+'_{'+rng.choice(['0','1','i'])+'}^{'+rng.choice(['t','n','3'])+'}'+product()
    def render(tokens,font):
        r=renderer.parse('$'+' '.join(tokens)+'$',dpi=110,prop=props[font])
        return preprocess(Image.fromarray(255-np.asarray(r.image)),max_width=MAX_WIDTH)
    def record(tokens,split,index,family,font='stix',variant=False):
        image=render(tokens,font);sha=hashlib.sha256(np.asarray(image).tobytes()+str(image.size).encode()).hexdigest()
        if sha in used_pixels and used_pixels[sha]!=repr(parse(tokens)):raise ValueError('Conflicting pixel labels')
        used_pixels[sha]=repr(parse(tokens))
        name=f'images/{split}_{index:05d}_{font}.png';image.save(DEST/name)
        return {'image':name,'tokens':tokens,'latex':' '.join(tokens),'tags':tags(tokens),'family':family,
                'width':image.width,'image_sha256':sha,'source_filename':f'physics_{split}_{index}'}
    splits={};start=time.perf_counter()
    for split,size in [('train',args.train),('val',args.val),('test',args.test)]:
        rows=[]
        while len(rows)<size:
            family=families[len(rows)%len(families)]
            try:
                tokens=tokenize(generate(family))
                if not 3<=len(tokens)<=52:continue
                identity=key(tokens)
                if identity in forbidden or identity in used_trees:continue
                # Keep both fraction notations in training; no cross-split identities.
                if '\\frac' in tokens and rng.random()<.5:tokens=slash_tokens(tokens);identity=key(tokens)
                if identity in forbidden or identity in used_trees:continue
                row=record(tokens,split,len(rows),family)
                if split=='train':
                    font=rng.choice(['dejavusans','cm']);other=record(tokens,split,len(rows),family,font)
                    row['image_variants']=[other['image']]
                used_trees[identity]=split;rows.append(row)
            except (ValueError,RuntimeError):render_failures+=1;continue
            if len(rows)%1000==0:print(f'{split}: {len(rows)}/{size} ({time.perf_counter()-start:.1f}s)',flush=True)
        splits[split]=rows
    # Replay earlier short formulas, with unchanged targets; exclude old nested
    # expressions from replay because their long sequences dominate this small model.
    replay=[]
    for line in (ROOT/'comparison/views/P_practical/train.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['pair_id'].startswith('nested_'):continue
        if key(row['tokens']) in used_trees:continue
        row['image']=os.path.relpath((ROOT/'comparison/views/P_practical'/row['image']).resolve(),DEST).replace('\\','/')
        row['image_variants']=[os.path.relpath((ROOT/'comparison/views/P_practical'/p).resolve(),DEST).replace('\\','/') for p in row.get('image_variants',[])]
        row['family']='legacy_replay';replay.append(row)
    splits['train']+=replay
    # Validation/test font and capture shifts reuse identities within the same
    # partition only. DejaVu Serif is never in the training font bank.
    for split in ['val','test']:
        fontrows=[];capturerows=[]
        for i,row in enumerate(splits[split]):
            fontrows.append(record(row['tokens'],split+'_font',i,row['family'],'dejavuserif'))
            image=Image.open(DEST/row['image']).rotate((.6 if i%2 else -.6),resample=Image.Resampling.BICUBIC,fillcolor=255).filter(ImageFilter.GaussianBlur(.3))
            rel=f'images/{split}_capture_{i:05d}.png';image.save(DEST/rel)
            capturerows.append({**row,'image':rel})
        splits[split+'_font']=fontrows;splits[split+'_capture']=capturerows
    laws=[]
    for i,(domain,formula) in enumerate(LAWS):
        original=tokenize(formula)
        for layout in ['stack','slash'] if '\\frac' in original else ['native']:
            tokens=slash_tokens(original) if layout=='slash' else original
            for font in ['stix','cm','dejavuserif']:
                row=record(tokens,'laws_'+layout,len(laws),domain,font)
                row.update({'law_id':i,'layout':layout,'font':font,'formula':formula})
                laws.append(row)
    splits['physics_laws']=laws
    vocab=SPECIAL+sorted({t for row in splits['train'] for t in row['tokens']})
    # Some relations/constant names are infrequent in random sampling: expose
    # each supported token through short training examples, never held-out laws.
    missing={t for rows in splits.values() for row in rows for t in row['tokens']}-set(vocab)
    for j,t in enumerate(sorted(missing)):
        if t in ['\\geq','\\leq','\\neq','\\propto','\\pm']:tokens=tokenize('x'+t+'y')
        else:tokens=tokenize('x='+t)
        for k in range(16):
            tokens[0]=rng.choice(list('abcdefghijklmnopqrstuvwxyz'))
            if key(tokens) in forbidden or key(tokens) in used_trees:continue
            row=record(tokens,'coverage',len(splits['train']),'coverage',rng.choice(['stix','cm','dejavusans']))
            splits['train'].append(row);used_trees[key(tokens)]='train'
    vocab=SPECIAL+sorted({t for row in splits['train'] for t in row['tokens']})
    for split,rows in splits.items():
        assert not {t for row in rows for t in row['tokens']}-set(vocab),(split,'unknown token')
        (DEST/f'{split}.jsonl').write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in rows)+'\n',encoding='utf-8')
    (DEST/'vocab.json').write_text(json.dumps(vocab,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'seed':20261101,'height':64,'max_width':MAX_WIDTH,'vocab_size':len(vocab),'vocabulary':vocab[4:],
             'splits':{split:len(rows) for split,rows in splits.items()},'replay_samples':len(replay),
             'training_fonts':['stix','cm','dejavusans'],'heldout_font':'dejavuserif',
             'physics_laws_held_out':len(LAWS),'render_rejections':render_failures,
             'scope':'Synthetic printed physics notation, not handwriting or a public OCR benchmark.',
             'sources':['https://matplotlib.org/stable/tutorials/text/mathtext.html',
                        'https://openstax.org/books/university-physics-volume-2/pages/8-key-equations',
                        'https://openstax.org/books/university-physics-volume-2/pages/12-key-equations']}
    (DEST/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
