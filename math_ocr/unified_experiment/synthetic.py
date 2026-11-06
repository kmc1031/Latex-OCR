"""Explicit TRAINING-ONLY font and clean-glyph composites, never photographs."""
import os,json,random,string,hashlib
import numpy as np
from PIL import Image
from engine import ROOT,CLASSES
from syntax import normalize,tokenize
os.environ['MPLCONFIGDIR']=str(ROOT/'plot_cache')

def prepare():
    target=ROOT/'synthetic_manifest.json'
    parent_audit=json.loads((ROOT/'symbol_parent_audit.json').read_text(encoding='utf-8')) if (ROOT/'symbol_parent_audit.json').exists() else {}
    banned=set(parent_audit.get('excluded_auxiliary_indices',[]))
    audit_hash=hashlib.sha256(json.dumps(sorted(banned)).encode()).hexdigest()
    stamp=ROOT/'synthetic_source_audit.json'
    if target.exists() and stamp.exists() and json.loads(stamp.read_text())['source_exclusion_hash']==audit_hash:return json.loads(target.read_text(encoding='utf-8'))
    from matplotlib.mathtext import MathTextParser
    from matplotlib.font_manager import FontProperties
    rng=random.Random(42);aux=np.load(ROOT/'auxiliary_glyphs.npz');x=aux['images'];labels=aux['labels'];pool={c:[int(j) for j in np.flatnonzero(labels==i) if int(j) not in banned] for i,c in enumerate(CLASSES)}
    missing=[c for c,ids in pool.items() if not ids]
    if missing:raise ValueError('No true source glyphs: '+repr(missing))
    real=json.loads((ROOT/'prepared_manifest.json').read_text(encoding='utf-8'));photos=json.loads((ROOT/'capture_plan.json').read_text(encoding='utf-8'))['records'];seen={r['expression_identity'] for r in real if r['split']!='train'}|{normalize(r['planned_target_tokens'])[1] for r in photos if r['split']!='train'};records=[];parser=MathTextParser('agg')
    for kind,count in [('basic',600),('sup',300),('sub',300),('fraction',400)]:
        for k in range(count):
            while True:
                r,a,b=rng.choices(string.ascii_letters,k=3);n=rng.randrange(1,20);op=rng.choice(['+','-','\\times '])
                raw=f'{r}={a}{op}{b}' if kind=='basic' else f'{r}={a}^{{{n}}}+{b}' if kind=='sup' else f'{r}={a}_{{{n}}}+{b}' if kind=='sub' else f'{r}=\\frac{{{a}+{n}}}{{{b}}}'
                ts,identity=normalize(tokenize(raw))
                if identity not in seen:seen.add(identity);break
            for style in (['vertical','slash'] if kind=='fraction' else ['normal']):
                prompt=f'{r}=({a}+{n})/({b})' if style=='slash' else raw
                usefont=k%2==0;ids=[]
                if usefont:
                    rendered=parser.parse('$'+prompt+'$',dpi=rng.randrange(90,130),prop=FontProperties(size=rng.randrange(24,38),math_fontfamily=rng.choice(['stix','dejavusans','cm'])))
                    im=Image.fromarray(255-np.asarray(rendered.image))
                else:
                    im=Image.new('L',(900,180),255)
                    def row(tokens,start,y,size=44):
                        at=start
                        for c in tokens:
                            idx=rng.choice(pool[c]);ids.append(idx);p=Image.fromarray(np.clip(255*(1-x[idx,0]),0,255).astype('uint8'));bb=Image.fromarray((np.asarray(p)<180).astype('uint8')*255).getbbox();p=p.crop(bb) if bb else p
                            scale=min(size/p.height,(size*1.2)/p.width);p=p.resize((max(1,round(p.width*scale)),max(1,round(p.height*scale))))
                            if c=='.':p=p.resize((4,4))
                            im.paste(p,(int(at),int(y+size-p.height)));at+=p.width+rng.randrange(4,13)
                        return at
                    y=rng.randrange(55,70);end=row([r,'='],10,y)
                    if kind=='fraction' and style=='vertical':
                        start=end+rng.randrange(5,15);right=row([a,'+']+list(str(n)),start,8,35);width=right-start+rng.randrange(8,20);bar=Image.new('L',(width,rng.randrange(2,5)),0);im.paste(bar,(start-3,65));row([b],start+width//2-12,90,38)
                    elif kind in ['sup','sub']:
                        end=row([a],end,y);end=row(list(str(n)),end,15 if kind=='sup' else y+44,23);row(['+',b],end,y)
                    else:row([t for t in tokenize(prompt)[2:] if t in CLASSES],end,y)
                    bb=Image.fromarray((np.asarray(im)<220).astype('uint8')*255).getbbox();im=im.crop(bb);canvas=Image.new('L',(im.width+24,im.height+24),255);canvas.paste(im,(12,12));im=canvas
                rel=f'data/synthetic_v3/{kind}_{k:04d}_{style}.png';(ROOT/rel).parent.mkdir(parents=True,exist_ok=True);im.save(ROOT/rel);sid=hashlib.sha256((identity+style).encode()).hexdigest()[:16]
                records.append({'sample_id':sid,'split':'train','source':'synthetic_font' if usefont else 'synthetic_clean_glyph','kind':kind,'style':style,'image':rel,'tokens':ts,'expression_identity':identity,'glyph_aux_indices':ids})
    assert len(records)==2000;target.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8');stamp.write_text(json.dumps({'source_exclusion_hash':audit_hash,'banned_auxiliary_indices':len(banned)}),encoding='utf-8');return records
