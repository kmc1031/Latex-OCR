"""CROHME 2013 image/box/MathML supervision, with formula-disjoint validation."""
from collections import Counter
import hashlib,json,math,random,zipfile
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image,ImageDraw
from common import ROOT,crop_symbol
from physics_syntax import tokenize,parse,canonical

XMLID='{http://www.w3.org/XML/1998/namespace}id'
ALIASES={'\\lt':'<','\\gt':'>','\\le':'\\leq','\\ge':'\\geq','\\ne':'\\neq','\\rightarrow':'\\rightarrow','sqrt':'\\sqrt','sin':'\\sin','cos':'\\cos','tan':'\\tan','log':'\\log','lim':'\\lim'}
def label(text):return ALIASES.get(text.strip(),text.strip())

def read_ink(raw):
    root=ET.fromstring(raw);traces={}
    for t in root.findall('{*}trace'):
        p=np.array([[float(v) for v in s.strip().split()[:2]] for s in t.text.strip().split(',')],np.float32)
        if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():raise ValueError('Invalid trace')
        traces[t.attrib['id']]=p
    glyphs=[];lookup={}
    for g in root.findall('.//{*}traceGroup'):
        refs=[v.attrib['traceDataRef'] for v in g.findall('{*}traceView')]
        if not refs:continue
        a=g.find('{*}annotation');link=g.find('{*}annotationXML')
        if a is None:raise ValueError('Missing symbol label')
        key=link.attrib.get('href') if link is not None else g.attrib.get(XMLID)
        points=np.concatenate([traces[r] for r in refs]);box=np.r_[points.min(0),points.max(0)]
        lookup[key]=len(glyphs);glyphs.append({'label':label(a.text),'box':box,'refs':refs})
    if not glyphs:raise ValueError('No segmentation ground truth')
    edges=[]
    def visit(n):
        tag=n.tag.rsplit('}',1)[-1];children=list(n);key=n.attrib.get(XMLID)
        if tag in ['mi','mn','mo','mtext']:
            if key not in lookup:raise ValueError('Unlinked MathML leaf')
            index=lookup[key];return index,index,glyphs[index]['label']
        if tag in ['math','mrow','mstyle','semantics']:
            parts=[visit(c) for c in children]
            if not parts:raise ValueError('Empty row')
            for a,b in zip(parts,parts[1:]):edges.append([a[1],b[0],'Right'])
            return parts[0][0],parts[-1][1],''.join(p[2]+' ' for p in parts).strip()
        if tag in ['msup','msub','msubsup','munder','mover','munderover']:
            parts=[visit(c) for c in children];base=parts[0]
            ops=['Sub','Sup'] if tag in ['msubsup','munderover'] else ['Sup' if tag in ['msup','mover'] else 'Sub']
            tex=base[2]
            for child,relation in zip(parts[1:],ops):
                edges.append([base[1],child[0],relation]);tex+=('^' if relation=='Sup' else '_')+'{'+child[2]+'}'
            return base[0],base[1],tex
        if tag in ['mfrac','msqrt']:
            if key not in lookup:raise ValueError('Unlinked fraction/radical')
            index=lookup[key];parts=[visit(c) for c in children]
            glyphs[index]['label']='\\frac' if tag=='mfrac' else '\\sqrt'
            if tag=='mfrac':
                edges.extend([[index,parts[0][0],'Above'],[index,parts[1][0],'Below']]);tex='\\frac{'+parts[0][2]+'}{'+parts[1][2]+'}'
            else:
                edges.append([index,parts[0][0],'Inside']);tex='\\sqrt{'+''.join(p[2] for p in parts)+'}'
            return index,index,tex
        raise ValueError('Unsupported MathML '+tag)
    mathml=root.find('.//{*}math')
    if mathml is None:raise ValueError('No MathML')
    _,_,tex=visit(mathml)
    return traces,glyphs,edges,tex

def render(traces,glyphs):
    allpoints=np.concatenate(list(traces.values()));low=allpoints.min(0);extent=np.maximum(allpoints.max(0)-low,1e-4)
    heights=[g['box'][3]-g['box'][1] for g in glyphs if g['box'][3]>g['box'][1]]
    scale=min(48/max(np.median(heights),1e-4),1008/extent[0],240/extent[1])
    size=tuple(int(math.ceil((v*scale+16)/4))*4 for v in extent)
    image=Image.new('L',size,255);factor=3
    def draw_refs(refs,region=None):
        offset=np.array(region[:2]) if region is not None else np.zeros(2)
        shape=(region[2]-region[0],region[3]-region[1]) if region is not None else size
        canvas=Image.new('L',(shape[0]*factor,shape[1]*factor),255);draw=ImageDraw.Draw(canvas)
        for key in refs:
            p=(traces[key]-low)*scale+8-offset;xy=[tuple(v*factor) for v in p]
            if len(xy)>1:draw.line(xy,fill=0,width=5,joint='curve')
            for x,y in [xy[0],xy[-1]]:draw.ellipse((x-2.5,y-2.5,x+2.5,y+2.5),fill=0)
        return canvas.resize(shape,Image.Resampling.LANCZOS)
    image=draw_refs(traces.keys());isolated=[]
    for g in glyphs:
        g['box']=np.r_[(g['box'][:2]-low)*scale+6,(g['box'][2:]-low)*scale+10].tolist()
        region=[int(np.floor(v)) if i<2 else int(np.ceil(v)) for i,v in enumerate(g['box'])]
        isolated.append(crop_symbol(draw_refs(g['refs'],region)))
        del g['refs']
    return image,isolated

def main():
    dest=ROOT/'data';(dest/'images').mkdir(parents=True,exist_ok=True)
    visible=[t for t in json.loads((ROOT.parent/'data_physics/vocab.json').read_text()) if t not in ['<PAD>','<BOS>','<EOS>','<UNK>','{','}','^','_']]
    # The visible symbol inventory is fixed before reading evaluation examples.
    vocab=visible+['[',']','<','>','|']
    vocab=list(dict.fromkeys(vocab));allowed=set(vocab)
    candidates=[];rejected=Counter();z=zipfile.ZipFile(ROOT/'raw/crohme2014.zip')
    for name in z.namelist():
        is_train='/CROHME2013_data/TrainINKML/' in name;is_test='/CROHME2013_data/TestINKMLGT/' in name
        if not name.endswith('.inkml') or not (is_train or is_test):continue
        raw=z.read(name)
        try:
            traces,glyphs,edges,tex=read_ink(raw)
            if any(g['label'] not in allowed for g in glyphs):raise ValueError('Unsupported glyph')
            tokens=tokenize(tex);identity=repr(canonical(parse(tokens)))
            if len(glyphs)>60:raise ValueError('Too many symbols')
        except (ValueError,ET.ParseError,KeyError,TypeError,IndexError) as exc:rejected[str(exc)]+=1;continue
        points=np.concatenate(list(traces.values()));coords=(points-points.min(0))/max(float(np.ptp(points,axis=0).max()),1e-8)
        inkhash=hashlib.sha256(np.round(coords,4).tobytes()).hexdigest()
        candidates.append((name,is_test,traces,glyphs,edges,tex,tokens,identity,inkhash))
    test_hashes={r[-1] for r in candidates if r[1]};seen=set();splits={s:[] for s in ['train','val','test']};glyph_arrays=[];glyph_rows=[]
    for name,is_test,traces,glyphs,edges,tex,tokens,identity,inkhash in candidates:
        if inkhash in seen or (not is_test and inkhash in test_hashes):rejected['duplicate_ink']+=1;continue
        seen.add(inkhash)
        bucket=int(hashlib.sha256(identity.encode()).hexdigest()[:8],16)%100
        split='test' if is_test else 'val' if bucket<15 else 'train'
        image,isolated=render(traces,glyphs);key=hashlib.sha256(name.encode()).hexdigest()[:16];rel=f'images/{key}.png';image.save(dest/rel)
        row={'id':key,'image':rel,'source':name,'ink_hash':inkhash,'formula_identity':identity,'latex':tex,'tokens':tokens,'glyphs':glyphs,'edges':edges,'size':list(image.size)}
        row['glyph_indexes']=[]
        for g,clean in zip(glyphs,isolated):
            index=len(glyph_rows);row['glyph_indexes'].append(index)
            glyph_arrays.append(np.stack([crop_symbol(image,g['box']),clean]))
            glyph_rows.append({'expression':key,'split':split,'label':g['label']})
        splits[split].append(row)
        if sum(map(len,splits.values()))%1000==0:print({s:len(r) for s,r in splits.items()},flush=True)
    train_formula={r['formula_identity'] for r in splits['train']}
    assert not train_formula&{r['formula_identity'] for r in splits['val']}
    for split,rows in splits.items():
        for r in rows:r['seen_formula']=r['formula_identity'] in train_formula
        (dest/f'{split}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n',encoding='utf-8')
    np.save(dest/'glyphs.npy',np.stack(glyph_arrays));(dest/'glyphs.json').write_text(json.dumps(glyph_rows),encoding='utf-8')
    (dest/'vocab.json').write_text(json.dumps(vocab),encoding='utf-8')
    summary={'source':'CROHME 2013 annotated partitions in pinned public extractor mirror','splits':{s:len(r) for s,r in splits.items()},
       'glyph_counts':dict(Counter(r['split'] for r in glyph_rows)),'vocabulary':len(vocab),'rejected':dict(rejected),
       'train_val_formula_overlap':0,'test_seen_formulas':sum(r['seen_formula'] for r in splits['test']),
       'scope':'Supported glyph/MathML subset, rasterized digital ink; not full CROHME benchmark or paper photos.'}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
