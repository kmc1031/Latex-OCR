"""A bounded human handwriting subset, retaining official data partitions.

No synthetic MathWriting expressions are used. Symbols come only from the
official training-derived symbol bank. Writer IDs are not published; this code
therefore makes no stronger writer-isolation claim than the original split.
"""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path,PurePosixPath
import random
import re
import tarfile
import time
import numpy as np
from PIL import Image

from handwriting_data import annotations,rasterize,preprocess_handwriting,capture
from physics_syntax import GREEK,canonical,parse,tokenize
from prepare_data import tags
from prepare_physics import LAWS

ROOT=Path(__file__).resolve().parent;DEST=ROOT/'data_handwriting'
CAPS={'train':{'greek_short':3000,'greek_long':1200,'other_short':8500,'other_long':2300},
      'valid':{'greek_short':200,'greek_long':100,'other_short':400,'other_long':100},
      'test':{'greek_short':200,'greek_long':100,'other_short':400,'other_long':100}}


def normalized(text):
    aliases={'\\le':'\\leq','\\ge':'\\geq','\\ne':'\\neq','\\dfrac':'\\frac','\\tfrac':'\\frac'}
    text=re.sub(r'\\operatorname\s*\{(sin|cos|tan|log|ln|exp)\}',lambda m:'\\'+m[1],text)
    return [aliases.get(t,t) for t in tokenize(text) if t not in ['\\left','\\right','\\,','\\!','\\;','\\:','\\quad','\\qquad']]


def main():
    DEST.mkdir(exist_ok=True);(DEST/'images').mkdir(exist_ok=True)
    vocab=json.loads((ROOT/'data_physics/vocab.json').read_text(encoding='utf-8'));allowed=set(vocab)
    archive=ROOT/'data_mathwriting/raw/mathwriting-2024.tgz'
    rng=random.Random(27132504);pools=defaultdict(list);seen=Counter();rejected=Counter();scanned=Counter()
    forbidden={repr(canonical(parse(tokenize(text)))) for _,text in LAWS}
    began=time.perf_counter();last=began
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('.inkml'):continue
            split=PurePosixPath(member.name).parts[-2]
            if split not in ['train','valid','test','symbols']:continue
            raw=tar.extractfile(member).read();scanned[split]+=1
            try:
                root,meta=annotations(raw);tokens=normalized(meta.get('normalizedLabel',meta.get('label','')))
                if not 1<=len(tokens)<=32 or set(tokens)-allowed:rejected['vocab_or_length']+=1;continue
                tree=parse(tokens)
                if split in ['train','symbols'] and repr(canonical(tree)) in forbidden:rejected['heldout_physics_law']+=1;continue
            except (ValueError,RecursionError):rejected['syntax']+=1;continue
            if split=='symbols':bucket='symbol:'+''.join(tokens);limit=20
            else:
                greek=any(t.startswith('\\') and t[1:] in GREEK for t in tokens)
                bucket=('greek_' if greek else 'other_')+('short' if len(tokens)<=20 else 'long')
                limit=CAPS[split][bucket]
            key=(split,bucket);seen[key]+=1
            entry=(member.name,tokens,raw)
            if len(pools[key])<limit:pools[key].append(entry)
            else:
                index=rng.randrange(seen[key])
                if index<limit:pools[key][index]=entry
            if time.perf_counter()-last>15:
                print(f'scan {dict(scanned)} eligible={sum(seen.values())} ({time.perf_counter()-began:.0f}s)',flush=True);last=time.perf_counter()
    rows_by_split={};hashes={};dropped=Counter();fontset=[]
    for split in ['train','symbols','valid','test']:
        entries=[entry for key,values in pools.items() if key[0]==split for entry in values];rng.shuffle(entries);rows=[]
        for index,(name,tokens,raw) in enumerate(entries):
            root,meta=annotations(raw)
            width=rng.uniform(1.0,2.1) if split in ['train','symbols'] else 1.5
            image=preprocess_handwriting(rasterize(root,width))
            sha=hashlib.sha256(np.asarray(image).tobytes()+str(image.size).encode()).hexdigest()
            if sha in hashes:dropped[split]+=1;continue
            hashes[sha]=split
            ink_id=PurePosixPath(name).stem
            if not re.fullmatch(r'[a-f0-9]{16}',ink_id):raise ValueError('Unexpected sample id')
            rel=f'images/{split}_{ink_id}.png';image.save(DEST/rel)
            rows.append({'image':rel,'tokens':tokens,'latex':' '.join(tokens),'image_sha256':sha,'width':image.width,
                         'family':'handwritten_symbol' if split=='symbols' else 'human_handwriting',
                         'original_split':split,'source_filename':ink_id,'ink_id':ink_id,'tags':tags(tokens),
                         'source':'MathWriting 2024','greek':any(t.startswith('\\') and t[1:] in GREEK for t in tokens)})
            if len(rows)%1000==0:print(f'render {split}: {len(rows)}/{len(entries)}',flush=True)
        rows_by_split[split]=rows
    # A small printed replay protects Greek/Latin coverage and the existing UI.
    printed=[json.loads(line) for line in (ROOT/'data_physics/train.jsonl').read_text(encoding='utf-8').splitlines()]
    rng.shuffle(printed);replay=[]
    for row in printed:
        if len(replay)>=2400:break
        # Store normalized replay copies in this dataset; source files stay immutable.
        image=Image.open(ROOT/'data_physics'/row['image']).convert('L')
        rel=f'images/printed_{len(replay):05d}.png';image.save(DEST/rel)
        replay.append({**row,'image':rel,'image_variants':[], 'family':'printed_replay','original_split':'printed_train'})
    rows_by_split['train']+=rows_by_split.pop('symbols')+replay
    rows_by_split['val']=rows_by_split.pop('valid')
    for split in ['val','test']:
        changed=[]
        for i,row in enumerate(rows_by_split[split]):
            image=capture(Image.open(DEST/row['image']),5197+i)
            rel=f'images/{split}_capture_{i:05d}.png';image.save(DEST/rel)
            sha=hashlib.sha256(np.asarray(image).tobytes()+str(image.size).encode()).hexdigest()
            changed.append({**row,'image':rel,'image_sha256':sha,'width':image.width})
        rows_by_split[split+'_capture']=changed
        rows_by_split[split+'_greek']=[r for r in rows_by_split[split] if r['greek']]
    oldval=[json.loads(line) for line in (ROOT/'data_physics/val.jsonl').read_text(encoding='utf-8').splitlines()]
    import os
    rows_by_split['val_print']=[{**r,'image':os.path.relpath((ROOT/'data_physics'/r['image']).resolve(),DEST).replace('\\','/')} for r in oldval]
    for split,rows in rows_by_split.items():
        (DEST/f'{split}.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
    (DEST/'vocab.json').write_text(json.dumps(vocab,indent=2),encoding='utf-8')
    summary={'source':'https://github.com/google-research/google-research/tree/master/mathwriting',
             'archive':'../data_mathwriting/raw/mathwriting-2024.tgz.json','license':'CC-BY-NC-SA-4.0',
             'seed':27132504,'max_tokens':32,'max_width':512,'height':64,
             'official_partitions_preserved':True,'source_synthetic_expressions_used':False,
             'scanned':dict(scanned),'eligible':{'/'.join(k):v for k,v in seen.items()},'rejected':dict(rejected),
             'splits':{s:len(rs) for s,rs in rows_by_split.items()},'duplicate_rasterizations_removed':dict(dropped),
             'train_family_counts':dict(Counter(r['family'] for r in rows_by_split['train'])),
             'scope':'Filtered, Greek-enriched 1-32 token digital-pen handwriting, rasterized here. Simulated capture shifts are not real paper photographs.'}
    (DEST/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
