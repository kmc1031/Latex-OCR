"""Add bounded real printed formulas from the already downloaded IM2LATEX export.

Uses official train/validation/test partitions, removes cross-partition formula
duplicates and all synthetic/law identities. This is a filtered subset, not the
full public benchmark and not a handwriting dataset.
"""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from physics_syntax import canonical,parse,tokenize,GREEK
from prepare_data import preprocess,tags

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data_physics'


def main():
    vocab=set(json.loads((DATA/'vocab.json').read_text(encoding='utf-8')))
    training=[json.loads(s) for s in (DATA/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    training=[r for r in training if r.get('family')!='real_printed']
    used=set();pixels=set()
    for split in ['train','val','test','physics_laws']:
        rows=training if split=='train' else [json.loads(s) for s in (DATA/(split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        for row in rows:used.add(repr(canonical(parse(row['tokens']))));pixels.add(row['image_sha256'])
    counts={};rejections={};allrows={}
    for source,size in [('train',1200),('val',200),('test',300)]:
        file=next((ROOT/'data/raw').glob(source+'-*.parquet'))
        rows=pq.read_table(file).to_pylist();random.Random(8173+len(source)).shuffle(rows);selected=[];rejected=Counter()
        for row in rows:
            raw=tokenize(row['formula'])
            aliases={'\\dfrac':'\\frac','\\tfrac':'\\frac','\\le':'\\leq','\\ge':'\\geq','\\ne':'\\neq'}
            tokens=[aliases.get(t,t) for t in raw if t not in ['\\left','\\right','\\,','\\!','\\;','\\:','\\quad','\\qquad']]
            if not 5<=len(tokens)<=52 or set(tokens)-vocab:rejected['unsupported_or_length']+=1;continue
            if not any(t.startswith('\\') and t[1:] in GREEK or len(t)==1 and t.isupper() for t in tokens):
                rejected['no_physics_symbols']+=1;continue
            try:
                key=repr(canonical(parse(tokens)))
                if key in used:rejected['duplicate_formula']+=1;continue
                image=preprocess(Image.open(io.BytesIO(row['image']['bytes'])),max_width=512)
            except (ValueError,OSError,RecursionError):rejected['invalid']+=1;continue
            sha=hashlib.sha256(np.asarray(image).tobytes()+str(image.size).encode()).hexdigest()
            if sha in pixels:rejected['duplicate_pixels']+=1;continue
            rel=f'images/real_{source}_{len(selected):05d}.png';image.save(DATA/rel)
            selected.append({'image':rel,'tokens':tokens,'latex':' '.join(tokens),'tags':tags(tokens),'family':'real_printed',
                             'width':image.width,'image_sha256':sha,'source_filename':row['filename']})
            used.add(key);pixels.add(sha)
            if len(selected)>=size:break
        allrows[source]=selected;counts[source]=len(selected);rejections[source]=dict(rejected)
        (DATA/f'real_{source}.jsonl').write_text('\n'.join(json.dumps(r) for r in selected)+'\n',encoding='utf-8')
        print(f'real {source}: {len(selected)}',flush=True)
        del rows
    training+=allrows['train']
    (DATA/'train.jsonl').write_text('\n'.join(json.dumps(r) for r in training)+'\n',encoding='utf-8')
    summary={'selected':counts,'rejections':rejections,'official_partitions_preserved':True,
             'claim_scope':'IM2LATEX filtered printed science notation, 5-52 tokens. Not the full benchmark.',
             'source':'https://huggingface.co/datasets/yuntian-deng/im2latex-100k',
             'provenance':'../data/raw/provenance.json','train_total':len(training)}
    (DATA/'real_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
