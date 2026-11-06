"""Audit and cache true training glyphs; do not synthesize missing classes."""
import hashlib,json,tarfile
from collections import Counter
import numpy as np
from PIL import Image
from engine import ROOT,CLASSES,crop
from syntax import parse,tokenize
import sys
sys.path.insert(0,str(ROOT.parent))
from handwriting_data import annotations,rasterize

def prepare():
    source=ROOT.parent/'segmentation/data';rows=json.loads((source/'glyphs.json').read_text());images=np.load(source/'glyphs.npy',mmap_mode='r');manifest=json.loads((ROOT/'prepared_manifest.json').read_text(encoding='utf-8'));plans=json.loads((ROOT/'capture_plan.json').read_text(encoding='utf-8'))['records']
    forbidden={r['expression_identity'] for r in manifest if r['split']!='train'}|{repr(parse(r['planned_target_tokens'])) for r in plans if r['split']!='train'}
    parents={}
    for line in (source/'train.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        try:parents[r['id']]=repr(parse(r['tokens']))
        except ValueError:pass
    x=[];y=[];ids=[];seen=set();source_seen=set();counts=Counter();excluded=0
    def add(image,label,key):
        digest=hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        sourcekey=(key['source'].split(' ')[0],key.get('id',key.get('index')))
        if (label,digest) in seen or sourcekey in source_seen:return
        seen.add((label,digest));source_seen.add(sourcekey);x.append(crop(image));y.append(CLASSES.index(label));ids.append(key);counts[label]+=1
    for i,r in enumerate(rows):
        label='-' if r['label']=='\\frac' else r['label']
        if r['split']!='train' or label not in CLASSES:continue
        if parents.get(r['expression']) in forbidden:excluded+=1;continue
        add(Image.fromarray(images[i,1]),label,{'source':'CROHME train clean mask','index':i,'parent':r['expression']})
    d=ROOT.parent/'data_handwriting'
    for line in (d/'train.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r.get('family')=='handwritten_symbol' and len(r['tokens'])==1 and r['tokens'][0] in CLASSES:
            add(Image.open(d/r['image']).convert('L'),r['tokens'][0],{'source':'MathWriting training-derived symbol','id':r['ink_id']})
    # Include all raw symbols, not only the old 20-per-class prepared subset.
    with tarfile.open(ROOT.parent/'data_mathwriting/raw/mathwriting-2024.tgz','r|gz') as tar:
        for member in tar:
            if not member.isfile() or '/symbols/' not in '/'+member.name or not member.name.endswith('.inkml'):continue
            sid=member.name.rsplit('/',1)[-1].split('.')[0]
            if ('MathWriting',sid) in source_seen:continue
            root,meta=annotations(tar.extractfile(member).read());ts=tokenize(meta.get('label',''))
            if len(ts)==4 and ts[0]=='\\mathrm' and ts[1]=='{' and ts[3]=='}':ts=ts[2:3]
            if len(ts)!=1:continue
            label='-' if ts[0]=='\\frac' else ts[0]
            if label not in CLASSES:continue
            try:add(rasterize(root,width=2,height=96,max_width=192),label,{'source':'MathWriting training-derived symbol','id':sid})
            except ValueError:continue
    np.savez_compressed(ROOT/'auxiliary_glyphs.npz',images=np.stack(x),labels=np.asarray(y))
    audit={'original_count':len(y),'class_counts':{c:counts[c] for c in CLASSES},'missing_originals_to_50':{c:50-counts[c] for c in CLASSES if counts[c]<50},'heldout_parent_glyphs_excluded':excluded,'unique_class_image_hashes':len(seen),'source_test_glyphs_used':False,'mathwriting_symbol_parent_content_audit':'Full symbols.jsonl parent resolution still required before locking global manifest. All symbol-bank originals are official training-derived.','provenance':ids}
    (ROOT/'auxiliary_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');print({k:v for k,v in audit.items() if k!='provenance'},flush=True)
    return audit

if __name__=='__main__':prepare()
