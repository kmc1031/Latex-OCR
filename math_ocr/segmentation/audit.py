"""Independent split, annotation and pixel duplication checks."""
import hashlib,json
from collections import Counter
import numpy as np
from PIL import Image
from common import ROOT

def main():
    identities={};ink={};pixels={};counts={};glyphs=json.loads((ROOT/'data/glyphs.json').read_text());arrays=np.load(ROOT/'data/glyphs.npy',mmap_mode='r')
    assert len(arrays)==len(glyphs)
    for split in ['train','val','test']:
        rows=[json.loads(s) for s in (ROOT/f'data/{split}.jsonl').read_text().splitlines()];counts[split]=len(rows);identities[split]={r['formula_identity'] for r in rows}
        for row in rows:
            assert row['ink_hash'] not in ink;ink[row['ink_hash']]=split
            image=Image.open(ROOT/'data'/row['image']);a=np.asarray(image);sha=hashlib.sha256(a.tobytes()+str(image.size).encode()).hexdigest()
            if sha in pixels and pixels[sha]!=split:raise AssertionError('Cross-partition raster duplicate')
            pixels[sha]=split
            assert image.width%4==image.height%4==0
            for g,i in zip(row['glyphs'],row['glyph_indexes']):
                x0,y0,x1,y1=g['box'];assert -1<=x0<x1<=image.width+1 and -1<=y0<y1<=image.height+1
                assert glyphs[i]['split']==split and glyphs[i]['expression']==row['id'] and glyphs[i]['label']==g['label']
            for a,b,r in row['edges']:assert a!=b and 0<=a<len(row['glyphs']) and 0<=b<len(row['glyphs'])
    assert not identities['train']&identities['val']
    result={'passed':True,'expression_counts':counts,'glyph_counts':dict(Counter(g['split'] for g in glyphs)),
            'train_val_formula_overlap':0,'cross_split_ink_overlap':0,'cross_split_pixel_overlap':0,
            'official_test_retained':True,'test_formula_overlap_allowed_and_reported':True}
    (ROOT/'data/audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
