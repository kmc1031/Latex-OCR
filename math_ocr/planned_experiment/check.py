"""Data isolation, matching and position ablation correctness checks."""
import json
import numpy as np
import torch
from PIL import Image,ImageDraw
import run as e

rows={s:json.loads((e.ROOT/f'{s}.json').read_text()) for s in ['train','val','test']}
for i,a in enumerate(rows):
    for b in list(rows)[i+1:]:
        for field in ['expression_id','image_sha256']:
            assert not ({r[field] for r in rows[a]}&{r[field] for r in rows[b]})
        assert not ({g['source_glyph'] for r in rows[a] for g in r['glyphs']}&{g['source_glyph'] for r in rows[b] for g in r['glyphs']})
for s,rs in rows.items():
    paired={}
    for r in rs:
        assert all(t in e.VOCAB for t in r['tokens']) and len(r['tokens'])<=64
        if r['kind']=='fraction':paired.setdefault(r['expression_id'],[]).append(r)
        with Image.open(e.ROOT/r['image']) as im:
            assert all(0<=g['box'][0]<g['box'][2]<=im.width and 0<=g['box'][1]<g['box'][3]<=im.height for g in r['glyphs'])
    assert all(len(v)==2 and v[0]['tokens']==v[1]['tokens'] for v in paired.values())
assert e.box_counts([[0,0,10,10],[0,0,10,10]],[[0,0,10,10]])==[1,1,0]
im=Image.new('L',(100,100),255);d=ImageDraw.Draw(im);d.rectangle((10,20,39,22),fill=0);d.rectangle((10,30,39,32),fill=0);d.rectangle((70,10,75,60),fill=0)
assert len(e.components(im,False))==3 and len(e.components(im,True))==2
m=e.Seq(False).eval();x=torch.randn(2,4,len(e.CLASSES)+4);z=x.clone();z[:,:,-4:]+=100
with torch.inference_mode():
    a=m.encode(x,torch.tensor([4,4]));b=m.encode(z,torch.tensor([4,4]));assert all(torch.equal(u,v) for u,v in zip(a,b))
e.save(e.ROOT/'checks.json',{'passed':True,'checks':['formula and pixel isolation','source glyph isolation','fraction target pairing','box bounds','token coverage','one-to-one matching','equals merging','L0 position invariance']})
print('8 integrity checks passed')
