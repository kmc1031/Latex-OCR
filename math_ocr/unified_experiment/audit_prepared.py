"""Audit real sources, targets, splits and image-only training preprocessing.

No test prediction, checkpoint fitting, or accuracy evaluation occurs here.
"""
import hashlib,json
from collections import Counter
from PIL import Image
import numpy as np
from engine import ROOT,CFG,segment
from syntax import normalize

real=json.loads((ROOT/'prepared_manifest.json').read_text(encoding='utf-8'));syn=json.loads((ROOT/'synthetic_manifest.json').read_text(encoding='utf-8'));plan=json.loads((ROOT/'capture_plan.json').read_text(encoding='utf-8'))['records']
by={s:[r for r in real if r['split']==s] for s in ['train','val','test']}
for i,a in enumerate(by):
    for b in list(by)[i+1:]:
        assert not ({r['expression_identity'] for r in by[a]}&{r['expression_identity'] for r in by[b]})
        assert not ({r['image_sha256'] for r in by[a]}&{r['image_sha256'] for r in by[b]})
hold={r['expression_identity'] for r in real if r['split']!='train'}|{normalize(r['planned_target_tokens'])[1] for r in plan if r['split']!='train'}
assert not ({r['expression_identity'] for r in syn}&hold)
assert not ({normalize(r['planned_target_tokens'])[1] for r in plan if r['split']=='train'}&{r['expression_identity'] for r in real if r['split']!='train'})
for r in real+syn:
    with Image.open(ROOT/r['image']) as im:
        if 'image_sha256' in r:assert hashlib.sha256(im.tobytes()).hexdigest()==r['image_sha256']
    _,identity=normalize(r['tokens']);assert identity==r['expression_identity'];assert len(r['tokens'])<=63;assert set(r['tokens'])<=set(CFG['output_tokens'])
failed=[];counts=[]
for i,r in enumerate(by['train']+syn):
    try:
        x,g,b=segment(Image.open(ROOT/r['image']));assert x.shape==(len(b),1,32,32) and g.shape==(len(b),8);assert np.isfinite(x).all() and np.isfinite(g).all();counts.append(len(b))
    except ValueError as exc:failed.append({'id':r['sample_id'],'source':r['source'],'error':str(exc)})
    if (i+1)%1000==0:print('preprocessed training images',i+1,flush=True)
result={'source_content_isolation':True,'source_pixel_isolation':True,'synthetic_holdout_isolation':True,'photo_prompt_holdout_isolation':True,'all_target_syntax_roundtrips':True,'all_image_hashes_verified':True,'real_counts':{s:len(v) for s,v in by.items()},'synthetic_training_count':len(syn),'training_preprocess_attempts':len(by['train'])+len(syn),'training_preprocess_failures':failed,'candidate_count_median':float(np.median(counts)),'candidate_count_max':max(counts),'test_predictions_generated':False,'scope':'Data and preprocessing checks, not OCR performance or completed experiment'}
(ROOT/'prepared_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print('AUDIT COMPLETE',len(failed),'training preprocessing failures')
