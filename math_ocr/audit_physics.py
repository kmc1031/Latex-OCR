"""Independent manifest/image and split-isolation checks before training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from physics_syntax import canonical,parse

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data_physics'


def main():
    vocab=json.loads((DATA/'vocab.json').read_text(encoding='utf-8'));identities={};pixels={};counts={};conflicts=[]
    for split in ['train','val','test','physics_laws','real_val','real_test']:
        rows=[json.loads(line) for line in (DATA/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        counts[split]=len(rows)
        for index,row in enumerate(rows):
            if index%1000==0:print(f'audit {split}: {index}/{len(rows)}',flush=True)
            assert all(t in vocab for t in row['tokens'])
            key=repr(canonical(parse(row['tokens'])))
            if key in identities and identities[key]!=split:conflicts.append([identities[key],split,'formula'])
            identities[key]=split
            for path in [row['image']]+row.get('image_variants',[]):
                with Image.open(DATA/path) as image:
                    assert image.height==64 and image.width<=512 and image.width%4==0
                    sha=hashlib.sha256(np.asarray(image).tobytes()+str(image.size).encode()).hexdigest()
                if path==row['image']:assert sha==row['image_sha256'],path
                if sha in pixels and pixels[sha][0]!=split:conflicts.append([pixels[sha][0],split,'pixels'])
                pixels[sha]=(split,key)
        print(f'audit {split}: {len(rows)}',flush=True)
    assert not conflicts,conflicts[:10]
    report={'passed':True,'splits':counts,'cross_partition_formula_overlap':0,'cross_partition_image_overlap':0,
            'unique_pixel_hashes':len(pixels),'vocab_size':len(vocab),'attention':False,'ctc':False}
    (DATA/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
