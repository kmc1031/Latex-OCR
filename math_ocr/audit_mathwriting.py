"""Verify InkML identity and raster isolation, independent of model predictions."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from physics_syntax import parse

ROOT=Path(__file__).resolve().parent/'data_handwriting'


def main():
    ids={};pixels={};counts={};vocab=set(json.loads((ROOT/'vocab.json').read_text(encoding='utf-8')))
    for split in ['train','val','test']:
        rows=[json.loads(line) for line in (ROOT/(split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        counts[split]=len(rows)
        for i,row in enumerate(rows):
            assert set(row['tokens'])<=vocab;parse(row['tokens'])
            if 'ink_id' in row:
                assert row['ink_id'] not in ids or ids[row['ink_id']]==split
                ids[row['ink_id']]=split
                expected={'val':'valid','test':'test'}.get(split)
                if expected:assert row['original_split']==expected
                else:assert row['original_split'] in ['train','symbols']
            with Image.open(ROOT/row['image']) as image:
                assert image.height==64 and image.width<=512 and image.width%4==0
                digest=hashlib.sha256(np.asarray(image).tobytes()+str(image.size).encode()).hexdigest()
            assert digest==row['image_sha256']
            assert digest not in pixels or pixels[digest]==split,(split,'pixel overlap')
            pixels[digest]=split
            if i%3000==0:print(f'hand audit {split}: {i}/{len(rows)}',flush=True)
    result={'passed':True,'splits':counts,'cross_partition_ink_id_overlap':0,'cross_partition_pixel_overlap':0,
            'official_partitions_preserved':True,'formula_identity_overlap_allowed':True,
            'explanation':'Different people can write the same expression. Formula strings are not the split unit in this handwriting suite.'}
    (ROOT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
