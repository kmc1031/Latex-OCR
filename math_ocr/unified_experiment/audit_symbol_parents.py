"""Resolve symbol-bank parents by normalized stroke geometry, conservatively."""
import hashlib,json,sys,tarfile
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from handwriting_data import annotations
from syntax import normalize,tokenize

def traces(root):
    result=[]
    for t in root.findall('{*}trace'):
        v=np.fromstring(t.text.replace(',',' '),sep=' ')
        result.append(v.reshape(-1,3)[:,:2])
    return result

def fingerprint(parts):
    points=np.concatenate(parts);lo=points.min(0);scale=max(float(np.ptp(points,axis=0).max()),1e-8)
    a=np.concatenate([np.round((p-lo)/scale,4) for p in parts]).astype('<f8')
    return hashlib.sha256(str([len(p) for p in parts]).encode()+a.tobytes()).hexdigest()

def main():
    rows=json.loads((ROOT/'prepared_manifest.json').read_text(encoding='utf-8'));forbidden={r['expression_identity'] for r in rows if r['split']!='train'}
    refs=defaultdict(list);parentmap=defaultdict(list);bank={};scanned=0
    with tarfile.open(ROOT.parent/'data_mathwriting/raw/mathwriting-2024.tgz','r|gz') as tar:
        for m in tar:
            if not m.isfile():continue
            if m.name.endswith('/symbols.jsonl'):
                for line in tar.extractfile(m):
                    r=json.loads(line);refs[r['sourceSampleId']].append(r['strokeIndices'])
                print('parent references',len(refs),flush=True)
            elif m.name.endswith('.inkml'):
                sid=Path(m.name).stem;split=m.name.split('/')[-2]
                if split=='symbols':
                    root,_=annotations(tar.extractfile(m).read());bank[sid]=fingerprint(traces(root))
                elif split=='train' and sid in refs:
                    root,meta=annotations(tar.extractfile(m).read());ts=tokenize(meta.get('normalizedLabel',meta.get('label','')))
                    try:identity=normalize(ts)[1]
                    except (ValueError,RecursionError):identity=None
                    parts=traces(root)
                    for indices in refs[sid]:
                        fp=fingerprint([parts[i] for i in indices]);parentmap[fp].append({'id':sid,'heldout_content':identity in forbidden})
                    scanned+=1
    mapped={sid:parentmap.get(fp,[]) for sid,fp in bank.items()};unmatched=[sid for sid,parents in mapped.items() if not parents];banned=[sid for sid,parents in mapped.items() if not parents or any(p['heldout_content'] for p in parents)]
    aux=json.loads((ROOT/'auxiliary_audit.json').read_text(encoding='utf-8'));exclude=[i for i,p in enumerate(aux['provenance']) if p['source'].startswith('MathWriting') and p['id'] in banned]
    report={'bank_symbols':len(bank),'parents_scanned':scanned,'matched_symbols':len(bank)-len(unmatched),'unmatched_symbols':unmatched,'conservative_banned_symbol_ids':banned,'excluded_auxiliary_indices':exclude,'method':'Normalized original stroke coordinates rounded to 4 decimals; ambiguous fingerprints excluded if ANY parent has heldout content. Unmatched bank symbols excluded.'}
    (ROOT/'symbol_parent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print({k:v for k,v in report.items() if not isinstance(v,list)},'excluded aux',len(exclude),flush=True)

if __name__=='__main__':main()
