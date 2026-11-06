"""Single experiment controller: prepare originals, audit, persist data gate.

It never replaces missing photographs by generated images or claims completion.
"""
import hashlib,heapq,json,sys,tarfile,time
from pathlib import Path,PurePosixPath
from collections import Counter
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from handwriting_data import annotations,rasterize
from physics_syntax import parse,tokenize,latex
from syntax import normalize

def save(name,obj):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    cfg=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'));plan=json.loads((ROOT/'capture_plan.json').read_text(encoding='utf-8'));allowed=set(cfg['output_tokens']);archive=ROOT.parent/'data_mathwriting/raw/mathwriting-2024.tgz';caps=cfg['mathwriting_primary_stratum_targets'];pools={};reject=Counter();scanned=Counter();oldtest=set();reserved=set();began=time.time()
    for r in plan['records']:
        if r['split']!='train':reserved.add(repr(parse(tokenize(r['planned_target_latex']))))
    old=ROOT.parent/'data_handwriting/test.jsonl'
    if old.exists():
        oldtest={json.loads(s)['ink_id'] for s in old.read_text(encoding='utf-8').splitlines()}
    save('state.json',{'state':'DATA_PREPARING','complete':False,'photographs_available':sum(bool(r.get('image_path')) for r in plan['records']),'photos_required':240})
    auditpath=ROOT/'data_audit.json'
    if not (ROOT/'prepared_manifest.json').exists() or not auditpath.exists() or json.loads(auditpath.read_text(encoding='utf-8')).get('source_selection_version')!=2:
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith('.inkml'):continue
                source=PurePosixPath(member.name);s=source.parts[-2];s='val' if s=='valid' else s
                if s not in caps:continue
                scanned[s]+=1
                if s=='test' and source.stem in oldtest:reject['historical_test_id']+=1;continue
                raw=tar.extractfile(member).read()
                try:
                    root,meta=annotations(raw);ts=tokenize(meta.get('normalizedLabel',meta.get('label','')));ts=[t for t in ts if t not in ['\\left','\\right','\\,','\\!','\\;','\\:']];ts=[{'\\dfrac':'\\frac','\\tfrac':'\\frac'}.get(t,t) for t in ts]
                    if set(ts)-allowed:reject['unsupported_token']+=1;continue
                    tree=parse(ts);identity=repr(tree)
                    visible=sum(t not in ['{','}','^','_'] for t in ts)
                    if not 3<=visible<=16 or len(ts)>63:reject['length']+=1;continue
                    if ts.count('\\frac')+ts.count('/')>1:reject['multiple_fraction']+=1;continue
                    def check(n):
                        if n[0] in ['sup','sub']:
                            child=n[2]
                            if child[0]!='symbol' or not (len(child[1])==1 and child[1].isalpha() or child[1].isdigit() and len(child[1])<=2):raise ValueError('nested_script')
                        for z in n[1:]:
                            if isinstance(z,tuple):check(z)
                    check(tree)
                    if identity in reserved:reject['reserved_photo_content']+=1;continue
                    kind='fraction' if '\\frac' in ts or '/' in ts else 'sup' if '^' in ts else 'sub' if '_' in ts else 'basic'
                    normalized,_=normalize(ts)
                    if len(normalized)>63:reject['normalized_length']+=1;continue
                    priority=int(hashlib.sha256((source.stem+'|42|'+s).encode()).hexdigest(),16);key=(s,kind);pool=pools.setdefault(key,{})
                    entry=(-priority,source.stem,identity,ts,raw)
                    if identity not in pool or priority < -pool[identity][0]:pool[identity]=entry
                except (ValueError,RecursionError,KeyError):reject['syntax_or_structure']+=1
                if sum(scanned.values())%25000==0:print('scan',dict(scanned),round(time.time()-began),flush=True)
        seen=set(reserved);selected=[];counts=Counter();imagehash=set()
        for s in ['test','val','train']:
            for kind in ['basic','sup','sub','fraction']:
                for _,sid,identity,ts,raw in sorted(pools.get((s,kind),{}).values(),reverse=True):
                    if counts[s,kind]>=caps[s][kind]:break
                    if identity in seen:continue
                    root,_=annotations(raw);im=rasterize(root,width=2,height=192,max_width=1024);digest=hashlib.sha256(im.tobytes()).hexdigest()
                    if digest in imagehash:continue
                    rel=f'data/images/{s}_{sid}.png';(ROOT/rel).parent.mkdir(parents=True,exist_ok=True);im.save(ROOT/rel);seen.add(identity);imagehash.add(digest);counts[s,kind]+=1
                    normalized,_=normalize(ts);selected.append({'sample_id':sid,'split':s,'source':'MathWriting human ink','kind':kind,'image':rel,'tokens':normalized,'source_tokens':ts,'latex':' '.join(normalized),'expression_identity':identity,'image_sha256':digest})
        save('prepared_manifest.json',selected);save('data_audit.json',{'source_selection_version':2,'scanned':dict(scanned),'rejected':dict(reject),'selected':{s:{k:counts[s,k] for k in caps[s]} for s in caps},'cross_split_content_overlap':0,'cross_split_image_overlap':0,'historical_test_ids_excluded':len(oldtest),'seconds':time.time()-began,'note':'All eligible unique syntax identities considered. No bounded duplicate candidate buffer. Normalized targets round-trip to identical syntax trees.'})
    rows=json.loads((ROOT/'prepared_manifest.json').read_text(encoding='utf-8'));missing=[r['sample_id'] for r in plan['records'] if not r.get('image_path') or not (ROOT/r['image_path']).exists() or r.get('annotation_status')!='reviewed']
    auxpath=ROOT/'auxiliary_audit.json';aux=json.loads(auxpath.read_text(encoding='utf-8')) if auxpath.exists() else {}
    save('missing_data.json',{'photo_ids':missing,'required_photos':240,'reviewed_photos':240-len(missing),'missing_original_glyphs_to_50':aux.get('missing_originals_to_50'),'other_unfinished_requirements':['MathWriting symbol-parent content audit before data lock','photo ingestion and annotation review','training augmentation/resume and orchestration verification','post-freeze diagnostics, uncertainty, latency and UI verification'],'note':'Training/evaluation implementations are not represented as verified full execution while photos are missing.'})
    save('state.json',{'state':'DATA_PREPARING','complete':False,'real_formulas_prepared':len(rows),'reviewed_photos':240-len(missing),'missing_photos':len(missing),'next':'Collect and review listed photos; finish remaining implementation within this same experiment.','test_predictions_generated':False,'training_started':False})
    print('Prepared',len(rows),'real formulas; missing reviewed photographs:',len(missing),flush=True)

if __name__=='__main__':
    active=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    if active.get('protocol_version')=='3.0':
        from run_digital import main as digital_main
        digital_main()
    else:main()
