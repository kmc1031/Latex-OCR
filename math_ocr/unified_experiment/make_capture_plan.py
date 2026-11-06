"""Generate planning prompts, NOT collected images or experimental results.

AI-assisted planning implementation. Parser is the project's physics_syntax.py.
"""
import hashlib
import json
import random
import string
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from physics_syntax import parse, tokenize

CLASSES = list(string.digits + string.ascii_uppercase + string.ascii_lowercase) + ['+', '-', '=', '\\times', '/', '(', ')', '.']
assert len(CLASSES) == len(set(CLASSES)) == 70

def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    existing = ROOT / 'capture_plan.json'
    if existing.exists():
        old = json.loads(existing.read_text(encoding='utf-8'))
        if any(r.get('image_path') or r.get('collection_status') != 'not_collected' for r in old.get('records', [])):
            raise RuntimeError('Collected data exists: do not overwrite this capture plan.')
    rng = random.Random(20261105)
    bag = []
    used = set()
    all_rows = []
    allocation = {'train': (6, 3, 3, 6), 'val': (4, 2, 2, 4), 'test': (10, 5, 5, 10)}
    def letter():
        if not bag:
            bag.extend(string.ascii_letters)
            rng.shuffle(bag)
        return bag.pop()
    for split, sizes in allocation.items():
        counters = Counter()
        for writer in ['A', 'B', 'C']:
            packets = []
            for kind, count in zip(['basic', 'sup', 'sub', 'fraction'], sizes):
                for _ in range(count):
                    while True:
                        r, a, b = letter(), letter(), letter()
                        j = counters[kind]
                        n = 1 + j % 9
                        if kind == 'basic':
                            d = (j + 3) % 10
                            templates = [f'{r}={a}+{b}', f'{r}={a}-{b}', f'{r}={a}\\times {b}',
                                         f'{r}=({a}+{b}){n}', f'{r}={a}+{b}+{n}.{d}', f'{r}={a}{b}+{n}']
                            target = templates[j % len(templates)]
                        elif kind == 'sup':
                            target = f'{r}={a}^{{{2+j%8}}}+{b}'
                        elif kind == 'sub':
                            target = f'{r}={a}_{{{j%13}}}+{b}'
                        else:
                            target = f'{r}=\\frac{{{a}+{n}}}{{{b}}}'
                        tree = parse(tokenize(target))
                        assert len(tokenize(target)) <= 63
                        identity = repr(tree)
                        if identity not in used:
                            used.add(identity)
                            break
                    counters[kind] += 1
                    eid = hashlib.sha256(identity.encode()).hexdigest()[:16]
                    if kind == 'fraction':
                        prompts = [('vertical', target), ('slash', f'{r}=({a}+{n})/({b})')]
                        if j % 2: prompts.reverse()
                    else:
                        prompts = [('normal', target)]
                    packet = []
                    for notation, prompt in prompts:
                        assert parse(tokenize(prompt)) == tree
                        surface = ['-' if t == '\\frac' else t for t in tokenize(prompt) if t not in ['{', '}', '^', '_']]
                        assert 3 <= len(surface) <= 16
                        assert set(surface) <= set(CLASSES)
                        packet.append({'split': split, 'writer_code': writer, 'primary_type': kind,
                            'notation': notation, 'expression_id': eid,
                            'fraction_pair_id': eid if kind == 'fraction' else None,
                            'prompt_latex': prompt, 'planned_target_latex': target,
                            'planned_target_tokens': tokenize(target), 'planned_surface_symbols': surface,
                            'planned_visible_symbols': len(surface), 'image_path': None,
                            'glyphs': None, 'collection_status': 'not_collected',
                            'annotation_status': 'not_reviewed'})
                    packets.append(packet)
            rng.shuffle(packets)
            for index, row in enumerate([r for packet in packets for r in packet], 1):
                sid = f'PH_{split.upper()}_{writer}_{index:03d}'
                row.update({'sample_id': sid, 'suggested_filename': sid + '.jpg',
                    'sheet_id': f'{split}_{writer}_{(index-1)//10+1:02d}'})
                all_rows.append(row)
    summary = {}
    for split in allocation:
        rows = [r for r in all_rows if r['split'] == split]
        counts = Counter(t for r in rows for t in r['planned_surface_symbols'])
        missing = sorted(set(CLASSES) - counts.keys())
        assert not missing, (split, missing)
        summary[split] = {'images_to_collect': len(rows), 'unique_contents': len({r['expression_id'] for r in rows}),
            'by_writer': dict(Counter(r['writer_code'] for r in rows)),
            'by_type_notation': dict(Counter(f"{r['primary_type']}/{r['notation']}" for r in rows)),
            'planned_symbol_occurrences': dict(sorted(counts.items())), 'missing_planned_classes': missing}
    for i, a in enumerate(allocation):
        for b in list(allocation)[i+1:]:
            assert not ({r['expression_id'] for r in all_rows if r['split'] == a} & {r['expression_id'] for r in all_rows if r['split'] == b})
    assert len(all_rows) == 240
    save('capture_plan.json', {'status': 'planning_only_no_images_collected', 'classes': CLASSES,
        'writer_code_mapping': 'Assign A/B/C once before collection; keep mapping fixed.', 'records': all_rows})
    save('capture_plan_checks.json', {'scope': 'planning prompts only, not collected photographs or model results', 'passed': True, 'checks': ['240 planned images', '70-class planned coverage in each split',
        'content isolation across splits', 'fraction pair semantic identity', 'visible symbol range', 'no image claimed collected'], 'summary': summary})
    md = ['# 수식 사진 수집 문항표', '', '**계획 문항이며 사진은 아직 수집하지 않았다.** A/B/C를 세 필자에 고정 배정한다.',
        '각 문항을 자연스럽게 손으로 쓰고 한 수식씩 촬영한다. LaTeX 코드를 글자로 쓰지 말고 그 코드가 뜻하는 수식을 쓴다.',
        '분수는 notation에 따라 세로 분수 또는 슬래시로 쓴다. 잘못 옮겨 쓴 경우 예측 전에 재촬영하고 라벨을 확인한다.',
        '시험 문항은 학습·검증에 사용하지 않는다. 한 종이에 서로 다른 split의 문항을 섞어 쓰지 않는다.', '']
    for split in allocation:
        for writer in ['A', 'B', 'C']:
            rows = [r for r in all_rows if r['split'] == split and r['writer_code'] == writer]
            md += [f'## {split} / 필자 {writer}: {len(rows)}장', '', '| 파일명 | 유형·표기 | 작성할 수식의 LaTeX |', '|---|---|---|']
            md += [f"| {r['suggested_filename']} | {r['primary_type']}/{r['notation']} | `{r['prompt_latex']}` |" for r in rows]
            md.append('')
    (ROOT/'CAPTURE_SHEET.md').write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps({s: {k:v for k,v in x.items() if k!='planned_symbol_occurrences'} for s,x in summary.items()}, ensure_ascii=False))

if __name__ == '__main__': main()
