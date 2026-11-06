"""Build the measured handwriting comparison, learning plot and full test gallery."""
from collections import Counter
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from physics_syntax import GREEK,GLYPHS,FUNCTIONS,parse

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'physics'


def mathml(tree):
    kind=tree[0]
    if kind=='symbol':
        value=tree[1];tag='mn' if value.replace('.','').isdigit() else 'mo' if value in FUNCTIONS else 'mi'
        return f'<{tag}>'+html.escape(GLYPHS.get(value,value[1:] if value in FUNCTIONS else value))+f'</{tag}>'
    a=mathml(tree[1])
    if kind in ['round','square']:return '<mrow><mo>'+('(' if kind=='round' else '[')+'</mo>'+a+'<mo>'+(')' if kind=='round' else ']')+'</mo></mrow>'
    if kind in ['div','sup','sub']:
        tag={'div':'mfrac','sup':'msup','sub':'msub'}[kind];return f'<{tag}>'+a+mathml(tree[2])+f'</{tag}>'
    if kind=='sqrt':return '<msqrt>'+a+'</msqrt>'
    if kind=='roman':return '<mstyle mathvariant="normal">'+a+'</mstyle>'
    if kind in ['vec','hat','bar','dot','ddot']:
        return '<mover accent="true">'+a+'<mo>'+{'vec':'→','hat':'^','bar':'¯','dot':'˙','ddot':'¨'}[kind]+'</mo></mover>'
    if kind in ['negative','positive']:return '<mrow><mo>'+('−' if kind=='negative' else '+')+'</mo><mo>(</mo>'+a+'<mo>)</mo></mrow>'
    if kind=='multiply':return '<mrow>'+a+mathml(tree[2])+'</mrow>'
    op={'add':'+','subtraction':'−','eq':'=','approx':'≈','neq':'≠','leq':'≤','geq':'≥','propto':'∝','pm':'±','dotproduct':'⋅','crossproduct':'×'}[kind]
    return '<mrow><mo>(</mo>'+a+'<mo>'+op+'</mo>'+mathml(tree[2])+'<mo>)</mo></mrow>'


def preview(tokens):
    try:return '<math xmlns="http://www.w3.org/1998/Math/MathML">'+mathml(parse(tokens,preserve_groups=True))+'</math>'
    except (ValueError,RecursionError):return '<span class="invalid">수식 구조 해석 실패</span>'


def substitutions(rows):
    counts=Counter()
    for row in rows:
        a=row['target_tokens'];b=row['prediction_tokens'];n=len(a);m=len(b)
        d=[[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1):d[i][0]=i
        for j in range(m+1):d[0][j]=j
        for i in range(1,n+1):
            for j in range(1,m+1):d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]!=b[j-1]))
        i,j=n,m
        while i or j:
            if i and j and d[i][j]==d[i-1][j-1]+(a[i-1]!=b[j-1]):
                if a[i-1]!=b[j-1]:counts[(a[i-1],b[j-1])]+=1
                i-=1;j-=1
            elif i and d[i][j]==d[i-1][j]+1:counts[(a[i-1],'∅')]+=1;i-=1
            else:counts[('∅',b[j-1])]+=1;j-=1
    return counts


def main():
    data=json.loads((OUT/'handwriting_metrics.json').read_text(encoding='utf-8'))
    summary=json.loads((ROOT/'data_handwriting/summary.json').read_text(encoding='utf-8'))
    decoder=json.loads((OUT/'decoder_selection.json').read_text(encoding='utf-8'))
    chosen=json.loads((OUT/'candidate_selection.json').read_text(encoding='utf-8'))
    exposure=Counter()
    for line in (ROOT/'data_handwriting/train.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row['family']!='printed_replay':exposure.update(row['tokens'])
    (OUT/'handwriting_token_exposure.json').write_text(json.dumps(dict(exposure),ensure_ascii=False,indent=2),encoding='utf-8')
    selected=data['handwriting'];labels={'legacy':'기존 모델 P','printed_transfer':'기호 확장·인쇄 학습','handwriting':'선택한 손글씨 모델'}
    lines=['# 손글씨 물리 수식 OCR 고도화 결과','',
           'Attention과 CTC 없이 CNN–BiLSTM–LSTM을 유지하고, 그리스 문자·영문 대소문자·수치·분수·첨자·벡터를 포함하는 어휘와 실제 손글씨 학습을 추가했다.','',
           '## 같은 이미지에서 비교한 결과','',
           '수치는 **수식 구조 전체가 일치한 이미지 비율**이다. 기호별 정확도나 물리 법칙의 옳고 그름을 측정하는 값이 아니다. 손글씨 시험은 일반 수식의 부분집합으로, 손글씨 물리 법칙만 모은 시험은 아니다. 물리 공식 시험은 별도의 인쇄 이미지다. 괄호·곱과 분수의 일부 표기 차이를 정규화하되, 변수명·숫자·첨자·분모를 교정하거나 공식을 조회해 정답으로 바꾸지 않는다. 모든 최종 평가는 UI와 같은 전처리·batch size 1에서 정답 토큰 없이 수행했다.','',
           '| 평가 이미지 | 수 | 기존 P | 인쇄 전이 모델 | 새 손글씨 모델 |','|---|---:|---:|---:|---:|']
    for key,title in [('test','실제 손글씨'),('test_greek','그리스 문자 포함 손글씨'),('test_capture','손글씨의 모의 조명·회전 변화'),('legacy_test','이전의 단순 인쇄 수식'),('physics_laws','학습에서 제외한 인쇄 물리 공식')]:
        lines.append('|'+title+'|'+str(selected[key]['n'])+'|'+'|'.join(f'{data[name][key]["semantic_exact"]:.1%}' for name in labels)+'|')
    improvement=100*(selected['test']['semantic_exact']-data['printed_transfer']['test']['semantic_exact'])
    greek_improvement=100*(selected['test_greek']['semantic_exact']-data['printed_transfer']['test_greek']['semantic_exact'])
    lines+=['',f'인쇄 전이 모델에서 손글씨 학습을 거친 최종 모델로 바꾸면 손글씨 전체 일치율은 {improvement:+.1f}%p, 그리스 문자가 포함된 식은 {greek_improvement:+.1f}%p 변했다.']
    if selected['legacy_test']['semantic_exact']<data['legacy']['legacy_test']['semantic_exact']:
        lines+=['기존의 단순 인쇄 수식은 P 모델이 더 높은 성능을 보였다. 해당 모델의 가중치를 보존하고 UI에서 별도로 선택할 수 있게 했다.']
    lines+=['','## 입력 길이에 따른 결과','','토큰 수에는 `\\frac`, 괄호·중괄호·첨자 기호도 포함된다. 단순 문자 수와 다르다.','',
            '| 토큰 수 | 시험 이미지 | 새 모델 구조 일치율 |','|---|---:|---:|']
    for length,score in selected['test']['lengths'].items():
        if score['n']:lines.append(f'|{length}|{score["n"]}|{score["semantic_exact"]:.1%}|')
    lines+=['','## 학습·선택 방식','',
            f'- 학습 {summary["splits"]["train"]:,}개: '+', '.join(f'{k} {v:,}' for k,v in summary['train_family_counts'].items())+'.',
            f'- 원본 MathWriting의 train/valid/test 구분을 유지했다. 검증 {summary["splits"]["val"]}개, 시험 {summary["splits"]["test"]}개이며 그중 그리스 문자 시험은 {summary["splits"]["test_greek"]}개다.',
            '- 학습과 평가 사이 InkML ID·이미지 해시 중복을 검사했다. 같은 수식을 다른 사람이 쓴 경우는 허용하므로 수식 문자열이 완전히 새로운 경우만 모은 시험은 아니다.',
            '- 그리스 문자를 강조하고 1–32토큰·지원 어휘로 필터링한 부분집합이다. MathWriting 전체 benchmark 성적과 비교할 수 없다.',
            '- 최종 시험은 학습·체크포인트·디코더 선택에 사용하지 않았다. 후보는 순차 전이 학습이며 학습 횟수와 손실도 달라서 구조 하나의 효과만 분리한 실험은 아니다.',
            '- UI의 세 예제는 검증 데이터에서 인식된 짧은 식을 골랐다. 일반화 성능의 근거는 아래에 모두 공개한 독립 시험 800개다.',
            f'- 선택한 구조: `{chosen["selected"]}`, 체크포인트 epoch {selected["epoch"]}. 각 구조의 최고 학습 검증 체크포인트를 같은 검증 200개씩·batch size 1·UI 전처리에서 다시 비교했다.',
            f'- 검증으로 선택한 beam width {decoder["selected"]["beam_width"]}, length alpha {decoder["selected"]["length_alpha"]}.',
            '- CNN의 세로 위치 정보를 유지하고 BiLSTM의 최종 상태와 고정 8개 위치 특징을 결합했다. 손글씨 학습에는 기호 존재 여부와 출력 길이 보조 분류 손실을 추가했다.',
            '- 고정 위치 보완 후보는 출력 단계·입력 너비·출력 길이로 정한 세 위치의 특징을 사용한다. 학습 중에는 실제 길이와 모델이 추정한 길이를 섞고, 추론에는 **모델이 추정한 길이만** 쓴다. Query/key·학습된 위치별 attention 가중치·CTC는 없다.',
            '- 추가 CNN 후보는 128채널 residual block 2개와 압축하지 않은 고정 8개 위치 특징을 추가했다. 구조 선택 수치와 각 설정은 `candidate_selection.json`에 남겼다.',
            '- 짧은 기호가 긴 식에 묻히지 않도록 보완 후보에서는 수식별로 평균한 cross-entropy를 사용했다.',
            '- 회전 증강은 여백을 확장해 수식이 잘리지 않게 했고, 추론에는 종이 배경의 조명을 완화하는 연속 밝기 보정을 추가했다.','',
            '### 검증에서 비교한 후보','',
            '검증 점수는 손글씨 0.5 + 그리스 문자 포함 손글씨 0.25 + 모의 촬영 0.15 + 인쇄 0.1의 가중 평균이다. 아래 검증 수치는 각 부분집합 200개에서 UI와 같은 전처리로 다시 측정했다.','',
            '| 후보 | 파라미터 | 선택 epoch | Beam | 손글씨 | 그리스 포함 | 가중 검증 점수 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,candidate in chosen['candidates'].items():
        lines.append(f'|{name}|{candidate["parameters"]:,}|{candidate["epoch"]}|{candidate["beam_width"]}|{candidate["semantic"]["val"]:.1%}|{candidate["semantic"]["val_greek"]:.1%}|{candidate["validation_score"]:.1%}|')
    lines+=['','### 검증에서 비교한 디코더','','문법이 유효하게 끝난 beam 후보 중 모델 점수가 가장 높은 출력을 고른다. 물리 공식 목록이나 정답으로 출력을 보정하지 않는다.','',
            '| Beam width | Length alpha | 가중 검증 점수 |','|---:|---:|---:|']
    for trial in decoder['trials']:lines.append(f'|{trial["beam_width"]}|{trial["length_alpha"]}|{trial["score"]:.1%}|')
    lines+=['','![학습 및 검증 추이](handwriting_learning.png)','',
            '학습 중 그래프는 저장된 입력과 batch size 32의 검증 기록이다. 위의 최종 선택 표는 검증 표본 수와 전처리를 통일하고 batch size 1로 다시 측정한 값이다.','',
            '## 남은 오류와 실사용 범위','']
    if selected['test']['semantic_exact']<.1:
        lines.append('**손글씨 물리 수식의 실사용 목표를 달성하지 못했다.** 현재 모델은 실제 손글씨를 일반적으로 변환하기에 부족하다. 데이터 준비·학습·UI 기반과 측정 결과를 남긴 실험 모델로 보존한다.')
    elif selected['test']['semantic_exact']<.8:
        lines.append('**손글씨 수식을 확인 없이 자동 변환하는 용도로는 아직 부족하다.** 짧고 분명한 수식을 선택하고 LaTeX 초안을 원본과 대조해 수정하는 용도로 사용해야 한다.')
    else:lines.append('짧은 손글씨 수식부터 사용할 수 있으나 인식한 숫자·첨자·분모는 원본과 대조해야 한다.')
    lines+=['','- MathWriting은 디지털 펜으로 쓴 좌표를 여기서 이미지로 만든 데이터다. 실제 공책·화이트보드를 촬영한 사진의 정확도는 측정하지 않았다.',
            '- 모의 촬영 시험은 원본 손글씨에 조명 그라데이션·작은 회전·잡음을 가한 데이터다. 실제 사진의 대체 benchmark가 아니다.',
            '- 복잡한 중첩 분수, 긴 식, 행렬·다중 행, 매우 흘려 쓴 글씨는 지원을 보장하지 않는다.',
            '- UI의 초록 상태나 문법 유효성은 수식이 정확하다는 뜻이 아니다.','',
            f'- 새 모델 시험 token error rate: {selected["test"]["token_error_rate"]:.1%}.',
            f'- 빈 이미지 대조군 구조 일치율: {selected["blank_control"]["semantic_exact"]:.1%}.','',
            '가장 많은 토큰 치환·누락·삽입(최소 편집 정렬 기준):','']
    predictions=json.loads((OUT/'handwriting_test.json').read_text(encoding='utf-8'))
    errors=substitutions(predictions)
    for (a,b),count in errors.most_common(12):lines.append(f'- `{a}` → `{b}`: {count}회')
    diagnostic_file=OUT/'generalization_diagnosis.json'
    if diagnostic_file.exists():
        diagnosis=json.loads(diagnostic_file.read_text(encoding='utf-8'))
        lines+=['','## 검증과 시험의 차이','',
                '시험 결과를 확인한 뒤 수행한 진단이다. 이 진단으로 모델·체크포인트·디코딩을 다시 선택하지 않았다.',
                f'- 내보낸 모델과 선택한 학습 체크포인트의 모든 가중치 일치: `{diagnosis["exported_weights_identical"]}`.',
                f'- 내보낸 모델의 전체 검증 800개 구조 일치율: {diagnosis["full_validation"]["semantic_exact"]:.1%}.','',
                '| 부분 | 식 수 | 학습에 같은 구조의 식이 있는 수 | 평균 토큰 수 | 같은 식 구조 일치율 | 새로운 식 구조 일치율 |',
                '|---|---:|---:|---:|---:|---:|']
        for split,title in [('val','검증'),('test','시험')]:
            item=diagnosis['splits'][split]
            lines.append(f'|{title}|{item["n"]}|{item["formulas_seen_in_training"]}|{item["mean_tokens"]:.1f}|{item["seen_formula"]["semantic_exact"]:.1%}|{item["new_formula"]["semantic_exact"]:.1%}|')
        lines+=['','학습과 검증 사이에는 같은 식을 다른 손글씨로 쓴 경우가 많고, 시험에는 새로운 식과 더 긴 식이 많았다. 이미지·InkML ID 중복과는 다른 문제다. 현재 모델은 새로운 식의 기호·구조를 조합하는 일반화 성능이 부족하다. 식의 중복 여부·길이·필체의 영향을 이 실험 하나로 분리할 수는 없다.','',
                '- [내보내기 및 일반화 진단 원본](generalization_diagnosis.json)']
    lines+=['','## 주요 그리스 문자 포함 식','',
            '아래는 해당 문자가 들어 있는 **식 전체**의 구조 일치율이다. 개별 문자의 인식률이 아니며 식 길이와 복잡도의 차이도 포함한다. 여러 문자가 들어 있는 식은 여러 행에 포함된다. 표본이 적은 행은 해석에 주의해야 한다.','',
            '| 문자 | 손글씨 학습 정답에서의 출현 수 | 시험 식 수 | 식 전체 일치율 |','|---|---:|---:|---:|']
    for name in ['alpha','beta','gamma','delta','theta','lambda','mu','pi','sigma','omega']:
        subset=[r for r in predictions if '\\'+name in r['target_tokens']]
        if subset:lines.append(f'|{GLYPHS[chr(92)+name]} (`{chr(92)+name}`)|{exposure[chr(92)+name]}|{len(subset)}|{sum(r["semantic_exact"] for r in subset)/len(subset):.1%}|')
    lines+=['','## 사용','',
            '`../start_ui.cmd`를 더블클릭한다. 기본 손글씨 모델과 기존 기본 모델을 UI에서 선택할 수 있다.','',
            '- [전체 손글씨 시험 800개 보기](handwriting_gallery.html)',
            '- [원본 수치](handwriting_metrics.json)',
            '- [검증 기반 구조 선택](candidate_selection.json)',
            '- [검증 기반 디코더 선택](decoder_selection.json)','',
            '데이터: [Google Research MathWriting](https://github.com/google-research/google-research/tree/master/mathwriting), CC-BY-NC-SA-4.0. 원본 다운로드 SHA-256 및 GCS MD5 검증은 `../data_mathwriting/raw/`의 JSON에 기록했다. 인쇄 전이 학습에는 [IM2LATEX 배포본](https://huggingface.co/datasets/yuntian-deng/im2latex-100k) 일부와 Matplotlib 합성 수식을 사용했다.']
    (OUT/'HANDWRITING_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # All test images are included; examples are not selected by correctness.
    cards=[]
    for i,row in enumerate(predictions):
        image=(ROOT/'data_handwriting'/row['image']).read_bytes();url='data:image/png;base64,'+base64.b64encode(image).decode()
        greek=any(t.startswith('\\') and t[1:] in GREEK for t in row['target_tokens'])
        cards.append(f'<article data-correct="{int(row["semantic_exact"])}" data-greek="{int(greek)}"><header>#{i+1} · '+('일치' if row['semantic_exact'] else '오류')+f' · {len(row["target_tokens"])} tokens</header><img src="{url}" alt="시험 손글씨 {i+1}"><div><b>정답</b>'+preview(row['target_tokens'])+'</div><code>'+html.escape(row['target'])+'</code><div><b>인식</b>'+preview(row['prediction_tokens'])+'</div><code>'+html.escape(row['prediction'])+'</code></article>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손글씨 OCR 전체 시험 결과</title><style>body{font:14px/1.7 system-ui,"Malgun Gothic",sans-serif;background:#f4f7f4;color:#263b30;margin:30px}main{max-width:1200px;margin:auto}h1{font-size:25px}nav{position:sticky;top:0;background:#f4f7f4;padding:15px 0;display:flex;gap:12px;z-index:2}button{padding:8px 15px;border:1px solid #cbd9ce;background:white;border-radius:6px;cursor:pointer}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}article{background:white;border:1px solid #dde6df;border-radius:9px;padding:18px;overflow:auto}article[data-correct="0"]{border-top:3px solid #b67354}article[data-correct="1"]{border-top:3px solid #4d8762}header{color:#789080;font-size:12px}img{max-width:100%;height:64px;object-fit:contain;object-position:left;display:block;margin:12px 0}article div{display:flex;gap:15px;align-items:center;margin:12px 0}b{font-size:11px;color:#8a9b90;min-width:28px}math{font-size:21px}code{display:block;font-size:11px;word-break:break-all;color:#819388}.invalid{color:#b67354}article[hidden]{display:none}</style><main><h1>손글씨 OCR · 전체 시험 결과</h1><p>독립 시험 이미지 전체를 표시합니다. 일치 여부는 변수·숫자·첨자를 포함한 수식 구조 비교 결과입니다.</p><nav><button data-filter="all">전체</button><button data-filter="error">오류만</button><button data-filter="greek">그리스 문자</button><span id="count"></span></nav><section>'''+''.join(cards)+'''</section></main><script>const cards=[...document.querySelectorAll('article')];document.querySelectorAll('button').forEach(b=>b.onclick=()=>{let n=0;cards.forEach(c=>{c.hidden=b.dataset.filter==='error'?c.dataset.correct==='1':b.dataset.filter==='greek'?c.dataset.greek==='0':false;if(!c.hidden)n++});document.getElementById('count').textContent=n+'개'});document.querySelector('button').click();</script></html>'''
    (OUT/'handwriting_gallery.html').write_text(page,encoding='utf-8')
    history=json.loads((ROOT/'runs'/chosen['selected']/'history.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(11,4));epochs=[r['epoch'] for r in history]
    axes[0].plot(epochs,[r['train_ce'] for r in history],color='#246b50');axes[0].set(xlabel='Epoch',ylabel='Training CE',title='Selected architecture training')
    for split,label in [('val','Handwriting'),('val_greek','Greek handwriting'),('val_capture','Simulated capture')]:
        axes[1].plot(epochs,[100*r['semantic'][split] for r in history],label=label)
    axes[1].set(xlabel='Epoch',ylabel='Whole-expression match (%)',title='Validation only');axes[1].legend()
    for ax in axes:ax.grid(alpha=.18)
    fig.tight_layout();fig.savefig(OUT/'handwriting_learning.png',dpi=160);plt.close(fig)
    print('Report, full gallery and learning plot written.',flush=True)


if __name__=='__main__':main()
