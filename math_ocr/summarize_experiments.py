"""Combine the three diagnostics without equating synthetic/real performance."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    def read(path):
        return json.loads((ROOT/path).read_text(encoding="utf-8"))
    real = read("runs/pilot/metrics.json")
    simple = read("runs/controlled/metrics.json")
    overfit = read("runs/overfit_check/metrics.json")
    def percent(x):
        return f"{x*100:.1f}%"
    def row(label, m):
        t=m["test"]
        return f"| {label} | {m['model']['train_samples']:,} | {t['exact_count']}/{t['n']} ({percent(t['exact_match'])}) | {percent(t['normalized_similarity'])} |"
    group_rows=[]
    for tag,label in [("linear","사칙연산"),("scripts","위·아래첨자"),("fraction","분수"),("radical","제곱근")]:
        m=simple["test"]["groups"][tag]
        group_rows.append(f"| {label} | {m['n']} | {percent(m['exact_match'])} | {percent(m['normalized_similarity'])} |")
    control_rows=[]
    for label,m in [("실제 IM2LATEX",real),("통제된 합성",simple)]:
        control_rows.append(f"| {label} | {percent(m['control_normal']['normalized_similarity'])} | {percent(m['control_blank']['normalized_similarity'])} | {percent(m['control_shuffled']['normalized_similarity'])} |")
    fraction_errors = [r for r in read("runs/controlled/predictions_test.json") if r['tags']['fraction'] and not r['exact']]
    denominator_only = 0
    for r in fraction_errors:
        a,b = r['target_tokens'],r['predicted_tokens']
        if len(a)==len(b) and len(a)>8 and all(x==y for i,(x,y) in enumerate(zip(a,b)) if i!=7) and a[7]!=b[7]:
            denominator_only += 1
    text=f"""# Attention 없는 CNN·LSTM 수식 OCR: 실행 결과

CNN + BiLSTM encoder + LSTM decoder를 직접 구현하고 RTX 3050 Ti 4GB에서 학습했다.
**제한된 기호와 구조에서 학습 가능성과 실제 논문 수식으로의 일반화 가능성을 별도로 검증했다.**

## 처음 보는 수식의 인식 결과

| 실험 | 학습 수식 수 | 시험 수식 완전 일치 | 시험 토큰 편집 유사도 |
|---|---:|---:|---:|
{row('실제 IM2LATEX 짧은 수식',real)}
{row('통제된 합성 수식',simple)}

두 행은 **서로 다른 데이터 분포와 난도**의 실험이다. 공정한 모델 간 성능 비교표가 아니다.

- 실제 IM2LATEX: 어휘 338개, 길이 3–32토큰, 24 epochs. 시험 수식에는 실제 논문에 쓰이는 그리스 문자와 다양한 LaTeX 명령이 포함된다.
- 통제된 합성: 어휘 28개, 길이 5–11토큰, 40 epochs. 사칙연산·위첨자·아래첨자·분수·근호의 5종 템플릿 계열에서 생성했다. 학습과 시험은 템플릿 계열·폰트를 공유하며 수식 문자열과 픽셀 이미지는 겹치지 않는다.
- 모든 정확도는 이전 정답 토큰을 주지 않고 BOS부터 EOS까지 생성한 **greedy decoding** 결과다.
- test는 checkpoint 선택에 쓰지 않았다. validation 전체 수식 완전 일치율, 동률이면 편집 유사도로 선택했다.
- 완전 일치율은 canonical LaTeX 토큰열의 일치다. 렌더링 동등성 평가는 하지 않았다. 편집 유사도는 기호별 정확도와 다르다.

실제 시험의 유일한 완전 일치 항목은 `S = -`라는 매우 짧은 항목이다. **현재 실제 논문 수식 인식은 성공했다고 보기 어렵다.**

## 통제된 수식의 구조별 결과

| 구조 | 시험 개수 | 완전 일치율 | 편집 유사도 |
|---|---:|---:|---:|
{chr(10).join(group_rows)}

분수 오답 {len(fraction_errors)}개 중 {denominator_only}개는 분모 한 기호만 틀렸다. 예를 들어 `\\frac {{ 6 + 1 }} {{ 1 }}`을 `\\frac {{ 6 + 1 }} {{ x }}`로 읽었다. 합성 데이터에서는 분모가 단일 기호인 문법을 사용하므로 위치를 명확히 비교할 수 있다. 세로 방향의 작은 기호를 새로운 조합에서 읽는 문제가 특히 크게 나타났다.

학습에 없는 **중첩 분수** 200개에서는 완전 일치율 {percent(simple['long_test']['exact_match'])}, 편집 유사도 {percent(simple['long_test']['normalized_similarity'])}였다. 실제 데이터의 더 긴 수식 250개에서도 완전 일치율 {percent(real['long_test']['exact_match'])}였다.

## 모델이 이미지를 사용하는가?

동일 시험 수식 앞 256개에 대해 이미지 조건만 바꾼 대조 실험이다.

| 데이터 | 정상 이미지 유사도 | 빈 이미지 유사도 | 다른 수식 이미지 유사도 |
|---|---:|---:|---:|
{chr(10).join(control_rows)}

blank는 이미지 내용만 지우고 너비는 유지했다. shuffled는 batch 안에서 이미지와 너비를 함께 바꿨다. 별도 학습한 언어 모델 baseline은 아니다.

## 구현 자체가 학습 가능한가?

별도의 무작위 초기화 모델로 실제 수식 **64개만 반복 학습**했을 때 {overfit['final']['exact_count']}/64 ({percent(overfit['final']['exact_match'])})를 재현했다. 이는 암기 검사 결과이며, 처음 보는 수식 정확도가 아니다. 약 {overfit['seconds']:.0f}초 동안 최대 {overfit['history'][-1]['epoch']}회 반복했다.

## 이번 결과로 판단할 수 있는 것

통제된 수식의 시험 정확도와 이미지 대조 결과를 통해, attention 없는 CNN·LSTM이 제한된 구조 안에서 새로운 기호 조합을 읽는지 판단할 수 있다. 실제 논문 수식과 새 중첩 구조 결과는 현재 모델의 확장 한계를 보여준다.

**이 모델을 그대로 크게 만들거나 데이터만 늘리면 실용 OCR이 된다고 결론 내릴 근거는 아직 부족하다.** 데이터 난도, 수식 길이, 기호 수, 학습량, 고정 길이 벡터의 정보 압축이 함께 달라지므로 이번 실험만으로 성능 저하의 원인을 하나로 특정할 수 없다.

다음 확장 실험은 다음 순서가 적절하다.

1. 통제된 문법에 중첩 분수를 학습용으로 추가해, 학습한 중첩 깊이 안의 새 수식과 더 깊은 수식을 구분해 평가한다.
2. 기호 수와 길이를 한 번에 하나씩 늘려 성능이 떨어지는 지점을 찾는다.
3. CNN 특징의 공간 구획을 고정 pooling으로 보존하거나 행·열 LSTM을 함께 사용하는 attention 없는 encoder를 비교한다. 모델 크기·학습 횟수를 맞춰 비교해야 한다.
4. 실제 데이터의 LaTeX 표기를 정규화하고 데이터 규모를 늘린 뒤, 검증된 구조로 손글씨 데이터에 확장한다.

## 파일과 사용법

- [모델 코드](model.py), [학습 코드](train.py), [실행 방법](README.md)
- [실제 수식 상세 결과](runs/pilot/REPORT.md), [전체 500개 예측 보기](runs/pilot/report.html)
- [통제된 수식 상세 결과](runs/controlled/REPORT.md), [전체 500개 예측 보기](runs/controlled/report.html)
- 추론용으로 사용할 학습 checkpoint: `runs/controlled/best.pt`. 실제 수식 checkpoint는 `runs/pilot/best.pt`.
- 실제 수식 원 데이터와 revision/checksum은 `data/raw/provenance.json`에 기록했다.

![통제된 수식 학습 및 대조 실험](runs/controlled/learning_curves.png)

![통제된 수식 예측 사례](runs/controlled/prediction_examples.png)
"""
    (ROOT/"RESULTS.md").write_text(text,encoding="utf-8")
    print(ROOT/"RESULTS.md")


if __name__=="__main__":
    main()
