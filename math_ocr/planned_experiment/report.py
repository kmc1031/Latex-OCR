"""Produce measured report, CPU timing, inference parity checks and gallery."""
import json,time,platform
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
import torch
import run as e
from recognize import Recognizer

def main():
    r=json.loads((e.ROOT/'results.json').read_text());audit=json.loads((e.ROOT/'data_audit.json').read_text());rows=json.loads((e.ROOT/'test.json').read_text());seeds=[42,43,44]
    selected=r['selected_lstm'];expected=json.loads((e.ROOT/f'{selected}_42_predictions.json').read_text());lookup={p['id']:p for p in expected}
    torch.set_num_threads(1);model=Recognizer();samples=[]
    for row in rows[:10]:model(e.ROOT/row['image'])
    # File -> segmentation -> CNN -> free-running LSTM; no labels enter Recognizer.
    for repeat in range(3):
        for row in rows:
            p=model(e.ROOT/row['image']);assert p['tokens']==lookup[row['id']]['pred'],row['id'];samples.append(p)
    timing={'device':'CPU','threads':1,'n_images':len(rows),'repeats':3,'n_measurements':len(samples),'warmup':10,'median_seconds':float(np.median([p['seconds'] for p in samples])),'p95_seconds':float(np.percentile([p['seconds'] for p in samples],95)),'stage_means':{k:float(np.mean([p['stage_seconds'][k] for p in samples])) for k in ['read','segment','classify','decode']},'inference_parity_passed':True,'note':'Three timing passes; input file reads included; model load and display excluded.'}
    e.save(e.ROOT/'timing.json',timing)
    transfer=[];source=e.BASE.parent/'data_handwriting'/'test.jsonl'
    if source.exists():
        for line in source.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);ts=row['tokens']
            if not ts or len(ts)>64 or any(t not in e.VOCAB[4:] for t in ts):continue
            if sum(t in e.CLASSES or t=='\\frac' for t in ts)>12 or ts.count('\\frac')>1 or ts.count('^')+ts.count('_')>1:continue
            p=model(e.BASE.parent/'data_handwriting'/row['image'])
            transfer.append({'image':row['image'],'truth':ts,'pred':p['tokens'],'exact':ts==p['tokens'],'distance':e.edit(p['tokens'],ts)})
    transfer_metrics={'n':len(transfer),'exact_match':sum(x['exact'] for x in transfer)/max(1,len(transfer)),'token_error_rate':sum(x['distance'] for x in transfer)/max(1,sum(len(x['truth']) for x in transfer)),'seed':42,'scope':'Historical MathWriting digital-ink test subset; 23-symbol token filter; <=12 visible symbols, <=1 fraction, <=1 script. Not paper photos or new independent test.'}
    e.save(e.ROOT/'mathwriting_transfer.json',{'metrics':transfer_metrics,'predictions':transfer})
    def agg(name,path):
        vals=[]
        for n in seeds:
            item=r['runs'][f'{name}_{n}']
            for k in path:item=item[k]
            vals.append(item)
        return f'{100*np.mean(vals):.2f} ± {100*np.std(vals,ddof=1):.2f}'
    lines=['# 계획 기반 CNN → LSTM 실제 실행 결과','',
      '2026-11-05 실행. 아래 수치는 저장된 학습·시험 결과에서 자동 생성했다. ±는 시드 3회의 표본 표준편차이며 신뢰구간이 아니다.','',
      '## 실행 범위와 계획 대비 변경','',
      '- 21회 학습: C0/C1/C2, L0/L1, Fvertical/Fslash를 시드 42·43·44로 반복했다.',
      '- MathWriting 대신 이미 확보된 CROHME 기호를 사용했다. source 분할은 기존 train/val/test를 유지했다.',
      '- O 표본이 없어 24종 중 O를 제외한 23종을 사용했다. 0/O 혼동 실험은 미실시다.',
      '- 폰트 혼합 없이 실제 손글씨 조각을 정해진 배치에 합성했다. 자유로운 실제 필기 배치·종이 사진 성능으로 해석할 수 없다.',
      '- 팀원 촬영 240장 및 추가 환경 시험은 데이터가 없어 미실시다. 사진 없이 대체 정확도를 만들지 않았다.',
      '- 원본 기호 test는 이전 프로젝트에서 사용된 집합이다. 이번 합성 수식 내용은 새로 생성했으나 완전히 새로운 독립 benchmark로 주장하지 않는다.',
      '- CNN 학습 사진 추가 및 사진 검증 대신, 확보된 기호 검증과 합성 수식 검증으로 설정을 선택했다.',
      '- 검증 최적 epoch·설정 선택 후에 시험을 계산했다. 코드의 frozen_selection.json과 학습 기록 참조.',
      '- CPU 시간은 450장을 3회 측정했다.',
      '- 초기 bbox-crop 파일럿의 예시 검수에서 이웃 획 혼입을 발견해 정답 획 전용 뷰로 모든 데이터를 재생성하고 21회 재학습했다. 수식 문항은 파일럿과 같으며 완전 미접촉 시험으로 주장하지 않는다. 상위 폴더의 파일럿 결과는 최종 결과에 합산하지 않는다.','',
      f"데이터: 합성 수식 {audit['counts']}; 기호 개수는 data_audit.json에 클래스별 기록. 원본 수식 ID·합성 수식 내용·합성 이미지 해시의 집합 간 중복 검사를 통과했다.",'',
      '## CNN 기호 분류','', '| 조건 | 시험 정확도 (%) | 시험 Macro F1 (%) |','|---|---:|---:|']
    for name in ['C0','C1','C2']:lines.append(f"| {name} | {agg(name,['test','accuracy'])} | {agg(name,['test','macro_f1'])} |")
    cm=sum(np.array(r['runs'][f"{r['selected_cnn']}_{s}"]['test']['confusion']) for s in seeds)
    confusions=sorted([(int(cm[i,j]),e.CLASSES[i],e.CLASSES[j]) for i in range(len(e.CLASSES)) for j in range(len(e.CLASSES)) if i!=j],reverse=True)
    e.save(e.ROOT/'confusions.json',{'classes':e.CLASSES,'matrix_sum_over_3_seeds':cm.tolist(),'top_errors':confusions[:15]})
    lines+=['',f"검증으로 선택한 CNN: **{r['selected_cnn']}**. C0=MLP, C1=CNN, C2=CNN+증강.",'','오분류 상위 항목(동일 시험 기호를 시드 3회 평가한 합계이며 독립 표본 수가 아님):','','| 정답 | 예측 | 오류 횟수 합계 |','|---|---|---:|']
    for count,a,b in confusions[:8]:lines.append(f'| `{a}` | `{b}` | {count} |')
    lines+=['','l의 시험 원본은 9개이므로 1/l 구분 결과의 표본 한계를 함께 고려해야 한다. O는 없어 0/O는 측정하지 않았다.','','## 위치 정보와 전체 인식','','| 조건 | 전체 일치율 (%) | 정규화 토큰 편집 거리 ×100 |','|---|---:|---:|']
    for name in ['L0','L1']:lines.append(f"| {name} | {agg(name,['test','exact_match'])} | {agg(name,['test','token_error_rate'])} |")
    lines+=['',f'검증으로 선택한 LSTM: **{selected}**. L0는 위치 네 값을 0으로 만들고, L1은 실제 값을 사용한다. 입력 차원·파라미터 수는 같다.','', '## 분수 교차 시험','','| 학습 | 상하 시험 일치율 (%) | 슬래시 시험 일치율 (%) |','|---|---:|---:|']
    for name in ['Fvertical','Fslash']:lines.append(f"| {name} | {agg(name,['test','by_type','vertical','exact_match'])} | {agg(name,['test','by_type','slash','exact_match'])} |")
    lines+=['','각 시험은 같은 내용의 75쌍이다. 두 표기 모두 정답은 동일한 \\frac 토큰열이다. 이 결과를 실제 종이 사진의 입력 권장 근거로 바로 확대하지 않는다. 두 표기의 학습·시험 배치가 다르므로 교차 시험 0%만으로 수학적 구조를 전혀 학습하지 않았다고 단정할 수는 없다.','', '## 기호 분할','','| 조건 | 정밀도 | 재현율 | F1 |','|---|---:|---:|---:|']
    for name,s in r['segmentation'].items():lines.append(f"| {name} | {s['precision']:.4f} | {s['recall']:.4f} | {s['f1']:.4f} |")
    lines+=['','S0=연결 성분, S1=동일 전처리+기하 병합. IoU≥0.5 최대 일대일 대응.','', '## 선택 모델의 단계 진단','','| 조건 | 전체 일치율 (%) |','|---|---:|']
    for mode,label in [('oracle','정답 위치·기호'),('gt','정답 위치+CNN')]:lines.append(f"| {label} | {agg(selected,['diagnostics',mode,'exact_match'])} |")
    lines.append(f"| 자동 분할+CNN | {agg(selected,['test','exact_match'])} |")
    lines+=['','정답 기호 one-hot 입력은 진단용이며 확률 분포가 달라 엄밀한 상한은 아니다. 단계 오류를 단순 합산하지 않는다.','', '## 구조별 결과','','| 종류 | 선택 모델 일치율 (%) |','|---|---:|']
    for k in ['basic','sup','sub','vertical','slash']:lines.append(f"| {k} | {agg(selected,['test','by_type',k,'exact_match'])} |")
    lines+=['','## 실행 시간·검증','',f"GPU 학습: {r['gpu']}, torch {r['torch']}. 전체 학습·평가 경과 {r['seconds']/60:.1f}분.",f"CPU 이미지당 중앙값 {timing['median_seconds']*1000:.2f}ms, 95백분위 {timing['p95_seconds']*1000:.2f}ms. 모델 로딩·화면 표시 제외.",'450장 모두 이미지 전용 CLI와 배치 평가의 출력 토큰이 동일함을 확인했다. CLI는 정답 파일이나 기호 좌표를 읽지 않는다.','', '## 재현','', '```powershell','../.venv/Scripts/python.exe run.py','../.venv/Scripts/python.exe report.py','../.venv/Scripts/python.exe recognize.py images/test/이미지이름.png','```','', 'run.py를 다시 실행하면 이 폴더의 학습 결과를 새로 생성한다. 기존 상위 프로젝트 모델은 수정하지 않는다.','', '학습 이력: *_history.json, 시험 예측: *_predictions.json, 전체 수치: results.json, CPU 시간: timing.json, 클래스별 표본·중복 검사: data_audit.json.','', '자료 출처·사용 조건: ../segmentation/raw/provenance.json 및 ../segmentation/REPORT.md. 구현은 생성형 AI로 작성했으며 클래스·정답 좌표는 기존 CROHME 자료에서 가져왔다.']
    lines+=['','## 실제 필기 배치로의 추가 전이 진단','',f"기존 MathWriting 디지털 획 시험 800장 중 어휘·길이 조건을 만족한 {len(transfer)}장에 고정된 시드42 모델을 적용했다. 전체 일치율 {transfer_metrics['exact_match']*100:.2f}%, 토큰 편집 거리 {transfer_metrics['token_error_rate']:.4f}.",'선택 조건과 전체 예측은 mathwriting_transfer.json 참조. 이 집합은 이전 프로젝트에서 사용한 시험 자료이며, 이번 결과로 학습 설정을 수정하지 않았다. 단일 시드 추가 진단이며 실제 종이 사진 시험은 아니다.']
    text='\n'.join(lines).replace('../segmentation/','../../segmentation/')
    text=text.replace('../.venv/Scripts/python.exe run.py','../../.venv/Scripts/python.exe ../run.py').replace('../.venv/Scripts/python.exe report.py','../../.venv/Scripts/python.exe ../report.py').replace('../.venv/Scripts/python.exe recognize.py','../../.venv/Scripts/python.exe ../recognize.py')
    (e.ROOT/'REPORT.md').write_text(text,encoding='utf-8')
    # Save a small image gallery with exact-match result and predicted text.
    chosen=[]
    for k in ['basic','sup','sub','vertical','slash']:
        rs=[x for x in rows if (x['style'] if x['kind']=='fraction' else x['kind'])==k];chosen+=rs[:2]
    canvas=Image.new('RGB',(900,170*len(chosen)),'white');draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(chosen):
        im=Image.open(e.ROOT/row['image']).convert('RGB');canvas.paste(im,(5,i*170));p=lookup[row['id']];draw.text((275,i*170+10),f"{row['kind']}/{row['style']} exact={p['exact']}",fill='black');draw.text((275,i*170+35),'GT: '+' '.join(p['truth']),fill='black');draw.text((275,i*170+60),'PR: '+' '.join(p['pred']),fill='black')
    canvas.save(e.ROOT/'examples.png');print('\n'.join(lines[:35]));print('REPORT COMPLETE')

if __name__=='__main__':main()
