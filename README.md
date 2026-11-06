# LaTeX OCR 연구

손글씨 수식을 기호로 분할하고 CNN·LSTM으로 LaTeX를 생성하는 인공지능 프로젝트입니다.

**현재 최신 모델의 정확도는 낮습니다. 새 MathWriting 수식 317개 중 1개만 전체 일치(0.32%)했으며, 실제 문서 작성에 바로 쓰기 위한 제품이 아닙니다. 직접 촬영한 종이 사진은 학습·시험에 사용하지 않았습니다.**

## 최신 통합 실험

| 항목 | 실제 결과 |
|---|---:|
| 실제 MathWriting 학습 / 검증 / 시험 | 3,872 / 363 / 317 |
| 합성 학습 보조 이미지 | 2,000 |
| CNN 기호 클래스 | 70 |
| 본학습 / 검증으로 선택한 에포크 | 26 / 18 |
| 새 시험 수식 전체 일치율 | 1/317 (0.32%) |
| 정규화 토큰 편집 거리 | 0.6494 |
| CPU 한 장당 중앙값 / p95 | 22.84ms / 37.45ms |

학습 자료 100개를 재인식하면 56%였지만 새로운 시험에서는 0.32%로, 과적합과 일반화 부족이 확인됐습니다. 개선 제안은 아직 이 결과에 적용하지 않았습니다. 브라우저 실클릭 검증은 권한 제한으로 미실시했으며, 로컬 API·수식 렌더링·복사 로직 단위 검사를 수행했습니다.

- [최종 보고서](math_ocr/unified_experiment/REPORT.md)
- [학습 곡선](math_ocr/unified_experiment/summary.png)
- [시험 전체 예측 JSON](math_ocr/unified_experiment/mathwriting_test.json)
- [시험 전체 결과표](research/test_results.html)
- [프로토콜](math_ocr/unified_experiment/PROTOCOL.md)

## 구성

```text
이미지 → 전처리 → 연결 성분·기하 병합 → CNN 기호 확률·위치
       → 양방향 LSTM 인코더 → LSTM 디코더 → LaTeX 표시·복사
```

| 위치 | 내용 |
|---|---|
| `math_ocr/unified_experiment/` | 최신 단일 통합 실험, 학습·추론·평가 코드와 실제 결과 |
| `math_ocr/segmentation/` | 이전 CROHME 기호 CNN·관계 MLP·구조 파서 연구 |
| `math_ocr/physics/` | 이전 전체 이미지 CNN–LSTM 손글씨 연구 |
| `math_ocr/comparison/` | 이전 분수 표기 비교 |
| `math_ocr/planned_experiment/` | 이전 통제된 합성 수식 실험 |

실험마다 데이터와 범위가 다르므로 보고서의 수치를 그대로 순위 비교하면 안 됩니다. 이전 결과와 최신 결과를 분리해 보관합니다.

## 추론과 화면 실행

Python 3.13, PyTorch 2.7.1, NumPy·Pillow·SciPy·Matplotlib 환경에서 실행했습니다. 실제 패키지 버전은 [환경 기록](math_ocr/unified_experiment/environment.json)을 확인하세요.

```powershell
python -m venv math_ocr/.venv
math_ocr/.venv/Scripts/python.exe -m pip install torch==2.7.1 numpy pillow scipy matplotlib pyarrow requests
cd math_ocr/unified_experiment
../.venv/Scripts/python.exe recognize.py "수식이미지.png"
../.venv/Scripts/python.exe web.py --port 8768
```

브라우저에서 http://127.0.0.1:8768 을 열어 이미지를 넣고 결과를 확인합니다. 추론용 가중치 `final_model.pt`를 포함하므로 추론 시 원본 학습 데이터는 필요하지 않습니다. GPU 학습용 CUDA wheel은 장치에 맞게 별도 설치해야 합니다.

## 재현 범위

소스, 설정, 데이터 선정 목록, 학습 이력, 시험 예측, 추론 전용 가중치를 포함합니다. 가상환경, 원본 데이터, 렌더링 이미지 전체, 대용량 캐시와 학습 optimizer checkpoint는 포함하지 않습니다. 일부 이전 보고서의 로컬 이미지·갤러리 링크는 전체 데이터를 다시 준비해야 사용할 수 있습니다.

전체 재학습에는 원본을 내려받아 데이터 준비 코드를 실행해야 합니다. 새 clone이 곧바로 모든 데이터와 캐시를 가진 상태는 아닙니다. 현재 고정 시험 결과를 보존하며 재학습하려면 별도 실행 디렉터리/데이터 목록을 사용해야 합니다.

## 출처와 작성

- [MathWriting 공식 자료](https://github.com/google-research/google-research/tree/master/mathwriting): CC BY-NC-SA 4.0. 원본 URL·해시는 `math_ocr/data_mathwriting/raw/`의 메타데이터에 있습니다.
- CROHME: 비상업적 학술 이용 조건과 배포본 URL·해시는 [출처 기록](math_ocr/segmentation/raw/provenance.json)을 확인하세요.
- 초기 인쇄 수식: [IM2LATEX-100K](https://im2markup.yuntiandeng.com/). 사용한 export/revision은 `download_data.py`에 기록했습니다.
- 생성형 AI를 활용하여 코드·실험 설계·분석을 작성했습니다. 결과 수치는 저장된 실행 이력과 예측에서 산출했습니다.

외부 데이터의 이용 조건은 각 원출처를 따릅니다. 이 저장소는 별도 코드 라이선스를 지정하지 않았습니다. 비밀 키·개인 문서·가상환경은 배포 대상에서 제외했습니다.
