> 이전 연구 개요입니다. 최신 통합 결과는 [저장소 첫 화면](../README.md)과 [최신 보고서](unified_experiment/REPORT.md)를 확인하세요.

# CNN과 LSTM 수식 OCR 프로젝트

전체 이미지를 읽는 CNN–BiLSTM–LSTM 모델과, 기호 분할 → 기호 CNN → 공간 관계 MLP → 구조 파서 모델을 구현했다. Self-attention, cross-attention, transformer, CTC는 사용하지 않는다.

최신 결과는 **[기호 분할·조합 모델 비교](segmentation/REPORT.md)** 와 [시험 482개 전체 결과](segmentation/gallery.html)에 있다. 전체 이미지 방식의 이전 손글씨 실험은 [손글씨 물리 수식 비교](physics/HANDWRITING_REPORT.md), 그 이전 실험은 [슬래시 표기 성능 비교](comparison/REPORT.md), 초기 기록은 [RESULTS.md](RESULTS.md), 제출용 개요는 [프로젝트 계획서](수식_OCR_프로젝트_계획서.md)에 있다.

## 선택 모델 사용

### 이미지 편집 UI

`start_ui.cmd`를 더블클릭하면 서버와 브라우저가 열린다. 또는 프로젝트 폴더에서 다음 명령을 실행한다. 주소는 **http://127.0.0.1:8765**, 종료는 실행 창에서 `Ctrl+C`.

```powershell
.venv/Scripts/python.exe ui_server.py --open
# 또는
./start_ui.ps1 --open
```

이미지를 업로드·드롭하거나 `Ctrl+V`로 붙여 넣는다. 드래그로 수식 영역을 선택하고 테두리로 크기·위치를 조절한다. 선택한 영역은 바로 인식할 수 있으며, **자르기 적용**을 누르면 그 영역을 기준으로 추가 편집할 수 있다. 회전, 가로·세로 shear, 가로·세로 배율을 지원한다. Affine 변환은 중심 기준으로 적용하며 여백을 확장해 글자가 잘리지 않게 한다. 보정값을 바꾸면 기존 선택은 해제되므로 보정 후 수식 영역을 다시 선택한다. 사진의 원근 보정은 포함하지 않는다.

- 실행 취소(`Ctrl+Z`), 보정값 초기화, 원본 복원, 확대/축소.
- 선택 영역만 모델에 전달; 전처리한 실제 입력 이미지와 원시 결과 확인.
- LaTeX 수식 미리보기, 코드 수정·복사, `.tex` 및 편집 PNG 저장.
- 외부 서버·CDN 없이 로컬 처리. 별도 웹 패키지 설치 불필요.
- 기본값은 실제 손글씨로 추가 학습한 물리 기호 실험 모델이다. UI에 측정한 손글씨 시험 정확도를 표시한다. **현재 손글씨 실사용 목표에는 미달했다.** 모델 테스트와 결과 확인용으로 제공한다.
- 새 조합 모델은 **기호 분할 · CNN (실험)** 프로필로 선택한다. 검증에서 연결요소 분할이 학습한 CNN 검출기보다 좋아 현재 기본 분할 방식으로 사용한다. 인식 후 **기호 분할 결과**에서 영역을 확인한다. 새 시험 482개에서 이전 전체 이미지 모델 5.2%, 새 모델 25.3%였으며, 실제 공책 사진 성능은 검증하지 않았다.
- 모델 선택에서 기존 P로 바꾸면 슬래시 분수가 포함된 짧은 기본 인쇄 수식에 기존 모델을 그대로 사용할 수 있다.

업로드는 20MB까지이며 편집 시 긴 변을 최대 2400px로 축소한다. 큰 affine 변환 결과는 긴 변 4096px / 약 1600만 픽셀 내로 축소한다. 모델 입력은 높이 64px, 최대 너비는 손글씨 모델 512px / 기존 P 384px이다. 손글씨 모드는 종이 배경의 밝기 차이를 완화한다. 인식 결과의 문법 유효성은 정확도를 보장하지 않는다. MathML 미리보기는 최초 인식 결과를 표시하며 코드 직접 수정 시에는 갱신되지 않는다.

서버는 `127.0.0.1`에서만 접속을 받는다. 포트 변경은 `ui_server.py --port 8766`. 체크포인트는 `--checkpoint`로 지정할 수 있으나 `predict_best.py`와 같은 seq2seq 추론 패키지 형식이어야 한다.

검증: `node --test ui/editor.test.js`(좌표·변환), 서버 실행 후 `.venv/Scripts/python.exe test_ui_api.py`(실제 모델 API). `test_ui_browser.cjs`는 Playwright/Edge가 있는 환경에서 업로드·crop·affine·실행 취소·인식·다운로드·모바일 배치를 검증한다. 재현용 입력은 `tmp/ui_fixture.png`, 실행 결과는 `ui_qa.json` 및 `ui_preview*.png`에 있다.

최종 손글씨 모델과 보고서가 있는 환경에서는 `.venv/Scripts/python.exe qa_handwriting.py`로 서버 시작 → 기본/손글씨 API 검사 → 브라우저·전체 시험 갤러리 검사 → 서버 종료를 한 번에 수행한다. 포트 8765에 다른 UI 서버가 실행 중이면 먼저 해당 실행 창에서 종료한다.

### 명령줄 추론

이 프로젝트 폴더에서 다음 명령으로 잘라낸 수식 이미지 한 장을 인식할 수 있다. 입력 표기는 비교에서 선택한 모델의 조건에 맞춘다.

```powershell
.venv/Scripts/python.exe predict_best.py "my_formula.png"
.venv/Scripts/python.exe predict_best.py "my_formula.png" --latex
```

`--latex`는 인식한 구조를 LaTeX로 변환한다. `--checkpoint comparison/best_model.pt`로 기존 P를 지정할 수 있다. 기본값은 `physics/best_model.pt`이며, 이 파일이 없는 초기 환경에서는 P를 사용한다.

## 손글씨 물리 수식 학습

기호 분할 방식의 후속 실험은 **[기호 CNN·공간 관계 조합 결과](segmentation/REPORT.md)** 에 있다. `segmentation/`에서 기호 분류 CNN, 영역 검출 CNN, 공간 관계 MLP와 구조 파서를 학습·평가한다. UI에서는 **기호 분할 · CNN (실험)** 프로필로 사용할 수 있고, 인식 후 기호별 검출 영역을 확인할 수 있다. 이전 모델과 같은 새 시험 이미지에서 비교하며, 정답 기호 위치를 제공한 진단과 실제 자동 분할 성능을 분리해 보고한다.

출력 어휘를 30개에서 **145개(특수 토큰 4개 포함)** 로 확장했다. 영문 대소문자, 0–9·소수, 그리스 문자 37종, 위·아래 첨자, 쌓인 분수·슬래시 분수, 근호, 벡터와 기본 함수·미적분 기호를 포함한다. 어휘에 포함된다는 것은 그 기호의 높은 정확도를 보장한다는 뜻은 아니다.

MathWriting의 실제 손글씨 수식 15,000개와 단일 기호 2,119개, 인쇄 replay 2,400개로 학습했다. 검증·시험은 각 800개다. 디지털 펜 좌표를 이미지로 바꾼 부분집합이며 실제 공책 사진의 성능은 측정하지 않았다. 출처·라이선스·필터 조건은 [데이터 설명](data_handwriting/README.md), 중복 검사는 [audit.json](data_handwriting/audit.json)에 있다.

고정 위치의 encoder 특징과 기호 존재 여부·출력 길이 보조 분류를 추가한 후보를 비교했다. 고정 위치 보완 후보는 추론 시 **모델이 예측한 길이**로 세 위치를 정한다. 더 깊은 CNN 후보는 residual block 2개와 압축하지 않은 8개 위치 특징을 추가했다. Query/key나 학습된 attention 가중치, CTC는 없다. 선택한 구조·설정은 [candidate_selection.json](physics/candidate_selection.json), beam 탐색 설정은 [decoder_selection.json](physics/decoder_selection.json)에 저장한다. 시험 데이터는 선택에 쓰지 않는다.

다음은 실행 순서다. 원본 데이터 다운로드와 추가 학습에는 상당한 공간과 GPU 시간이 필요하다. 이미 생성한 데이터와 체크포인트가 있으면 다운로드·학습을 반복하지 않아도 된다. 전체 학습 이력과 실제 실행 인자는 각 `runs/*/history.json`, `config.json`에 있다.

```powershell
# 인쇄 기호 사전 학습용 데이터: 초기 data 및 comparison 데이터가 있는 환경
.venv/Scripts/python.exe prepare_physics.py
.venv/Scripts/python.exe prepare_physics_real.py
.venv/Scripts/python.exe prepare_physics_atoms.py
.venv/Scripts/python.exe audit_physics.py

# 실제 손글씨 데이터: 원본 약 2.9GiB
.venv/Scripts/python.exe download_mathwriting.py
.venv/Scripts/python.exe prepare_mathwriting.py
.venv/Scripts/python.exe audit_mathwriting.py

# 순차 전이 학습 후보. 중간에 중단한 단계도 원래 학습률 schedule을 표시한다.
.venv/Scripts/python.exe train_physics.py --run physics_v1 --epochs 50
.venv/Scripts/python.exe train_physics.py --handwriting --data-dir data_handwriting --init runs/physics_v1/best.pt --run physics_handwriting_v1 --epochs 40 --lr 0.0005
.venv/Scripts/python.exe train_physics.py --handwriting --data-dir data_handwriting --init runs/physics_handwriting_v1/best.pt --run physics_handwriting_local --epochs 40 --lr 0.0004 --local-features 96 --balanced-ce
.venv/Scripts/python.exe train_physics.py --handwriting --data-dir data_handwriting --init runs/physics_handwriting_local/best.pt --run physics_handwriting_deep --epochs 35 --lr 0.00035 --local-features 96 --balanced-ce --raw-context --refinement-blocks 2

# 검증으로 모델과 디코더 선택 → 미사용 시험 평가 → 보고서
.venv/Scripts/python.exe choose_handwriting_candidate.py
.venv/Scripts/python.exe evaluate_handwriting.py
.venv/Scripts/python.exe diagnose_handwriting.py
.venv/Scripts/python.exe make_handwriting_report.py
```

초기 단계 일부는 검증 정체를 보고 중단했다. 실제 실행된 epoch는 각 run의 `history.json`을 확인한다. 정확한 재개에는 각 run의 `last.pt`와 `config.json` 인자를 사용한다. 최종 시험을 다시 보고 모델 선택을 바꾸면 독립 시험이라는 해석을 유지할 수 없다.

## 주요 위치

- `model.py`, `train.py`: CNN–BiLSTM–LSTM과 기본 학습.
- `train_practical.py`, `augmentation.py`: 여러 폰트·변형·중첩 수식을 활용한 추가 학습.
- `comparison/best_model.pt`: 검증 성능으로 선택한 추론용 모델.
- `physics/best_model.pt`: 현재 UI 기본 손글씨 추론 패키지.
- `physics/HANDWRITING_REPORT.md`, `physics/handwriting_gallery.html`: 실제 손글씨 비교와 시험 전체 결과.
- `data_handwriting/`, `data_mathwriting/raw/`: 실제 손글씨 이미지와 공식 원본·출처 기록.
- `comparison/REPORT.md`, `comparison/prediction_viewer.html`: 성능 비교와 분수 100개 예측.
- `comparison/views/`, `comparison/practical_images/`: 비교용 수식 목록과 이미지.
- `runs/`: 학습 기록과 전체 checkpoint.
- `data/`, `data_controlled/`: 초기 공개 데이터 및 통제된 합성 데이터.
- `.venv/`: 프로젝트 전용 Python 환경.

## 초기 모델 P의 구조

| 단계 | 설정 |
|---|---|
| 입력 | grayscale 64px 높이, 종횡비 유지, 최대 너비 384px, 흰 배경을 0으로 변환 |
| CNN | 3×3 conv, 채널 24→48→96→128, 최종 높이 8, 너비 W/4 |
| 열별 특징 | 각 열의 채널×세로 위치를 concatenate, Linear→192 |
| Encoder | 양방향 2층 LSTM, 방향별 hidden 192 |
| Context | 최상위 층의 양방향 최종 hidden 연결, **384차원** |
| Decoder | embedding 96, context를 매 단계 같은 값으로 전달, 2층 LSTM hidden 256 |
| 출력 | Linear→LaTeX 토큰, BOS에서 시작하여 EOS까지 greedy decoding |

학습은 teacher forcing + PAD를 제외한 cross-entropy를 쓴다. 보고하는 정확도는 teacher forcing 없이 측정한다. 가로 padding 영역은 packed sequence로 encoder의 최종 상태에서 제외한다. 세로 방향 평균 pooling 대신 세로 위치를 보존한다.

## 파일

- `model.py`: 모델 정의.
- `download_data.py`: 저자 Hugging Face 배포본의 고정 revision 다운로드, SHA-256 검사.
- `prepare_data.py`: 실제 IM2LATEX 짧은 수식 선택, 전처리, 중복 제거.
- `generate_controlled.py`: 사칙연산·첨자·분수·근호의 통제된 합성 수식 생성.
- `train.py`: 학습, validation으로 checkpoint 선택, 자유 생성 평가, 이미지 대조 실험.
- `overfit_check.py`: 실제 수식 64개 암기 가능 여부 검사. 일반화 평가가 아님.
- `predict.py`: 새 이미지 한 장 추론.
- `make_report.py`: 학습 그래프, 예측 이미지, Markdown 및 HTML 결과표 생성.
- `audit_data.py`: 저장된 이미지 checksum·크기·split 중복을 독립적으로 재검사.
- `summarize_experiments.py`: 세 실험을 종합한 결과 보고서 생성.
- `runs/pilot`: 실제 IM2LATEX 실험.
- `runs/controlled`: 통제된 합성 수식 실험.
- `runs/overfit_check`: 암기 진단.

## 슬래시 비교 다시 실행

```powershell
.venv/Scripts/python.exe prepare_slash_comparison.py
.venv/Scripts/python.exe prepare_practical_data.py
.venv/Scripts/python.exe run_slash_comparison.py
.venv/Scripts/python.exe evaluate_slash_comparison.py
```

기본 A 모델은 초기 실험의 `runs/controlled/best.pt`를 재사용한다. 실행 기록이 있는 비교 조건은 건너뛰므로 새로 학습하려면 새 run 이름을 지정한다.

## 다시 실행

PowerShell에서 이 폴더를 작업 디렉터리로 사용한다. `.venv`는 이번 실험 전용이며 기존 과제 환경과 분리되어 있다.

```powershell
# 새 환경이 필요한 경우 (Python 3.13, NVIDIA CUDA)
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe numpy pillow matplotlib pyarrow requests

# 실제 IM2LATEX: 다운로드 약 337MB
.venv/Scripts/python.exe download_data.py
.venv/Scripts/python.exe prepare_data.py --train 6000 --val 500 --test 500 --long-test 250
.venv/Scripts/python.exe train.py --run pilot --epochs 24 --batch-size 32 --lr 0.001
.venv/Scripts/python.exe make_report.py --run pilot

# 통제된 수식
.venv/Scripts/python.exe generate_controlled.py
.venv/Scripts/python.exe train.py --run controlled --data-dir data_controlled --epochs 40 --batch-size 32 --lr 0.001 --max-decode 48
.venv/Scripts/python.exe make_report.py --run controlled

# 암기 진단
.venv/Scripts/python.exe overfit_check.py
.venv/Scripts/python.exe audit_data.py
.venv/Scripts/python.exe summarize_experiments.py

# 새 수식 이미지 인식 (원하는 checkpoint 명시)
.venv/Scripts/python.exe predict.py "my_formula.png" --checkpoint runs/controlled/best.pt

# 학습 연장: epochs는 추가 횟수가 아니라 마지막 epoch 번호
.venv/Scripts/python.exe train.py --run pilot --resume runs/pilot/last.pt --epochs 48 --lr 0.0003
```

실행 당시 직접 의존성 버전은 `requirements.lock.txt`, 모델/optimizer/어휘/학습 기록은 checkpoint에 저장했다. seed는 고정했지만 cuDNN benchmark를 켰으므로 GPU 환경에 따라 비트 단위로 같은 결과를 보장하지 않는다. 마지막 checkpoint와 최고 validation checkpoint를 별도로 보관한다.

## 해석 범위

실제 IM2LATEX 실험은 원본 전체 benchmark가 아닌 저자 HF 배포본의 **길이 3–32토큰 부분집합**이다. 통제된 실험은 train/test 수식 문자열과 이미지는 다르지만 생성 템플릿 계열·폰트를 공유한다. 통제된 실험의 정확도를 임의의 수식이나 손글씨 정확도로 해석하면 안 된다.

완전 일치율은 전체 수식을 맞힌 비율이다. 편집 유사도는 토큰 편집 거리로 계산하며 기호별 정확도와 다르다. `control_blank`는 너비는 유지하고 이미지 내용만 없애며, `control_shuffled`는 batch 내에서 이미지와 너비를 함께 바꾼다. 별도 학습한 언어 모델 baseline과는 구분한다.

데이터 출처: [IM2LATEX 원 프로젝트](https://github.com/harvardnlp/im2markup), [사용한 HF 배포본](https://huggingface.co/datasets/yuntian-deng/im2latex-100k). 원본 파일 checksum과 revision은 `data/raw/provenance.json`에 있다.
