# 손글씨 데이터

[Google Research MathWriting 2024](https://github.com/google-research/google-research/tree/master/mathwriting)의 디지털 펜 좌표를 이미지로 변환했다. 원본 라이선스는 **CC-BY-NC-SA-4.0**이다. 학교 프로젝트의 비상업적 실험에 사용했다. 상업적 배포에 그대로 사용할 수 있는 데이터로 취급하지 않는다.

원본 train/valid/test 구분을 유지한다. 지원 어휘 141개와 1–32토큰 길이로 필터링하고 그리스 문자를 포함한 식의 비중을 높였다. 원본의 synthetic expression 파티션은 사용하지 않았다.

| 부분 | 수 |
|---|---:|
| 사람이 쓴 학습 수식 | 15,000 |
| 학습용 단일 기호 | 2,119 |
| 인쇄 수식 replay | 2,400 |
| 전체 학습 | 19,519 |
| 검증 / 시험 | 각 800 |
| 그리스 문자가 있는 검증 / 시험 | 각 300 |

이미지는 높이 64px, 너비 최대 512px이다. 실제 필기 궤적을 안티앨리어싱하여 그리고 배경 밝기를 보정했다. 촬영 변화 시험은 같은 필기에 조명 그라데이션·작은 회전·잡음을 적용한 별도 이미지다. **공책을 촬영한 실제 사진은 아니다.** 같은 식을 다른 필기자가 쓴 경우는 허용한다. 공개 writer ID가 없으므로 모든 필기자가 분리된 시험이라는 주장은 하지 않는다.

`audit.json`은 원본 파티션, InkML ID와 픽셀 해시 중복을 검사한 결과다. `summary.json`에는 샘플 선택 seed, 원본 스캔 수, 필터 조건과 실제 분할 수가 있다. 다운로드 원본의 주소·SHA-256·GCS MD5는 `../data_mathwriting/raw/*.tgz.json`에 기록했다. 인쇄 replay의 출처는 `../data_physics/`와 `../data/raw/provenance.json`이다.

프로젝트 폴더에서 재생성한다. 전체 원본은 약 2.9GiB이며 이미지 파일과 체크포인트 공간이 추가로 필요하다.

```powershell
.venv/Scripts/python.exe download_mathwriting.py
.venv/Scripts/python.exe prepare_mathwriting.py
.venv/Scripts/python.exe audit_mathwriting.py
```

평가와 모델 선택은 `../choose_handwriting_candidate.py`, `../select_physics_decoder.py`, `../evaluate_handwriting.py`에서 수행한다. 시험 데이터는 구조·체크포인트·beam width 선택에 사용하지 않는다. 모델과 UI가 실제로 받는 이미지 전처리를 최종 평가에도 적용한다.
