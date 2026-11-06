# 계획 기반 실험

최종 실험은 `isolated/REPORT.md`를 확인한다. run.py, recognize.py, report.py,
check.py의 기본 출력/입력 폴더는 isolated/다.

이 폴더 최상위에 남은 결과는 초기 bbox-crop 파일럿이다. 시각 검수에서
주변 획 혼입을 발견하여 채택하지 않았다. 최종 실험은 glyphs.npy의
두 번째 뷰(정답 획만 렌더링한 이미지)로 데이터를 새로 생성하고 재학습했다.
파일럿 결과를 최종 결과와 합산하지 않는다.

실행 위치: 이 폴더. 기존 상위 OCR 모델은 변경하지 않는다.

```powershell
../.venv/Scripts/python.exe run.py
../.venv/Scripts/python.exe check.py
../.venv/Scripts/python.exe report.py
../.venv/Scripts/python.exe recognize.py isolated/images/test/파일명.png
```

생성형 AI로 작성한 구현이다. 학습 데이터는 로컬 CROHME 기호 자료이며
원본 출처는 ../segmentation/raw/provenance.json에 있다. MathWriting은
고정된 모델의 추가 전이 진단에만 사용했다. 실제 종이 사진은 없다.
