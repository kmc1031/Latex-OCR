# 기호 분할 수식 OCR

실험 결과는 [REPORT.md](REPORT.md), 전체 독립 시험은 [gallery.html](gallery.html)에 있다. 이전 모델·데이터는 덮어쓰지 않았다.

기호 분류 CNN, 공간 관계 MLP, 기호 영역 검출 CNN을 각각 학습한다. 자동 분할은 연결요소+일부 기호 병합과 CNN 검출을 검증에서 비교하여 고른다. Attention·CTC를 사용하지 않는다. 일반 인식 경로는 이미지만 받으며, 정답 bbox·기호·획 순서는 평가의 명시적인 진단 조건에만 사용한다.

이 폴더에서 실행한다. Python 환경은 상위 프로젝트의 `.venv`를 공유한다.

```powershell
../.venv/Scripts/python.exe download_crohme.py
../.venv/Scripts/python.exe prepare.py
../.venv/Scripts/python.exe audit.py
../.venv/Scripts/python.exe train_symbols.py
../.venv/Scripts/python.exe train_relations.py
../.venv/Scripts/python.exe train_detector.py --epochs 8
../.venv/Scripts/python.exe evaluate.py --select
../.venv/Scripts/python.exe compare_previous.py
../.venv/Scripts/python.exe evaluate_transfer.py
../.venv/Scripts/python.exe make_report.py
```

이번 검출기 실행은 국소 문맥 파일럿 약 2회 학습 후 `models/detector_local.pt`에서 넓은 문맥 검출기로 전이하여 8회 학습했다. 실제 기록은 `models/detector_local_history.json`과 `models/detector_history.json`이다. 정확히 같은 전이 단계에는 다음 명령을 쓴다.

```powershell
../.venv/Scripts/python.exe train_detector.py --epochs 8 --warm-start models/detector_local.pt
```

검증·시험은 같은 수식의 다른 필체가 학습/검증에 섞이지 않도록 구성한다. 원래 공식 시험은 유지하고, 학습에 있던 수식과 새로운 수식의 성능을 따로 보고한다. 지원 어휘·구조의 부분집합이므로 전체 CROHME benchmark 성적은 아니다. 실제 공책 사진 성능은 측정하지 않았다.

이미지 한 장을 인식한다. `--prepared`는 이 실험의 정규화된 이미지에만 사용한다.

```powershell
../.venv/Scripts/python.exe predict.py my_formula.png --overlay segments.png
```

기존 `../start_ui.cmd`를 열고 **기호 분할 · CNN (실험)**을 선택할 수도 있다. 별도 모델 선택을 제공하므로 기존 모델을 계속 사용할 수 있다.

`models/symbols.pt`, `models/relations.pt`, `models/detector.pt`, `models/settings.json`이 추론에 쓰인다. 기호 학습용 대용량 `glyphs.npy`는 추론에 필요하지 않다. 데이터 출처와 라이선스는 결과 보고서 및 `raw/provenance.json`에 기록했다.
