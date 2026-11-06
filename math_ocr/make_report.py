"""Make static plots and an offline, self-contained prediction viewer."""
import argparse
import base64
from collections import Counter
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent


def pct(x):
    return f"{100*x:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pilot")
    args = parser.parse_args()
    run = ROOT / "runs" / args.run
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    data = Path(metrics["model"]["args"].get("data_dir", ROOT / "data"))
    if not data.is_absolute():
        data = ROOT / data
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    controlled = data.name == "data_controlled"
    if controlled:
        provenance_text = "자체 생성한 5종 문법 + Matplotlib/STIX, seed 20261022."
        dataset_name = "통제된 합성"
        scope_text = "학습과 같은 다섯 템플릿 계열 안에서 새로운 기호 조합을 인식하는 진단이다. 실제 논문 수식 benchmark가 아니다."
        split_text = "분할 간 동일 canonical LaTeX / 전처리 픽셀 hash 중복 없음. 생성 문법은 train/val/test에서 공유한다."
        length_text = "선택한 5종 문법. long_test는 학습에 없는 중첩 분수 조합."
    else:
        provenance = json.loads((data / "raw/provenance.json").read_text(encoding="utf-8"))
        provenance_text = f"[{provenance['repository']}](https://huggingface.co/datasets/{provenance['repository']}), revision `{provenance['revision']}`."
        dataset_name = "실제 IM2LATEX"
        scope_text = "저자 HF 배포본의 짧은 수식 부분집합 실험이며, 원 논문의 전체 benchmark와 직접 비교할 수 없다."
        split_text = "원본 train/val/test 소속 유지. 동일 canonical LaTeX / 파일명 / 전처리 픽셀 hash의 split 간 중복 없음."
        length_text = "학습/일반 평가 길이 3–32토큰. long_test는 33–56토큰."
    predictions = json.loads((run / "predictions_test.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_ce"] for r in history], color="#2664d9")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Token cross-entropy")
    axes[1].plot(epochs, [100*r["val_exact"] for r in history], label="Exact match")
    axes[1].plot(epochs, [100*r["val_similarity"] for r in history], label="Edit similarity")
    axes[1].set(title="Free-running validation", xlabel="Epoch", ylabel="Percent", ylim=(0, 100))
    axes[1].legend()
    names = ["Normal", "Blank image", "Shuffled image"]
    values = [metrics[f"control_{k}"]["normalized_similarity"]*100 for k in ["normal", "blank", "shuffled"]]
    bars = axes[2].bar(names, values, color=["#2664d9", "#a8b1be", "#ef9d52"])
    axes[2].bar_label(bars, fmt="%.1f%%")
    axes[2].set(title="Image conditioning control (same 256 targets)", ylabel="Edit similarity (%)", ylim=(0, 100))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(run / "learning_curves.png", dpi=160)
    plt.close(fig)

    # Illustrative examples: first four correct and first four incorrect in fixed test order.
    correct = [r for r in predictions if r["exact"]][:4]
    wrong = [r for r in predictions if not r["exact"]][:4]
    examples = correct + wrong
    canvas = Image.new("RGB", (1500, max(1, len(examples))*145 + 55), "#f6f8fc")
    draw = ImageDraw.Draw(canvas)
    fonts = Path("C:/Windows/Fonts")
    font = ImageFont.truetype(str(fonts / "consola.ttf"), 17)
    small = ImageFont.truetype(str(fonts / "consola.ttf"), 15)
    draw.text((20, 15), "HELD-OUT TEST: first four exact matches + first four errors (fixed test order)", font=font, fill="#24354b")
    for i, r in enumerate(examples):
        y = 55 + i*145
        image = Image.open(data / r["image"]).convert("RGB")
        image.thumbnail((440, 96))
        canvas.paste(image, (15, y+15))
        color = "#217c50" if r["exact"] else "#bb4141"
        draw.text((480, y+7), "CORRECT" if r["exact"] else "ERROR", font=font, fill=color)
        for j, (label, text) in enumerate([("GT: ", r["target"]), ("PR: ", r["prediction"])]):
            # Long outputs wrap without truncating the illustrated result.
            import textwrap
            lines = textwrap.wrap(label + text, width=100)[:3]
            for k, line in enumerate(lines):
                draw.text((480, y+35+j*47+k*16), line, font=small, fill="#263448")
    canvas.save(run / "prediction_examples.png")

    table_lines = []
    for name in ["train_probe", "val", "test", "long_test", "control_normal", "control_blank", "control_shuffled"]:
        m = metrics[name]
        table_lines.append(f"| {name} | {m['n']} | {pct(m['exact_match'])} | {pct(m['normalized_similarity'])} | {m['token_error_rate']:.3f} |")
    group_lines = []
    for name, m in metrics["test"]["groups"].items():
        if m["n"]:
            group_lines.append(f"| {name} | {m['n']} | {pct(m['exact_match'])} | {pct(m['normalized_similarity'])} |")
    top_confusions = Counter()
    for r in predictions:
        # Positional equal-length mismatches only; avoid pretending this is aligned token analysis.
        if len(r["target_tokens"]) == len(r["predicted_tokens"]):
            top_confusions.update((a,b) for a,b in zip(r["target_tokens"], r["predicted_tokens"]) if a != b)
    wall = sum(r["seconds"] for r in history)
    test = metrics["test"]
    lo, hi = test["exact_wilson_95"]
    report = f"""# Attention 없는 수식 OCR 파일럿 결과

{dataset_name} 수식으로 학습한 CNN → BiLSTM → 고정 길이 벡터 → LSTM decoder 실험이다.
모든 정확도는 정답 이전 토큰을 제공하지 않는 greedy decoding으로 측정했다.

## 실험 조건

- 모델: {metrics['model']['parameters']:,} parameters, attention 없음.
- 입력: 흑백 높이 64px, 종횡비 유지, 가로 최대 384px, 오른쪽 padding.
- 출력: LaTeX 어휘 {summary['vocab_size']}개. {length_text}
- 데이터: train {summary['splits']['train']['samples']}, validation {summary['splits']['val']['samples']}, test {summary['splits']['test']['samples']}, 별도 long_test {summary['splits']['long_test']['samples']}.
- {split_text}
- 학습: AdamW, LR {metrics['model']['args']['lr']}, batch {metrics['model']['args']['batch_size']}, {len(history)} epochs; validation 완전 일치율, 동률이면 edit similarity로 checkpoint 선택.
- 선택 checkpoint: epoch {metrics['checkpoint_epoch']}. epoch 합계 약 {wall/60:.1f}분 (초기 준비·최종 평가 시간 제외).
- GPU: {metrics['model']['gpu']}; 학습 중 PyTorch 최대 allocated memory {metrics.get('peak_gpu_allocated_mb',0):.0f}MiB.
- 데이터: {provenance_text}
- {scope_text}

## 자유 생성 평가

| 평가 집합 | 개수 | 수식 완전 일치율 | 정규화 편집 유사도 | 토큰 오류율 TER ↓ |
|---|---:|---:|---:|---:|
{chr(10).join(table_lines)}

시험 완전 일치 {test['exact_count']}/{test['n']}, Wilson 95% 구간 {pct(lo)}–{pct(hi)}. 이 구간은 선택된 부분집합의 표본 불확실성만 나타내며 다른 폰트·손글씨로의 일반화를 보장하지 않는다.

정규화 편집 유사도는 `mean(1 - edit_distance / max(target_length, prediction_length))`이다. 기호별 정확도와 같지 않다.
TER은 전체 토큰 편집 거리 합을 정답 토큰 수 합으로 나눈 값이다. 1보다 클 수도 있다.

## 구조별 시험 결과

| 구조 | 개수 | 완전 일치율 | 편집 유사도 |
|---|---:|---:|---:|
{chr(10).join(group_lines)}

분수·첨자·근호 집합은 서로 겹친다. 표본이 적은 범주는 큰 성능 차이를 단정하기 어렵다.

## 입력 이미지 대조 실험

`control_normal`, `control_blank`, `control_shuffled`는 동일한 시험 수식 앞 256개를 평가했다.
blank는 이미지 값만 0으로 바꾸며 원래 너비는 유지한다. shuffled는 batch 안에서 이미지와 너비를 함께 한 칸씩 이동시켜 다른 수식의 이미지를 제공한다.
이는 학습된 OCR의 입력 의존성 실험이며, 별도로 학습한 언어 모델과의 비교는 아니다.

## 해석 시 제한

- 이번 실행은 하나의 random seed에 대한 파일럿이다.
- 짧은 인쇄 수식에서의 학습 가능성을 평가한다. 손글씨·사진·다른 폰트·긴 수식의 실용 성능은 별도 검증이 필요하다.
- fixed context vector의 정보 압축과 작은 학습량 때문에 긴 수식/희귀 기호에서 오류가 날 수 있다.
- 학습 loss 감소만으로 성공을 판단하지 말고, unseen test 완전 일치율과 이미지 대조 실험을 함께 본다.
- test의 어휘 밖 토큰 포함 샘플 {summary['splits']['test'].get('oov_samples',0)}개, long_test {summary['splits']['long_test'].get('oov_samples',0)}개도 제외하지 않고 평가했다.

![학습 곡선](learning_curves.png)

![예측 사례](prediction_examples.png)

전체 500개 예측은 `report.html`에서 정답/오답 필터로 확인할 수 있다. 원시 지표는 `metrics.json`, 예측은 `predictions_test.json`에 저장했다.
"""
    (run / "REPORT.md").write_text(report, encoding="utf-8")
    image_data = base64.b64encode((run / "learning_curves.png").read_bytes()).decode()
    cards = []
    for i, r in enumerate(predictions):
        encoded = base64.b64encode((data / r["image"]).read_bytes()).decode()
        status = "correct" if r["exact"] else "error"
        cards.append(f'''<article class="sample {status}"><div class="samplehead"><b>#{i+1} · {"정답" if r["exact"] else "오답"}</b><span>편집 유사도 {pct(r["similarity"])}</span></div><img alt="시험 수식 {i+1}" src="data:image/png;base64,{encoded}"><p><b>정답</b> <code>{html.escape(r["target"])}</code></p><p><b>예측</b> <code>{html.escape(r["prediction"])}</code></p></article>''')
    doc = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Attention 없는 수식 OCR · 실험 결과</title><style>
body{{font-family:"Malgun Gothic",sans-serif;background:#f3f6fb;color:#23334c;margin:0}}main{{max-width:1200px;margin:auto;padding:40px 24px}}h1{{font-size:30px}}.muted{{color:#617087;line-height:1.8}}.stats{{display:flex;gap:16px;flex-wrap:wrap;margin:28px 0}}.stat{{background:white;border-radius:12px;padding:22px;min-width:180px}}.stat strong{{display:block;font-size:32px;margin-top:8px}}.curve{{width:100%;background:white;border-radius:12px}}.toolbar{{position:sticky;top:0;padding:16px 0;background:#f3f6fb;display:flex;gap:10px;align-items:center}}button{{border:1px solid #c2cdde;background:white;color:#24354b;border-radius:8px;padding:10px 18px;cursor:pointer}}button.active{{background:#2664d9;color:white}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:16px}}.sample{{background:white;padding:20px;border-radius:12px;border-top:4px solid #c65e5e;overflow:hidden}}.sample.correct{{border-top-color:#278556}}.sample img{{display:block;max-width:100%;height:auto;margin:20px 0}}.samplehead{{display:flex;justify-content:space-between;font-size:13px}}code{{font-family:Consolas,monospace;overflow-wrap:anywhere;font-size:13px}}p{{line-height:1.6}}@media(max-width:520px){{.grid{{grid-template-columns:1fr}}main{{padding:24px 12px}}}}</style><main>
<p class="muted">CNN + BiLSTM encoder + LSTM decoder · {dataset_name} pilot</p><h1>Attention 없이 수식을 읽을 수 있을까?</h1><p class="muted">{dataset_name} 수식 {summary['splits']['train']['samples']:,}개 학습. 정답을 보지 않고 끝까지 생성한 시험 수식 {test['n']}개 결과.<br>{scope_text}</p>
<div class="stats"><div class="stat">시험 수식 완전 일치<strong>{pct(test['exact_match'])}</strong>{test['exact_count']} / {test['n']}</div><div class="stat">시험 편집 유사도<strong>{pct(test['normalized_similarity'])}</strong>기호별 정확도와 다른 지표</div><div class="stat">확장 시험 완전 일치<strong>{pct(metrics['long_test']['exact_match'])}</strong>별도 {metrics['long_test']['n']}개</div></div>
<img class="curve" alt="학습 및 입력 이미지 대조 실험" src="data:image/png;base64,{image_data}"><h2>전체 시험 예측</h2><p class="muted">원래 시험 데이터 순서. 위의 수식 이미지를 읽고 정답 LaTeX와 예측을 비교할 수 있습니다.</p><div class="toolbar"><button class="active" data-filter="all">전체</button><button data-filter="correct">정답만</button><button data-filter="error">오답만</button><span id="count"></span></div><div class="grid">{''.join(cards)}</div>
</main><script>const buttons=document.querySelectorAll('button[data-filter]');function filter(k){{let n=0;document.querySelectorAll('.sample').forEach(x=>{{const show=k==='all'||x.classList.contains(k);x.hidden=!show;if(show)n++;}});buttons.forEach(b=>b.classList.toggle('active',b.dataset.filter===k));document.getElementById('count').textContent=n+'개';}}buttons.forEach(b=>b.addEventListener('click',()=>filter(b.dataset.filter)));filter('all');</script></html>'''
    (run / "report.html").write_text(doc, encoding="utf-8")
    print(f"Report saved: {run / 'REPORT.md'}")


if __name__ == "__main__":
    main()
