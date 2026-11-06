"""Compare actual predictions, structural accuracy, denominator accuracy and latency."""
from dataclasses import asdict
import base64
import html
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import train
from model import FormulaOCR,ModelConfig
from slash_semantics import fractions,parse

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"comparison"
VARIANTS={
 "A":{"name":"A: stacked → LaTeX","view":"A_stack_tex","checkpoint":"runs/controlled/best.pt","input":"stack","output":"tex"},
 "B":{"name":"B: stacked → slash","view":"B_stack_slash","checkpoint":"runs/slash_B/best.pt","input":"stack","output":"slash"},
 "C":{"name":"C: slash → LaTeX","view":"C_slash_tex","checkpoint":"runs/slash_C/best.pt","input":"slash","output":"tex"},
 "D":{"name":"D: slash → slash","view":"D_slash_slash","checkpoint":"runs/slash_D/best.pt","input":"slash","output":"slash"},
 "P":{"name":"P: practical slash LSTM","view":"P_practical","checkpoint":"runs/slash_practical/best.pt","input":"slash","output":"slash"},
}


def load_model(path,device):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    kind=checkpoint.get("model_type","seq2seq")
    assert kind=="seq2seq", "Only CNN/LSTM encoder-decoder models are allowed"
    model=FormulaOCR(ModelConfig(**checkpoint["config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model,checkpoint,kind


def structural_metrics(predictions):
    rows=[]
    for original in predictions:
        r=dict(original)
        ref=parse(r["target_tokens"])
        try: prediction=parse(r["predicted_tokens"]) if r["ended"] else None
        except ValueError: prediction=None
        rf=fractions(ref);pf=fractions(prediction)
        r.update(semantic_exact=bool(prediction==ref),syntax_valid=prediction is not None,
            numerator_exact=bool(rf and len(pf)==len(rf) and all(a[1]==b[1] for a,b in zip(rf,pf))),
            denominator_exact=bool(rf and len(pf)==len(rf) and all(a[2]==b[2] for a,b in zip(rf,pf))))
        rows.append(r)
    def summarize(rs):
        if not rs:return {"n":0}
        fs=[r for r in rs if r["tags"]["fraction"]]
        return {"n":len(rs),"semantic_exact":sum(r["semantic_exact"] for r in rs)/len(rs),
            "semantic_count":sum(r["semantic_exact"] for r in rs),
            "string_exact":sum(r["exact"] for r in rs)/len(rs),
            "syntax_valid":sum(r["syntax_valid"] for r in rs)/len(rs),
            "normalized_similarity":float(np.mean([r["similarity"] for r in rs])),
            "fraction_n":len(fs),
            "numerator_exact":sum(r["numerator_exact"] for r in fs)/len(fs) if fs else None,
            "denominator_exact":sum(r["denominator_exact"] for r in fs)/len(fs) if fs else None}
    result=summarize(rows)
    result["groups"]={tag:summarize([r for r in rows if r["tags"][tag]]) for tag in ["linear","fraction","scripts","radical"]}
    return result,rows


def benchmark(model,records,vocab,device):
    dataset=train.FormulaDataset(records[:60],vocab)
    batches=[train.collate([dataset[i]]) for i in range(len(dataset))]
    for images,widths,_,_ in batches[:8]:model.generate(images.to(device),widths,48).cpu()
    timings=[]
    for images,widths,_,_ in batches:
        if device.type=="cuda":torch.cuda.synchronize()
        start=time.perf_counter()
        model.generate(images.to(device),widths,48).cpu().tolist()
        if device.type=="cuda":torch.cuda.synchronize()
        timings.append((time.perf_counter()-start)*1000)
    return {"n":len(timings),"batch_size":1,"median_ms":float(np.median(timings)),
        "p95_ms":float(np.percentile(timings,95)),"includes":"prepared CPU tensor -> GPU transfer -> output token IDs; excludes image preprocessing"}


def main():
    torch.set_num_threads(4)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":torch.backends.cudnn.benchmark=True
    summary={"selection":"Mean semantic exact accuracy over original/font/capture validation; original validation then latency break ties.",
        "scope":"A-D: five-family synthetic grammar, STIX. P: three fonts and additional nested expressions. Disjoint train/test expressions.",
        "device":str(device),"variants":{}}
    saved={}
    for key,variant in VARIANTS.items():
        print(f"EVALUATE {key}",flush=True)
        train.DATA=OUT/"views"/variant["view"]
        model,checkpoint,kind=load_model(ROOT/variant["checkpoint"],device)
        vocab=checkpoint["vocab"]
        assert vocab==json.loads((train.DATA/"vocab.json").read_text(encoding="utf-8"))
        item={**variant,"kind":kind,"epoch":checkpoint["epoch"],"parameters":sum(p.numel() for p in model.parameters())}
        saved[key]={}
        for split in ["val","val_font","val_capture","test","long_test","test_font","test_capture"]:
            records=train.read_records(split)
            dataset=train.FormulaDataset(records,vocab)
            loader=DataLoader(dataset,batch_size=32,collate_fn=train.collate)
            _,raw=train.evaluate(model,loader,device,max_tokens=48)
            metrics,rows=structural_metrics(raw)
            for r,record in zip(rows,records):r["pair_id"]=record["pair_id"]
            item[split]=metrics;saved[key][split]=rows
            (OUT/f"predictions_{key}_{split}.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
            print(f"{key} {split}: structural={metrics['semantic_exact']:.4f}",flush=True)
        item["latency"]=benchmark(model,train.read_records("val"),vocab,device)
        item["robust_validation_exact"]=float(np.mean([item[s]["semantic_exact"] for s in ["val","val_font","val_capture"]]))
        item["training_seconds"]=sum(r["seconds"] for r in checkpoint["history"])
        # For candidates whose best epoch precedes their final epoch, account for the full run.
        history_path=(ROOT/variant["checkpoint"]).parent/"history.json"
        if history_path.exists():item["training_seconds"]=sum(r["seconds"] for r in json.loads(history_path.read_text()))
        summary["variants"][key]=item
        del model
        if device.type=="cuda":torch.cuda.empty_cache()
    winner=max(summary["variants"],key=lambda k:(summary["variants"][k]["robust_validation_exact"],summary["variants"][k]["val"]["semantic_exact"],-summary["variants"][k]["latency"]["median_ms"]))
    summary["winner"]=winner
    v=summary["variants"][winner]
    train.DATA=OUT/"views"/v["view"]
    model,checkpoint,kind=load_model(ROOT/v["checkpoint"],device)
    records=train.read_records("test")
    dataset=train.FormulaDataset(records,checkpoint["vocab"])
    _,raw=train.evaluate(model,DataLoader(dataset,batch_size=1,collate_fn=train.collate),device,max_tokens=48)
    single,rows=structural_metrics(raw)
    summary["winner_single_image_test"]=single
    (OUT/"best_predictions_single.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    package={"model":{k:t.detach().cpu() for k,t in model.state_dict().items()},"model_type":kind,
        "config":checkpoint["config"],"vocab":checkpoint["vocab"],"input_notation":v["input"],
        "output_notation":v["output"],"selected_variant":winner,"source_checkpoint":v["checkpoint"],
        "scope":summary["scope"],"validation_semantic_exact":v["val"]["semantic_exact"],
        "robust_validation_exact":v["robust_validation_exact"]}
    torch.save(package,OUT/"best_model.pt")
    (OUT/"metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    make_report(summary,saved)
    print(f"WINNER {winner}: {single['semantic_exact']:.4f} single-image exact",flush=True)


def make_report(summary,saved):
    variants=summary["variants"];winner=summary["winner"]
    def pct(x):return f"{100*x:.1f}%"
    table=[];robust=[]
    for key,v in variants.items():
        table.append(f"| {key} | {v['input']} → {v['output']} | {v['kind']} | {pct(v['test']['semantic_exact'])} | {pct(v['test']['groups']['fraction']['semantic_exact'])} | {pct(v['test']['denominator_exact'])} | {v['latency']['median_ms']:.1f} |")
        robust.append(f"| {key} | {pct(v['test_font']['semantic_exact'])} | {pct(v['test_capture']['semantic_exact'])} | {pct(v['long_test']['semantic_exact'])} |")
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout="constrained")
    colors=["#2476d1" if k==winner else "#9aaec5" for k in variants]
    for ax,field,title in [(axes[0],"test","Normal test"),(axes[1],"test_font","DejaVu Sans font"),(axes[2],"test_capture","Mild capture distortion")]:
        values=[100*v[field]["semantic_exact"] for v in variants.values()]
        bars=ax.bar(list(variants),values,color=colors)
        ax.bar_label(bars,fmt="%.1f%%",fontsize=9)
        ax.set(title=title,ylabel="Whole-expression structural accuracy (%)",ylim=(0,110))
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUT/"comparison.png",dpi=160);plt.close(fig)
    best=variants[winner]
    report=f"""# 슬래시 표기: 실제 학습 비교 결과

CNN·LSTM encoder–decoder, attention·CTC 없음. **원본·폰트·왜곡 validation 평균 구조 정확도로 선택한 구성: {winner} ({best['name']}).**

## 같은 시험 수식 500개

| 조건 | 입력 → 출력 | 모델 | 수식 전체 정확도 | 분수 100개 정확도 | 분모 정확도 | 1장 추론 중앙값 ms |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(table)}

정확도는 출력 표기를 공통 수식 트리로 변환한 뒤 비교했다. 괄호·분자·분모·기호가 모두 일치해야 한다. 교환법칙이나 약분 같은 대수적 동등성은 사용하지 않는다.

## 활용 시 조건 변화

| 조건 | DejaVu Sans 폰트 | 약한 흐림·기울임·노이즈 | 별도 중첩 분수 |
|---|---:|---:|---:|
{chr(10).join(robust)}

폰트: A–D는 STIX로 학습, DejaVu Sans로 평가. **P는 STIX·DejaVu Sans·Computer Modern을 학습하므로 해당 폰트는 P에게 미지의 폰트가 아니다.** capture: 같은 시험 수식에 ±1도 회전, 0.35px blur, 밝기 노이즈 sigma 4/255. 중첩 분수는 200개이며 A–D는 단층 분수만 학습했다. P는 별도 중첩 수식 1,000개도 학습했고 이 시험 수식과는 겹치지 않는다.

## 분석과 권장 구성

- **입력 이미지를 슬래시로 쓰는 것이 큰 차이를 보였다.** 출력만 바꾼 A→B는 {100*(variants['B']['test']['semantic_exact']-variants['A']['test']['semantic_exact']):.1f}%p 변화였고, 입력을 바꾼 A→C는 {100*(variants['C']['test']['semantic_exact']-variants['A']['test']['semantic_exact']):.1f}%p 개선됐다. 입력과 출력을 모두 슬래시로 맞춘 D의 분모 인식은 {pct(variants['D']['test']['denominator_exact'])}였다.
- **활용용 기본 모델은 P로 선택했다.** 같은 폰트의 기본 시험에서는 D가 P보다 2개 더 맞혔지만, DejaVu Sans 시험은 D {pct(variants['D']['test_font']['semantic_exact'])}, P {pct(variants['P']['test_font']['semantic_exact'])}였다. 여러 폰트가 들어올 수 있는 사용에서는 폰트 보강이 큰 도움이 됐다. 두 구성 모두 같은 CNN·LSTM 구조를 쓴다.
- **현재 사용 범위는 짧고 단순한 인쇄 수식이다.** P도 중첩 분수는 {pct(variants['P']['long_test']['semantic_exact'])}에 그쳤다. 임의의 긴 수식이나 손글씨에 바로 적용할 만큼 검증된 상태는 아니며, 손글씨 학습은 계획서의 후속 개발 범위다.

## 사용 가능한 결과물

- 선택 모델: `best_model.pt` (optimizer를 제외한 inference package).
- 한 장씩 평가한 시험 정확도: **{pct(summary['winner_single_image_test']['semantic_exact'])}**. 표의 기본 평가는 batch 32, 실제 한 장 입력도 별도로 확인했다.
- GPU 한 장 추론: 전처리된 CPU tensor에서 전송·모델·토큰 출력까지. 이미지 crop/resize와 파일 읽기 시간은 제외했다. warm-up 뒤 validation 이미지 60장으로 측정했다.
- `predict_best.py image.png`로 실행할 수 있다. 선택 모델이 기대하는 입력은 `{best['input']}`, 기본 출력은 `{best['output']}`다.

## 데이터와 학습

기존 통제된 데이터의 수식 목록과 분할을 그대로 재사용했다: train 3,000, validation 300, test 500, 중첩 200. 모델 간 같은 식을 읽는다. 같은 수식의 다른 렌더링은 train과 test에 분산하지 않았다. 출력 변환 4,000개의 수식 트리 동등성을 검사했다.

A는 동일 목록으로 이미 학습한 40 epoch checkpoint를 재사용했다. B/C/D는 같은 CNN–BiLSTM–LSTM, AdamW LR .001, batch 32, 40 epochs. 기호 어휘는 각 조건의 학습 정답에서 구성했다. CTC는 사용자 요청으로 제외했고 CTC 학습 모델은 사용하지 않았다.

P는 D에서 시작해 수식 4,000개(기존 3,000+새 중첩 1,000), 폰트 3종, 회전·흐림·노이즈 augmentation으로 24 epochs 추가 학습했다. 구조는 같은 CNN–BiLSTM–LSTM, cross-entropy이며 LR .0003에서 cosine 감소를 사용했다. P는 원본·폰트·capture validation 평균으로 checkpoint를 선택했다. 따라서 P의 변화는 출력 표기만의 효과가 아니다.

A–D checkpoint는 원본 validation 정확도로 선택했다. 최종 구성은 모든 모델의 원본·폰트·capture validation 구조 정확도의 평균으로 선택했다. 동률이면 원본 validation 정확도, 이후 추론 중앙값을 사용했다. test 결과로 checkpoint나 최종 구성을 선택하지 않았다. 기본 seed 42, 실사용 추가 학습 seed 44의 실행 결과다.

이 점수는 **제한된 5종 생성 문법의 인쇄 수식**에 대한 결과다. 임의의 논문 수식, 손글씨, 카메라 사진 전체 페이지의 정확도를 뜻하지 않는다. 기존 사진 속 상하 분수를 슬래시 입력으로 자동 변환하는 기능은 포함하지 않는다.

![실험 비교](comparison.png)

분수별 원본 이미지와 모든 모델의 예측은 [prediction_viewer.html](prediction_viewer.html)에서 볼 수 있다. 원시 지표는 `metrics.json`, 모든 예측은 `predictions_*.json`에 있다.
"""
    (OUT/"REPORT.md").write_text(report,encoding="utf-8")
    cards=[]
    references=[json.loads(x) for x in (OUT/"views/A_stack_tex/test.jsonl").read_text().splitlines()]
    byid={k:{r["pair_id"]:r for r in rows["test"]} for k,rows in saved.items()}
    slash_records={r["pair_id"]:r for r in [json.loads(x) for x in (OUT/"views/D_slash_slash/test.jsonl").read_text().splitlines()]}
    for ref in references:
        if not ref["tags"]["fraction"]:continue
        pair=ref["pair_id"];sr=slash_records[pair]
        def embed(path):return base64.b64encode(path.read_bytes()).decode()
        stack=embed((OUT/"views/A_stack_tex"/ref["image"]).resolve())
        slash=embed((OUT/"views/D_slash_slash"/sr["image"]).resolve())
        preds=[]
        for key in variants:
            r=byid[key][pair]
            preds.append(f"<p><b>{key} {'✓' if r['semantic_exact'] else '✗'}</b> <code>{html.escape(r['prediction'])}</code></p>")
        cards.append(f'<article><div class="images"><img src="data:image/png;base64,{stack}" alt="상하 분수"><img src="data:image/png;base64,{slash}" alt="슬래시"></div><p>정답: <code>{html.escape(ref["latex"])}</code></p>{"".join(preds)}</article>')
    doc=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>슬래시 OCR 비교</title><style>body{{font-family:"Malgun Gothic",sans-serif;background:#f4f7fb;color:#20334c}}main{{max-width:1100px;margin:auto;padding:24px}}article{{background:white;padding:24px;margin:20px 0;border-radius:12px}}.images{{display:flex;gap:30px;align-items:center;flex-wrap:wrap}}img{{max-width:100%}}code{{font-family:Consolas,monospace;overflow-wrap:anywhere}}p{{line-height:1.6}}</style><main><h1>같은 분수 100개, 표기와 모델 비교</h1><p>왼쪽 상하 분수 · 오른쪽 슬래시. A/B는 왼쪽, C/D/P는 오른쪽을 입력받았습니다.</p><p>A: LaTeX 출력 · B: slash 출력 · C: LaTeX 출력 · D: slash 출력 · P: 다중 폰트·중첩 추가 학습.</p>{''.join(cards)}</main></html>'''
    (OUT/"prediction_viewer.html").write_text(doc,encoding="utf-8")


if __name__=="__main__":main()
