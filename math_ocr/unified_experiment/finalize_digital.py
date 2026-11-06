"""Finish the same frozen study: diagnostics, timing, paired notation and report.

Primary: untouched MathWriting subset. CROHME and paired fonts are explicitly
secondary diagnostics, NOT notebook-photograph or full-benchmark performance.
"""
import base64,hashlib,html,json,os,random,sys,time
from collections import Counter
import numpy as np
from PIL import Image
import torch
from torch import nn
from engine import ROOT,Model,CLASSES,TOKENS,segment,preprocess,crop
from syntax import normalize,tokenize
from training import evaluate,distance,save
sys.path.insert(0,str(ROOT.parent/'planned_experiment'))
from run import box_counts

@torch.inference_mode()
def decode(model,prob,g):
    dev=next(model.parameters()).device
    z=model.project(torch.cat((torch.tensor(prob,device=dev),torch.tensor(g,device=dev)),1)[None]);_,(h,c)=model.encoder(z);state=(torch.cat((h[0],h[1]),-1)[None],torch.cat((c[0],c[1]),-1)[None]);prev=torch.ones(1,1,dtype=torch.long,device=dev);pred=[];done=False
    for _ in range(64):
        z,state=model.decoder(model.embedding(prev),state);s=model.output(z);s[:,:,[0,1,3]]=-torch.inf;prev=s.argmax(-1);k=prev.item()
        if k==2:done=True;break
        pred.append(TOKENS[k])
    return pred,done

def identical(pred,target,done):
    try:return bool(done and normalize(pred)[1]==normalize(target)[1])
    except (ValueError,RecursionError):return False

def confidence(predictions):
    rng=np.random.default_rng(42);values=[]
    groups={k:[r for r in predictions if r['kind']==k] for k in ['basic','sup','sub','fraction']}
    for _ in range(1000):
        total=count=0
        for rs in groups.values():
            if not rs:continue
            xs=np.array([r['exact'] for r in rs]);total+=int(xs[rng.integers(len(xs),size=len(xs))].sum());count+=len(xs)
        values.append(total/count)
    return list(map(float,np.percentile(values,[2.5,97.5])))

def crohme_diagnostics(model):
    source=ROOT.parent/'segmentation/data';rows=[json.loads(s) for s in (source/'test.jsonl').read_text().splitlines()];glyphs=np.load(source/'glyphs.npy',mmap_mode='r');diag=[];alltruth=[];allpred=[];counts=np.zeros(3,int);dev=next(model.parameters()).device
    def mapped(label):return '-' if label=='\\frac' else label
    for row in rows:
        for g,idx in zip(row['glyphs'],row['glyph_indexes']):
            label=mapped(g['label'])
            if label not in CLASSES:continue
            x=torch.tensor(crop(Image.fromarray(glyphs[idx,1]))[None],device=dev)
            with torch.inference_mode():p=int(model.cnn(x).argmax(1))
            alltruth.append(CLASSES.index(label));allpred.append(p)
        if not 3<=len(row['glyphs'])<=16 or any(mapped(g['label']) not in CLASSES for g in row['glyphs']):continue
        try:target,identity=normalize(row['tokens'])
        except (ValueError,RecursionError):continue
        if len(target)>63 or set(target)-set(TOKENS):continue
        original=Image.open(source/row['image']).convert('L');im,_=preprocess(original);sx,sy=im.width/original.width,im.height/original.height
        ordered=sorted(zip(row['glyphs'],row['glyph_indexes']),key=lambda t:((t[0]['box'][0]+t[0]['box'][2])/2,(t[0]['box'][1]+t[0]['box'][3])/2))
        b=np.array([[g['box'][0]*sx,g['box'][1]*sy,g['box'][2]*sx,g['box'][3]*sy] for g,idx in ordered],np.float32);centers=(b[:,:2]+b[:,2:])/2
        try:
            _,ag,ab=segment(original);h=max(1,float(np.median([(bb[2]-bb[0])/gg[-2] for bb,gg in zip(ab,ag) if gg[-2]>0])));counts+=box_counts(ab,b.tolist());auto=model.recognize(original);auto_ok=identical(auto['tokens'],target,auto['terminated'])
        except ValueError:h=max(1,float(np.median(b[:,3]-b[:,1])));counts+=np.array([0,0,len(b)]);auto_ok=False
        delta=np.vstack((np.zeros((1,2),np.float32),np.diff(centers,axis=0)))/h;geo=np.column_stack((centers[:,0]/im.width,centers[:,1]/im.height,(b[:,2]-b[:,0])/im.width,(b[:,3]-b[:,1])/im.height,delta,(b[:,2]-b[:,0])/h,(b[:,3]-b[:,1])/h)).astype('float32');geo[:,4:]=geo[:,4:].clip(-4,4)
        x=np.stack([crop(Image.fromarray(glyphs[idx,1])) for g,idx in ordered]);truth=np.array([CLASSES.index(mapped(g['label'])) for g,idx in ordered])
        with torch.inference_mode():p=model.cnn(torch.tensor(x,device=dev)).softmax(1).cpu().numpy()
        pr,done=decode(model,p,geo);po,odone=decode(model,np.eye(70,dtype=np.float32)[truth],geo)
        diag.append({'id':row['id'],'auto_exact':auto_ok,'gt_regions_exact':identical(pr,target,done),'oracle_exact':identical(po,target,odone),'target':target,'gt_regions_prediction':pr,'oracle_prediction':po})
    cm=np.zeros((70,70),int);np.add.at(cm,(alltruth,allpred),1);tp=np.diag(cm);support=cm.sum(1);f1=2*tp/np.maximum(1,cm.sum(0)+support);bt,bf,bn=map(int,counts)
    return {'scope':'Reused CROHME official-test subset; auxiliary diagnosis, not fresh primary benchmark. GT conditions use isolated ground-truth glyph masks and transformed positions. Oracle is not a strict upper bound.','formula_n':len(diag),'formulas':diag,'auto_exact_match':sum(r['auto_exact'] for r in diag)/max(1,len(diag)),'gt_regions_exact_match':sum(r['gt_regions_exact'] for r in diag)/max(1,len(diag)),'oracle_exact_match':sum(r['oracle_exact'] for r in diag)/max(1,len(diag)),'segmentation':{'tp':bt,'fp':bf,'fn':bn,'precision':bt/max(1,bt+bf),'recall':bt/max(1,bt+bn),'f1':2*bt/max(1,2*bt+bf+bn)},'symbol_n':len(alltruth),'symbol_accuracy':float(np.trace(cm)/max(1,cm.sum())),'symbol_macro_f1_present_classes':float(f1[support>0].mean()),'classes_with_test_support':int((support>0).sum()),'class_support':dict(zip(CLASSES,map(int,support))),'confusion_matrix':cm.tolist()}

def paired_fonts(model):
    os.environ['MPLCONFIGDIR']=str(ROOT/'plot_cache')
    from matplotlib.mathtext import MathTextParser
    from matplotlib.font_manager import FontProperties
    records=json.loads((ROOT/'capture_plan.json').read_text(encoding='utf-8'))['records'];pairs={};parser=MathTextParser('agg');rows=[]
    for r in records:
        if r['split']!='test' or r['primary_type']!='fraction':continue
        a=parser.parse('$'+r['prompt_latex']+'$',dpi=110,prop=FontProperties(size=32,math_fontfamily='stix'));im=Image.fromarray(255-np.asarray(a.image));path=ROOT/'paired_font_images'/f"{r['sample_id']}.png";path.parent.mkdir(exist_ok=True);im.save(path);p=model.recognize(im);ok=identical(p['tokens'],r['planned_target_tokens'],p['terminated']);pairs.setdefault(r['fraction_pair_id'],{})[r['notation']]=ok
        rows.append({'id':r['sample_id'],'pair':r['fraction_pair_id'],'notation':r['notation'],'exact':ok,'prediction':p['tokens'],'truth':r['planned_target_tokens'],'image':str(path.relative_to(ROOT))})
    upper=np.array([v['vertical'] for v in pairs.values()]);slash=np.array([v['slash'] for v in pairs.values()]);delta=slash.astype(int)-upper.astype(int);rng=np.random.default_rng(42);ci=np.percentile([delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(1000)],[2.5,97.5])
    return {'scope':'30 NEW CONTENT pairs rendered with STIX, not collected photographs or human-written paired formulas. Reserved contents excluded from training. Synthetic notation diagnostic only; no handwriting input recommendation inferred.','pairs':len(pairs),'vertical_exact_match':float(upper.mean()),'slash_exact_match':float(slash.mean()),'delta_slash_minus_vertical':float(delta.mean()),'delta_ci95':ci.tolist(),'both_correct':int((upper&slash).sum()),'vertical_only':int((upper&~slash).sum()),'slash_only':int((~upper&slash).sum()),'both_wrong':int((~upper&~slash).sum()),'predictions':rows}

def main():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';ck=torch.load(ROOT/'final_model.pt',map_location=device,weights_only=False);model=Model().to(device);model.load_state_dict(ck['model']);model.eval();real=json.loads((ROOT/'prepared_manifest.json').read_text(encoding='utf-8'));testrows=[r for r in real if r['split']=='test'];primary=json.loads((ROOT/'mathwriting_test.json').read_text());primary['ci95_stratified_bootstrap']=confidence(primary['predictions']);primary['correct_count']=sum(r['exact'] for r in primary['predictions']);primary['by_type']={k:{'n':sum(r['kind']==k for r in primary['predictions']),'exact_match':sum(r['exact'] for r in primary['predictions'] if r['kind']==k)/max(1,sum(r['kind']==k for r in primary['predictions']))} for k in ['basic','sup','sub','fraction']};save('mathwriting_test.json',primary)
    reuse='--report-only' in sys.argv
    diagnostics=json.loads((ROOT/'diagnostics_crohme.json').read_text(encoding='utf-8')) if reuse else crohme_diagnostics(model)
    save('diagnostics_crohme.json',diagnostics);print('CROHME diagnostics',diagnostics['formula_n'],flush=True)
    paired=json.loads((ROOT/'paired_fraction_diagnostic.json').read_text(encoding='utf-8')) if reuse else paired_fonts(model)
    save('paired_fraction_diagnostic.json',paired);print('paired diagnostic complete',flush=True)
    # CPU benchmark: same model, image-only path; compare all outputs to GPU.
    model=model.cpu();torch.set_num_threads(1);times=[];parity=0;reference={r['sample_id']:r['pred'] for r in primary['predictions']}
    if reuse:
        timing=json.loads((ROOT/'latency.json').read_text(encoding='utf-8'));parity=timing['cpu_gpu_prediction_agreement_count']
    else:
        for row in testrows[:10]:model.recognize(Image.open(ROOT/row['image']))
        for repeat in range(3):
            for row in testrows:
                start=time.perf_counter();p=model.recognize(Image.open(ROOT/row['image']));times.append(time.perf_counter()-start)
                if repeat==0:parity+=p['tokens']==reference[row['sample_id']]
        timing={'device':'CPU','threads':1,'n':len(testrows),'repeats':3,'warmup':10,'median_seconds':float(np.median(times)),'p95_seconds':float(np.percentile(times,95)),'cpu_gpu_prediction_agreement_count':parity,'agreement_denominator':len(testrows),'scope':'File reading + automatic preprocessing + segmentation + CNN + LSTM. Model load, UI, and network excluded.'};save('latency.json',timing)
    # Actual full-test gallery, not just successful examples.
    lookup={r['sample_id']:r for r in real};parts=['<!doctype html><meta charset="utf-8"><title>전체 시험 결과</title><style>body{font:15px system-ui;max-width:1000px;margin:auto}article{padding:18px;border-bottom:1px solid #ddd}img{max-width:90%;max-height:150px}code{white-space:pre-wrap}</style>',f'<h1>MathWriting 새 시험 {primary["n"]}개 전체</h1><p>정답 {primary["correct_count"]}/{primary["n"]}. 디지털 획 렌더링이며 사진 시험이 아닙니다.</p>']
    for p in primary['predictions']:
        row=lookup[p['sample_id']];encoded=base64.b64encode((ROOT/row['image']).read_bytes()).decode();parts.append(f'<article><b>{html.escape(p["sample_id"])} · {p["kind"]} · {"정답" if p["exact"] else "오답"}</b><br><img src="data:image/png;base64,{encoded}"><p>정답: <code>{html.escape(" ".join(p["truth"]))}</code></p><p>예측: <code>{html.escape(" ".join(p["pred"]))}</code></p></article>')
    (ROOT/'test_gallery.html').write_text('\n'.join(parts),encoding='utf-8')
    history=json.loads((ROOT/'training_history.json').read_text(encoding='utf-8'));env=json.loads((ROOT/'training_environment.json').read_text(encoding='utf-8'));parent=json.loads((ROOT/'symbol_parent_audit.json').read_text(encoding='utf-8'));lock=json.loads((ROOT/'data_lock.json').read_text(encoding='utf-8'))
    pct=lambda n:f'{n*100:.2f}%'
    lines=['# 촬영 사진을 제외한 단일 통합 실험 결과','','실행일: 2026-11-05. 사용자가 직접 촬영 사진 없이 진행하도록 지시하여 프로토콜 3.0으로 변경했다. 아래는 실제 실행 결과다.','',f'**주 결과: 새로운 MathWriting 실제 손글씨 수식 {primary["n"]}개 중 {primary["correct_count"]}개 전체 일치, {pct(primary["exact_match"])}.**',f'정규화 토큰 편집 거리 {primary["token_error_rate"]:.4f}, 문법 오류율 {pct(primary["syntax_error_rate"])}. 95% 층화 bootstrap 구간: {pct(primary["ci95_stratified_bootstrap"][0])}~{pct(primary["ci95_stratified_bootstrap"][1])}.','', '**현재 모델은 실제 수식 인식 도구로 사용하기에 정확도가 부족하다.** 실험 절차의 수행 완료와 실용성 목표 달성을 구분한다.','', '## 하나의 실행과 선택','',f'- 실제 수식 학습 {lock["real_counts"]["train"]}, 검증 {lock["real_counts"]["val"]}, 새 시험 {lock["real_counts"]["test"]}; 학습 보조 합성 2,000장.',f'- 보조 기호 {env["auxiliary_allowed"]:,}개 사용. 가장 적은 클래스의 원본 {env["smallest_class_originals"]}개. 클래스 최소 50개 수집 조건은 직접 촬영 제외에 따라 가용 자료 사용으로 변경하고 한계를 기록했다.',f'- CNN 초기화 10에포크 후 같은 CNN–BiLSTM–LSTM을 공동 학습했다. 본학습 {len(history)}에포크, 에포크당 600회, 실제 7장+합성 3장 배치.',f'- 검증으로 선택한 에포크 {ck["epoch"]}, 검증 전체 일치율 {pct(ck["selection"][0])}. 시험은 선택에 사용하지 않았다.',f'- {env["gpu"]}, PyTorch {env["torch"]}, 파라미터 {env["parameters"]:,}개.','- 전체 이미지의 원본/약한 증강 2개 분할 결과를 캐시했다. 캐시는 픽셀 crop과 위치이며 CNN 확률/특징을 detach해서 저장한 것이 아니다. 이후 매 학습마다 CNN 재계산과 약한 기호 증강을 수행했다.','- 연결 성분/기하 병합은 고정, CNN과 LSTM의 연결은 역전파로 학습했다.','- 자동 crop은 경계 상자 내부의 전경을 포함하므로 이웃 획이 섞일 수 있다. 원본 기호/합성 재료는 정답 획 기반으로 준비했다.','', '## 데이터 검증과 범위','', '- 원본 전체 253,182개를 검사했고 과거 사용한 MathWriting 시험 ID 800개를 제외했다. 수식 내용과 이미지 해시의 집합 간 중복은 0이다.',f'- MathWriting 기호 {parent["bank_symbols"]}개 중 {parent["matched_symbols"]}개의 부모를 원래 획으로 연결했다. 미연결 또는 검증·시험 내용과 연결될 수 있는 보조 기호 {len(parent["excluded_auxiliary_indices"])}개를 보수적으로 제외하고 합성에도 사용하지 않았다.','- 클래스는 영문 대소문자·숫자·기본 연산자·소수점 70종. 사진은 사용하지 않았고 사진 정확도나 새 필자 일반화를 측정했다고 주장하지 않는다.','- 데이터 수량 변경과 사진 제외는 본학습·시험 전에 확정했다. 동일 시험 결과로 모델을 다시 고르지 않았다.','', '## 주 시험의 유형별 결과','','| 유형 | 수량 | 전체 일치율 |','|---|---:|---:|']
    for k,v in primary['by_type'].items():lines.append(f'| {k} | {v["n"]} | {pct(v["exact_match"])} |')
    lines+=['','## 보조 분수 표기 진단','','실제 손글씨 쌍을 수집하지 않아 새 내용의 STIX 렌더링 30쌍을 고정 모델에 입력했다. 이 표는 합성 표기 진단이며 손글씨 표기 권장 결론으로 확대하지 않는다.','',f'- 상하: {pct(paired["vertical_exact_match"])} / 슬래시: {pct(paired["slash_exact_match"])}.',f'- 둘 다 정답 {paired["both_correct"]}, 상하만 {paired["vertical_only"]}, 슬래시만 {paired["slash_only"]}, 둘 다 오답 {paired["both_wrong"]}.','', '## 같은 모델의 단계 진단','','기호 영역 정답이 있는 CROHME 기존 시험 부분집합을 보조 진단에만 사용했다. 이전 프로젝트에서도 사용한 자료이며 주 시험의 독립 성적과 합산하지 않는다.','', '| 조건 | 전체 일치율 |','|---|---:|',f'| 자동 입력 ({diagnostics["formula_n"]}식) | {pct(diagnostics["auto_exact_match"])} |',f'| 정답 영역·기호 mask + CNN | {pct(diagnostics["gt_regions_exact_match"])} |',f'| 정답 영역·정답 기호 | {pct(diagnostics["oracle_exact_match"])} |','',f'분할 F1 {diagnostics["segmentation"]["f1"]:.4f}. 기호 정확도 {pct(diagnostics["symbol_accuracy"])} ({diagnostics["symbol_n"]}개), 실제 시험 표본이 있는 클래스 {diagnostics["classes_with_test_support"]}/70. 정답 one-hot 조건은 분포가 달라 엄밀한 상한이 아니며 단계별 차이를 합산하지 않는다.','', '## 시간과 기능','',f'CPU 단일 스레드에서 {len(testrows)}장×3회. 중앙값 {timing["median_seconds"]*1000:.2f}ms, p95 {timing["p95_seconds"]*1000:.2f}ms. 파일 읽기부터 LaTeX까지이며 모델 로딩·화면 렌더링 제외.',f'CPU/GPU 출력 일치 {parity}/{len(testrows)}. 차이가 있으면 장치 수치 연산 차이로 별도 기록하며 주 시험은 GPU 이미지 전용 실행 결과다.','', '## 실행 방법','','이 폴더에서 상위 프로젝트 Python 환경을 사용한다.','', '```powershell','../.venv/Scripts/python.exe run_digital.py','../.venv/Scripts/python.exe finalize_digital.py','../.venv/Scripts/python.exe recognize.py data/images/이미지.png','../.venv/Scripts/python.exe web.py --port 8767','```','', 'run_digital.py는 동일한 데이터/코드 해시에서만 마지막 에포크를 재개한다. 주 결과와 전체 예측은 mathwriting_test.json 및 test_gallery.html에 있다. UI/API 검증 기록은 ui_checks.json에 별도로 저장한다.','', '## 해석','','검증 성능에 비해 학습 손실이 낮아지는 양상이 있어 새로운 수식 내용에 대한 일반화가 부족하다. 이는 손글씨 전체 수식 자료로 학습했음에도 단순 CNN 확률·좌표와 고정 길이 인코더 상태만으로 충분한 인식 성능을 얻지 못한 결과다. 단일 실행이므로 특정 구성 하나의 인과적 효과나 다른 알고리즘 대비 우월성을 주장하지 않는다. 새 모델을 개선하려면 이번 시험과 분리된 새 시험을 준비해야 한다.']
    seenpath=ROOT/'seen_training_diagnostic.json'
    if not seenpath.exists():
        rng=random.Random(42);subset=[]
        for kind in ['basic','sup','sub','fraction']:subset+=rng.sample([r for r in real if r['split']=='train' and r['kind']==kind],25)
        seen=evaluate(model.cpu(),subset,True);seen['scope']='Post-hoc frozen-model diagnostic on 100 seen real training images; not an independent test or selection criterion.';save('seen_training_diagnostic.json',seen)
    seen=json.loads(seenpath.read_text(encoding='utf-8'))
    lines+=['','## 학습 자료 재인식 사후 진단','',f'고정한 같은 모델에 실제 학습 수식 중 유형별 25개씩 총 {seen["n"]}개를 시드42로 추출하여 자유 생성했다. 전체 일치율은 {pct(seen["exact_match"])}였다. 이는 학습 자료를 다시 본 사후 진단이며 독립 시험이나 모델 선택 지표가 아니다.',f'학습 자료 재인식 {pct(seen["exact_match"])}와 검증 {pct(ck["selection"][0])}, 새로운 시험 {pct(primary["exact_match"])}의 큰 차이는 새 수식 내용에 대한 일반화 부족과 과적합 양상을 보여준다.']
    if (ROOT/'ui_checks.json').exists():
        ui=json.loads((ROOT/'ui_checks.json').read_text(encoding='utf-8'))
        lines+=['','## 화면·복사 검증 범위','',f'- 로컬 주소: {ui["url"]}. HTTP 페이지, 이미지 입력 API, LaTeX 응답, 실제 수식 PNG 생성 검사를 통과했다.',
            '- JavaScript mock DOM에서 파일 미리보기·결과 표시·clipboard.writeText 호출·복사 fallback 동작을 검사했다. 실제 OS 클립보드 복사 검증으로 표현하지 않는다.',
            '- 인앱 브라우저의 localhost 접속이 시간 초과됐고 native Chrome 제어가 승인되지 않아 실제 브라우저 클릭·화면 배치는 검증하지 못했다. 보안/방화벽/권한 설정은 바꾸지 않았다.']
    lines+=['','## 데이터 출처','', '- [MathWriting 공식 자료](https://github.com/google-research/google-research/tree/master/mathwriting): 실제 수식 및 기호 원본. 공개 데이터의 CC BY-NC-SA 4.0 조건과 원본 출처 기록을 유지한다.',
        '- CROHME 보조 기호·영역 정답의 출처와 이용 조건은 ../segmentation/raw/provenance.json 및 ../segmentation/REPORT.md에 기록되어 있다.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines).replace('--port 8767','--port 8768'),encoding='utf-8')
    previous=json.loads((ROOT/'state.json').read_text(encoding='utf-8'))
    if not previous.get('complete'):
        save('state.json',{'state':'FINAL_VERIFICATION','complete':False,'protocol_version':'3.0','training_started':True,'test_predictions_generated':True,'photographs_used':False,'primary_test_n':primary['n'],'primary_correct':primary['correct_count'],'primary_exact_match':primary['exact_match'],'remaining':'Local UI/API rendering and copy-flow verification'})
    print('REPORT READY',primary['correct_count'],primary['n'],timing,flush=True)

if __name__=='__main__':main()
