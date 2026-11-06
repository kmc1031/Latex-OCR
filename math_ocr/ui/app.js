import {defaults, boundedRect, canvasOf, transformed, extracted} from './editor.js';

const $ = id => document.getElementById(id);
const icons = {
  upload:'M12 16V3m-4 4 4-4 4 4M4 15v5h16v-5',
  crop:'M7 3v14h14M3 7h14v14',
  undo:'M9 4 4 9l5 5M4 9h10a6 6 0 0 1 0 12',
  'rotate-left':'M4 10a8 8 0 1 1 1 8M4 4v6h6',
  'rotate-right':'M20 10a8 8 0 1 0-1 8m1-14v6h-6',
  reset:'M4 10a8 8 0 1 1 1 8M4 4v6h6',
  image:'M3 4h18v16H3zM3 16l5-5 5 5 3-3 5 5M15 8h.01',
  'arrow-right':'M4 12h16m-5-5 5 5-5 5',
  shield:'M12 3 4 6v6c0 4 5 8 8 9 3-1 8-5 8-9V6zM8 12l3 3 5-6',
  scan:'M8 3H3v5m13-5h5v5M3 16v5h5m8 0h5v-5M7 12h10',
  info:'M12 8h.01M12 11v6M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
  spark:'m12 3 2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4z',
  copy:'M8 8h12v13H8zM16 8V3H3v14h5',
  download:'M12 3v12m-4-4 4 4 4-4M4 16v5h16v-5',
  file:'M13 3H5v18h14V9zM13 3v6h6M8 14h8m-8 3h5'
};
for(const node of document.querySelectorAll('[data-icon]')) {
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.setAttribute('viewBox','0 0 24 24');svg.setAttribute('aria-hidden','true');
  const path=document.createElementNS(svg.namespaceURI,'path');path.setAttribute('d',icons[node.dataset.icon]);svg.append(path);node.append(svg);
}

const state={original:null,base:null,rendered:null,params:defaults(),crop:null,name:'',zoom:1,ready:false,busy:false,importing:false,loading:0};
state.profile='';
let glyphs={},functions=[];
const history=[];
let checkpoint=null, drag=null, toastTimer;
const controlIds=['angle','shear-x','shear-y','scale-x','scale-y'];
const keys=['angle','shearX','shearY','scaleX','scaleY'];
const snapshot=()=>({base:state.base,params:{...state.params},crop:state.crop?{...state.crop}:null});

function toast(message,error=false) {
  $('toast').textContent=message;$('toast').classList.toggle('error',error);$('toast').hidden=false;
  clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,4200);
}
function controls() {
  const enabled=!!state.base&&!state.busy&&!state.importing;
  for(const id of [...controlIds,'crop-tool','reset','rotate-left','rotate-right','reset-angle','reset-adjust','zoom-out','zoom-in','zoom-fit','download-image']) $(id).disabled=!enabled;
  for(const id of ['apply-crop','clear-crop']) $(id).disabled=!enabled||!state.crop;
  $('undo').disabled=!enabled||!history.length;
  $('recognize').disabled=!enabled||!state.ready;
  $('model-profile').disabled=state.busy||state.importing||!state.ready;
  for(const button of document.querySelectorAll('[data-sample]')) button.disabled=state.busy||state.importing||!state.ready;
  for(const id of ['open-file','choose-file']) $(id).disabled=state.busy||state.importing;
}
function clearResult() {
  $('latex-output').value='';$('latex-output').disabled=true;
  $('copy-latex').disabled=true;$('download-tex').disabled=true;
  $('formula-preview').replaceChildren(Object.assign(document.createElement('span'),{className:'preview-placeholder',textContent:'인식한 수식이 여기에 표시됩니다.'}));
  $('result-status').textContent='대기 중';$('result-status').className='status-pill';
  $('result-meta').hidden=true;$('raw-output').textContent='아직 결과가 없습니다.';
  $('prepared-image').hidden=true;$('prepared-size').textContent='';
  if($('segments-details'))$('segments-details').hidden=true;
  $('edit-hint').textContent='인식 후 코드를 직접 수정할 수 있어요.';
}
function remember() {
  if(checkpoint) history.push(checkpoint);
  if(history.length>12)history.shift();
  checkpoint=snapshot();clearResult();controls();
}
function syncSliders() {
  controlIds.forEach((id,i)=>{
    $(id).value=state.params[keys[i]];
    $(id+'-value').value=state.params[keys[i]]+(id.startsWith('scale')?'%':'°');
  });
}
function fit() {
  if(!state.rendered)return;
  const stage=$('drop-zone');
  const scale=Math.min((stage.clientWidth-48)/state.rendered.width,(stage.clientHeight-48)/state.rendered.height,4)*state.zoom;
  $('canvas-frame').style.width=`${Math.max(1,state.rendered.width*scale)}px`;
  $('canvas-frame').style.height=`${Math.max(1,state.rendered.height*scale)}px`;
  $('zoom-fit').textContent=`${Math.round(scale*100)}%`;
  $('zoom-fit').title='화면에 맞추기';updateOverlay();
}
function render() {
  state.rendered=transformed(state.base,state.params);
  const canvas=$('editor-canvas');canvas.width=state.rendered.width;canvas.height=state.rendered.height;
  canvas.getContext('2d').drawImage(state.rendered,0,0);
  $('empty-state').hidden=true;$('canvas-scroll').hidden=false;
  $('image-info').textContent=`${state.name} · ${canvas.width} × ${canvas.height} px`;
  $('image-info').title=$('image-info').textContent;
  $('crop-overlay').setAttribute('viewBox',`0 0 ${canvas.width} ${canvas.height}`);
  syncSliders();fit();updateSelection();controls();
}
function updateOverlay() {
  const rect=state.crop;
  $('crop-dim').style.display=rect?'':'none';$('crop-guides').style.display=rect?'':'none';
  if(!rect)return;
  for(const id of ['mask-hole','crop-outline']) {
    for(const [key,value] of Object.entries({x:rect.x,y:rect.y,width:rect.w,height:rect.h})) $(id).setAttribute(key,value);
  }
  const {x,y,w,h}=rect;
  $('crop-grid').setAttribute('d',`M${x+w/3} ${y}v${h}M${x+2*w/3} ${y}v${h}M${x} ${y+h/3}h${w}M${x} ${y+2*h/3}h${w}`);
  const size=7*state.rendered.width/Math.max(1,$('canvas-frame').getBoundingClientRect().width);
  const positions={nw:[x,y],n:[x+w/2,y],ne:[x+w,y],e:[x+w,y+h/2],se:[x+w,y+h],s:[x+w/2,y+h],sw:[x,y+h],w:[x,y+h/2]};
  $('crop-handles').replaceChildren(...Object.entries(positions).map(([name,[cx,cy]])=>{
    const handle=document.createElementNS('http://www.w3.org/2000/svg','rect');
    Object.entries({x:cx-size/2,y:cy-size/2,width:size,height:size,rx:size*.12,'data-handle':name}).forEach(([k,v])=>handle.setAttribute(k,v));
    handle.style.cursor=`${name}-resize`;return handle;
  }));
}
function updateSelection() {
  updateOverlay();
  if(!state.rendered)return;
  const image=extracted(state.rendered,state.crop);
  $('selection-preview').src=image.toDataURL('image/png');$('selection-preview').hidden=false;
  $('selection-title').textContent=state.crop?'선택 영역만 인식':'전체 이미지 인식';
  $('selection-info').textContent=`${image.width} × ${image.height} px · ${state.crop?'선택 영역':'편집 이미지'}가 모델에 전달됩니다.`;
  controls();
}
function changeTransform() {
  const hadSelection=!!state.crop;state.crop=null;render();clearResult();
  if(hadSelection)toast('보정 후 수식 영역을 다시 선택해 주세요.');
}

async function loadBlob(blob,name,sequence=null) {
  if(state.busy)return;
  if(blob.size>20*1024*1024){toast('20MB 이하의 이미지로 선택해 주세요.',true);return;}
  sequence??=++state.loading;
  if(sequence!==state.loading)return;
  state.importing=true;controls();
  const url=URL.createObjectURL(blob);
  try{
    const image=new Image();image.src=url;await image.decode();
    if(sequence!==state.loading)return;
    const factor=Math.min(1,2400/Math.max(image.naturalWidth,image.naturalHeight));
    const canvas=canvasOf(Math.max(1,Math.round(image.naturalWidth*factor)),Math.max(1,Math.round(image.naturalHeight*factor)));
    canvas.getContext('2d').drawImage(image,0,0,canvas.width,canvas.height);
    state.original=canvas;state.base=canvas;state.params=defaults();state.crop=null;state.name=name;state.zoom=1;
    history.length=0;checkpoint=snapshot();clearResult();render();
    if(factor<1)toast('원활한 편집을 위해 긴 변을 2400px로 줄였습니다.');
  }catch{if(sequence===state.loading)toast('이미지를 읽을 수 없습니다. PNG, JPG 또는 WebP 파일을 선택해 주세요.',true);}
  finally{URL.revokeObjectURL(url);if(sequence===state.loading){state.importing=false;controls();}}
}
const chooseFile=()=>{if(!state.busy)$('file-input').click();};
$('open-file').onclick=chooseFile;$('choose-file').onclick=chooseFile;
$('file-input').onchange=e=>{const file=e.target.files[0];if(file)loadBlob(file,file.name);e.target.value='';};
for(const button of document.querySelectorAll('[data-sample]')) button.onclick=async()=>{
  if(state.busy||state.importing||!state.ready)return;
  const sequence=++state.loading,name=button.dataset.sample;
  state.importing=true;controls();
  try{const response=await fetch('/api/sample/'+name+'?profile='+encodeURIComponent(state.profile));if(!response.ok)throw Error();await loadBlob(await response.blob(),'예제_'+name+'.png',sequence);}
  catch{if(sequence===state.loading)toast('예제를 불러올 수 없습니다. 서버 연결을 확인해 주세요.',true);}
  finally{if(sequence===state.loading){state.importing=false;controls();}}
};
document.addEventListener('paste',e=>{
  if(e.target.matches('textarea,input:not([type=range])'))return;
  const item=Array.from(e.clipboardData?.items||[]).find(item=>item.type.startsWith('image/'));
  if(item){e.preventDefault();loadBlob(item.getAsFile(),'붙여넣은 이미지.png');}
});
document.addEventListener('dragover',e=>e.preventDefault());
document.addEventListener('drop',e=>e.preventDefault());
const drop=$('drop-zone');let dragDepth=0;
drop.addEventListener('dragenter',e=>{e.preventDefault();dragDepth++;if(!state.busy)drop.classList.add('drag-over');});
drop.addEventListener('dragleave',()=>{if(--dragDepth<=0){drop.classList.remove('drag-over');dragDepth=0;}});
drop.addEventListener('drop',e=>{e.preventDefault();dragDepth=0;drop.classList.remove('drag-over');const file=e.dataTransfer.files[0];if(file)loadBlob(file,file.name);});

controlIds.forEach((id,i)=>{
  $(id).addEventListener('input',()=>{state.params[keys[i]]=Number($(id).value);changeTransform();});
  $(id).addEventListener('change',remember);
});
for(const [id,turn] of [['rotate-left',-90],['rotate-right',90]]) $(id).onclick=()=>{state.params.quarter=(state.params.quarter+turn)%360;changeTransform();remember();};
$('reset-angle').onclick=()=>{state.params.angle=0;state.params.quarter=0;changeTransform();remember();};
$('reset-adjust').onclick=()=>{state.params=defaults();changeTransform();remember();};
$('reset').onclick=()=>{state.base=state.original;state.params=defaults();state.crop=null;state.zoom=1;render();remember();};
$('undo').onclick=()=>{
  if(!history.length||state.busy)return;
  Object.assign(state,history.pop());checkpoint=snapshot();clearResult();render();
};
$('crop-tool').onclick=()=>toast('이미지 위를 드래그하세요. 선택 테두리로 크기와 위치를 조절할 수 있어요.');
$('clear-crop').onclick=()=>{state.crop=null;updateSelection();remember();};
$('apply-crop').onclick=()=>{
  if(!state.crop)return;
  state.base=extracted(state.rendered,state.crop);state.params=defaults();state.crop=null;state.zoom=1;
  render();remember();toast('선택한 수식 영역을 잘랐습니다.');
};
const point=e=>{
  const bounds=$('crop-overlay').getBoundingClientRect();
  return {x:Math.max(0,Math.min(state.rendered.width,(e.clientX-bounds.left)*state.rendered.width/bounds.width)),y:Math.max(0,Math.min(state.rendered.height,(e.clientY-bounds.top)*state.rendered.height/bounds.height))};
};
$('crop-overlay').addEventListener('pointerdown',e=>{
  if(state.busy||state.importing||!state.rendered||e.button!==0)return;e.preventDefault();
  const p=point(e),r=state.crop,handle=e.target.dataset.handle;
  const inside=r&&p.x>=r.x&&p.x<=r.x+r.w&&p.y>=r.y&&p.y<=r.y+r.h;
  drag={start:p,rect:r?{...r}:null,mode:handle||(inside?'move':'new')};
  $('crop-overlay').setPointerCapture(e.pointerId);
  if(drag.mode==='new')state.crop={x:p.x,y:p.y,w:0,h:0};
});
$('crop-overlay').addEventListener('pointermove',e=>{
  if(!drag||state.busy||state.importing)return;
  const p=point(e),{start,rect,mode}=drag,w=state.rendered.width,h=state.rendered.height;
  if(mode==='new')state.crop=boundedRect(start.x,start.y,p.x,p.y,w,h);
  else if(mode==='move')state.crop={...rect,x:Math.round(Math.max(0,Math.min(w-rect.w,rect.x+p.x-start.x))),y:Math.round(Math.max(0,Math.min(h-rect.h,rect.y+p.y-start.y)))};
  else{
    let x0=rect.x,y0=rect.y,x1=rect.x+rect.w,y1=rect.y+rect.h;
    if(mode.includes('w'))x0=p.x;if(mode.includes('e'))x1=p.x;if(mode.includes('n'))y0=p.y;if(mode.includes('s'))y1=p.y;
    state.crop=boundedRect(x0,y0,x1,y1,w,h);
  }
  updateOverlay();
});
function finishCrop(e) {
  if(!drag)return;drag=null;
  if($('crop-overlay').hasPointerCapture(e.pointerId))$('crop-overlay').releasePointerCapture(e.pointerId);
  if(state.crop&&(state.crop.w<5||state.crop.h<5))state.crop=null;
  updateSelection();remember();
}
$('crop-overlay').addEventListener('pointerup',finishCrop);
$('crop-overlay').addEventListener('pointercancel',finishCrop);
$('zoom-out').onclick=()=>{state.zoom=Math.max(.25,state.zoom/1.25);fit();};
$('zoom-in').onclick=()=>{state.zoom=Math.min(8,state.zoom*1.25);fit();};
$('zoom-fit').onclick=()=>{state.zoom=1;fit();};
new ResizeObserver(fit).observe($('drop-zone'));
document.addEventListener('keydown',e=>{
  if(e.target.matches('textarea,input:not([type=range])'))return;
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();$('undo').click();}
  if(e.key==='Escape'&&state.crop&&!state.busy){state.crop=null;updateSelection();remember();}
  if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();$('recognize').click();}
});

function mathNode(kind,...children) {
  const element=document.createElementNS('http://www.w3.org/1998/Math/MathML',kind);
  for(const child of children)element.append(typeof child==='string'?document.createTextNode(child):child);
  return element;
}
function mathTree(tree) {
  const [kind,a,b]=tree;
  if(kind==='round'||kind==='square')return mathNode('mrow',mathNode('mo',kind==='round'?'(':'['),mathTree(a),mathNode('mo',kind==='round'?')':']'));
  if(kind==='symbol'){
    const node=mathNode(/^\d+(\.\d+)?$/.test(a)?'mn':functions.includes(a)?'mo':['\\int','\\oint','\\sum','\\prod'].includes(a)?'mo':'mi',glyphs[a]||(functions.includes(a)?a.slice(1):a));
    if(functions.includes(a))node.setAttribute('mathvariant','normal');
    return node;
  }
  if(kind==='div')return mathNode('mfrac',mathTree(a),mathTree(b));
  if(kind==='sqrt')return mathNode('msqrt',mathTree(a));
  if(kind==='sup'||kind==='sub')return mathNode(kind==='sup'?'msup':'msub',mathTree(a),mathTree(b));
  if(kind==='multiply')return mathNode('mrow',mathTree(a),mathTree(b));
  if(kind==='roman'){const node=mathNode('mstyle',mathTree(a));node.setAttribute('mathvariant','normal');return node;}
  if(['vec','hat','bar','dot','ddot'].includes(kind)){
    const node=mathNode('mover',mathTree(a),mathNode('mo',({vec:'→',hat:'^',bar:'¯',dot:'˙',ddot:'¨'})[kind]));node.setAttribute('accent','true');return node;
  }
  if(kind==='positive'||kind==='negative')return mathNode('mrow',mathNode('mo',kind==='positive'?'+':'−'),mathNode('mo','('),mathTree(a),mathNode('mo',')'));
  return mathNode('mrow',mathNode('mo','('),mathTree(a),mathNode('mo',({add:'+',subtraction:'−',eq:'=',approx:'≈',neq:'≠',leq:'≤',geq:'≥',propto:'∝',pm:'±',dotproduct:'⋅',crossproduct:'×'})[kind]),mathTree(b),mathNode('mo',')'));
}
async function recognize() {
  if(state.busy||state.importing||!state.ready||!state.rendered)return;
  const image=extracted(state.rendered,state.crop);
  state.busy=true;controls();clearResult();document.body.classList.add('busy');
  $('recognize-label').textContent='인식 중…';$('result-status').textContent='인식 중';
  const abort=new AbortController(), timeout=setTimeout(()=>abort.abort(),60000);
  try{
    const blob=await new Promise(resolve=>image.toBlob(resolve,'image/png'));
    const response=await fetch('/api/recognize?profile='+encodeURIComponent(state.profile),{method:'POST',headers:{'Content-Type':'image/png'},body:blob,signal:abort.signal});
    const result=await response.json();if(!response.ok)throw Error(result.error||'인식 요청을 완료하지 못했습니다.');
    $('latex-output').disabled=false;$('latex-output').value=result.latex||result.prediction;
    $('copy-latex').disabled=false;$('download-tex').disabled=!result.syntax_valid;
    $('raw-output').textContent=result.prediction;
    $('prepared-image').src=result.prepared_image;$('prepared-image').hidden=false;
    if(result.segmentation_overlay){
      if(!$('segments-details')){
        const details=document.createElement('details');details.id='segments-details';details.className='raw-details';
        const summary=document.createElement('summary');summary.textContent='기호 분할 결과';const img=document.createElement('img');img.id='segments-image';img.alt='검출한 기호 영역';img.style.cssText='max-width:100%;max-height:220px;object-fit:contain;background:white;margin-top:10px';
        details.append(summary,img);$('result-meta').after(details);
      }
      $('segments-image').src=result.segmentation_overlay;$('segments-details').hidden=false;
    }
    $('prepared-size').textContent=`모델 입력 ${result.prepared_size.join(' × ')} px · 흰 여백 자동 제거`;
    const valid=result.syntax_valid&&result.terminated;
    $('result-status').textContent=valid?(state.handwriting?'실험 결과 · 확인 필요':'인식 완료'):'결과 확인 필요';
    $('result-status').className='status-pill '+(valid&&!state.handwriting?'success':'warning');
    $('result-meta').replaceChildren(Object.assign(document.createElement('span'),{textContent:`${result.elapsed_ms}ms · ${result.device==='cuda'?'GPU':'CPU'}`}),Object.assign(document.createElement('span'),{textContent:'수식과 대조해 확인하세요'}));$('result-meta').hidden=false;
    if(result.ast){const math=mathNode('math',mathTree(result.ast));math.setAttribute('display','block');$('formula-preview').replaceChildren(math);}
    else{$('formula-preview').replaceChildren(Object.assign(document.createElement('span'),{className:'preview-placeholder',textContent:'수식 구조를 해석하지 못했어요. 원시 결과를 확인하고 영역을 다시 선택해 주세요.'}));$('edit-hint').textContent='변환하지 못한 원시 결과입니다. 수식 영역을 조정해 다시 인식해 주세요.';}
    if(!result.terminated)toast('출력 길이에 도달했습니다. 더 짧은 수식 영역을 선택해 주세요.',true);
  }catch(error){$('result-status').textContent='인식 실패';$('result-status').className='status-pill warning';toast(error.name==='AbortError'?'응답 시간이 초과되었습니다. 서버를 확인해 주세요.':error.message==='Failed to fetch'?'서버 연결이 끊겼습니다. UI 서버를 다시 실행해 주세요.':error.message,true);}
  finally{clearTimeout(timeout);state.busy=false;document.body.classList.remove('busy');$('recognize-label').textContent='수식 인식하기';controls();}
}
$('recognize').onclick=recognize;
$('latex-output').addEventListener('input',()=>{
  $('copy-latex').disabled=!$('latex-output').value.trim();$('download-tex').disabled=!$('latex-output').value.trim();
  $('edit-hint').textContent='코드를 수정했어요. 위 수식 미리보기는 최초 인식 결과입니다.';
});
$('copy-latex').onclick=async()=>{
  try{await navigator.clipboard.writeText($('latex-output').value);toast('LaTeX 코드를 복사했습니다.');}
  catch{$('latex-output').focus();$('latex-output').select();toast('코드를 선택했습니다. Ctrl+C로 복사해 주세요.');}
};
function save(blob,name) {
  const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$('download-image').onclick=()=>extracted(state.rendered,state.crop).toBlob(blob=>save(blob,'formula-edited.png'),'image/png');
$('download-tex').onclick=()=>save(new Blob([$('latex-output').value+'\n'],{type:'text/plain;charset=utf-8'}),'formula.tex');
async function connect(profile='') {
  state.ready=false;controls();
  try{
    const response=await fetch('/api/status'+(profile?'?profile='+encodeURIComponent(profile):''));if(!response.ok)throw Error();const info=await response.json();state.ready=info.ready;
    state.profile=info.profile;
    $('model-profile').replaceChildren(...info.profiles.map(p=>Object.assign(document.createElement('option'),{value:p.id,textContent:p.name,selected:p.id===info.profile})));
    glyphs=info.glyphs||{};functions=info.functions||[];
    const physics=info.syntax==='physics-v1';
    const handwriting=['handwriting-v1','segmentation-v1'].includes(info.preprocessing);
    state.handwriting=handwriting;
    $('guide-symbol').textContent=physics?'ω = 2πf':'a / b';
    $('guide-copy').textContent=handwriting?'밝은 배경의 짧고 분명한 손글씨 수식을 넣어 주세요. 종이의 조명 차이는 자동으로 보정해요. 결과는 원본과 대조해 확인하세요.':physics?'그리스 문자·벡터·분수·삼각함수를 포함한 짧은 인쇄 수식을 학습했어요. 복잡한 식은 원본과 대조해 확인해 주세요.':'기존 모델은 짧은 인쇄 수식을 학습했어요. 분수는 (a+b)/(c)처럼 슬래시로 적힌 이미지에 맞춰져 있습니다.';
    if(handwriting&&typeof info.evaluation.handwriting_match==='number'){
      $('guide-copy').textContent=`손글씨 시험 ${info.evaluation.handwriting_samples}개에서 전체 수식 일치율 ${(info.evaluation.handwriting_match*100).toFixed(1)}%인 실험 모델입니다. 짧은 수식의 LaTeX 초안을 만들고 원본과 대조해 수정해 주세요. 종이의 조명 차이는 자동으로 보정해요.`;
      if(info.evaluation.handwriting_match<.1)$('guide-copy').textContent=`손글씨 시험 ${info.evaluation.handwriting_samples}개에서 전체 수식 일치율 ${(info.evaluation.handwriting_match*100).toFixed(1)}%로, 실사용에 부족한 실험 모델입니다. 인식 결과는 테스트용으로 확인해 주세요. 종이의 조명 차이는 자동으로 보정해요.`;
    }
    $('guide-details').textContent=physics?'영문 대소문자, 숫자·소수, 그리스 문자 37종, 위·아래 첨자, 쌓인 분수와 슬래시 분수, 근호, 삼각함수, 벡터와 기본 미적분 기호. 한 번에 수식 한 줄을 선택하세요. 복잡한 중첩·행렬·여러 줄 수식은 지원 범위 밖입니다.':'숫자 1–9, x·y·z·a·b·c, + − = /, 괄호, 위·아래 첨자, 제곱근. 손글씨와 복잡한 수식은 후속 학습이 필요합니다.';
    $('guide-model').textContent=`CNN + BiLSTM + LSTM · ${info.model} · Attention / CTC 미사용`;
    if(info.architecture)$('guide-model').textContent=`${info.architecture} · Attention / CTC 미사용`;
    if(info.scope)$('guide-details').textContent=info.scope;
    document.querySelectorAll('[data-sample]').forEach((button,i)=>{
      const sample=info.samples[i];button.dataset.sample=sample.id;button.setAttribute('aria-label',sample.label+' 예제 열기');
      button.replaceChildren(Object.assign(document.createElement('span'),{textContent:sample.symbol}),document.createTextNode(sample.label));
    });
    $('connection').className='connection ready';$('connection').lastElementChild.textContent=`모델 준비 완료 · ${info.device==='cuda'?'GPU':'CPU'}`;
  }catch{state.ready=false;$('connection').className='connection error';$('connection').lastElementChild.textContent='모델 연결 실패';setTimeout(()=>connect(profile),5000);}
  controls();
}
$('model-profile').onchange=()=>{clearResult();connect($('model-profile').value);};
connect();
