// Unit-test the delivered JS using mock elements; does not control a browser.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
(async () => {
  const root=__dirname, html=fs.readFileSync(path.join(root,'ui_page_snapshot.html'),'utf8');
  const fixture=JSON.parse(fs.readFileSync(path.join(root,'ui_fixture.json'),'utf8'));
  const nodes={};let copied='',fallback=false;
  const get=id=>nodes[id]??=( {value:'',textContent:'',src:'',files:[],focus(){},select(){this.selected=true;} });
  const context={document:{getElementById:get,execCommand(cmd){assert.equal(cmd,'copy');fallback=true;assert(get('latex').selected);}},navigator:{clipboard:{async writeText(s){copied=s;}}},FileReader:class{readAsDataURL(){this.result='data:image/png;base64,fixture';this.onload();}},fetch:async()=>({ok:true,json:async()=>fixture}),console};
  vm.runInNewContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
  get('file').files=[{}];get('file').onchange();assert(get('status').textContent.includes('준비'));assert(get('original').src.startsWith('data:image'));
  await get('run').onclick();assert.equal(get('latex').value,fixture.latex);assert.equal(get('formula').src,fixture.render_png||'');
  await get('copy').onclick();assert.equal(copied,fixture.latex);assert(get('copied').textContent.includes('복사'));
  context.navigator.clipboard.writeText=async()=>{throw Error('mock denied');};await get('copy').onclick();assert(fallback);
  const p=path.join(root,'ui_checks.json'),checks=JSON.parse(fs.readFileSync(p,'utf8'));checks.js_unit_checks={file_preview:true,response_display:true,clipboard_write_with_mock:true,clipboard_fallback_with_mock:true,real_os_clipboard_tested:false};fs.writeFileSync(p,JSON.stringify(checks,null,2),'utf8');console.log('UI JS mock checks passed; live browser/OS clipboard not claimed.');
})().catch(e=>{console.error(e);process.exitCode=1;});
