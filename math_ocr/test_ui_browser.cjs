// Optional browser integration checks. Uses the Codex bundled Playwright or
// FORMULA_PLAYWRIGHT; it does not install browser packages into the project.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const {pathToFileURL}=require('node:url');
const {chromium} = require(process.env.FORMULA_PLAYWRIGHT || path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const root = __dirname;

(async()=>{
  const executable=chromium.executablePath();
  const browser=await chromium.launch(fs.existsSync(executable)?{headless:true}:{headless:true,channel:'msedge'});
  const page=await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  page.on('console',msg=>{if(msg.type()==='error')errors.push(msg.text());});
  const checks=[];
  try{
    await page.goto('http://127.0.0.1:8765/',{waitUntil:'networkidle'});
    assert.match(await page.locator('#connection').innerText(),/준비 완료/);
    if(await page.locator('#model-profile option[value="legacy"]').count()){
      await page.locator('#model-profile').selectOption('legacy');
      await page.waitForFunction(()=>document.getElementById('model-profile').value==='legacy'&&!document.getElementById('model-profile').disabled);
    }
    await page.screenshot({path:path.join(root,'ui_preview_empty.png'),fullPage:true});
    await page.getByRole('button',{name:'분수 예제 열기'}).click();
    await page.waitForFunction(()=>!document.getElementById('recognize').disabled);
    let response=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/recognize');
    await page.locator('#recognize').click();
    const fraction=await (await response).json();
    assert.ok(fraction.syntax_valid);assert.match(fraction.latex,/\\frac/);
    await page.waitForFunction(()=>!document.body.classList.contains('busy'));
    checks.push('sample selection and real fraction recognition');

    await page.locator('#file-input').setInputFiles(path.join(root,'tmp/ui_fixture.png'));
    await page.waitForFunction(()=>document.getElementById('image-info').textContent.includes('ui_fixture'));
    const image=page.locator('#editor-canvas');
    const original=await image.evaluate(c=>c.toDataURL());
    // Keyboard range changes exercise both input preview and change history.
    await page.locator('#shear-x').focus();await page.keyboard.press('ArrowRight');
    assert.match(await page.locator('#shear-x-value').textContent(),/0.5/);
    assert.notEqual(await image.evaluate(c=>c.toDataURL()),original);
    await page.locator('#undo').click();
    assert.equal(await image.evaluate(c=>c.toDataURL()),original);
    await page.locator('#rotate-right').click();
    assert.deepEqual(await image.evaluate(c=>[c.width,c.height]),[500,1000]);
    await page.locator('#undo').click();
    assert.equal(await image.evaluate(c=>c.toDataURL()),original);
    await page.locator('#scale-x').focus();await page.keyboard.press('ArrowRight');
    assert.equal(await image.evaluate(c=>c.width),1010);
    await page.locator('#reset-adjust').click();
    assert.equal(await image.evaluate(c=>c.toDataURL()),original);
    checks.push('affine preview, quarter rotation, scale, reset, and pixel-exact undo');

    // A sheet contains other writing; select only the formula at native pixel coordinates.
    const bounds=await image.boundingBox();
    const coord=(x,y)=>[bounds.x+x/1000*bounds.width,bounds.y+y/500*bounds.height];
    await page.mouse.move(...coord(76,170));await page.mouse.down();
    await page.mouse.move(...coord(924,382),{steps:12});await page.mouse.up();
    assert.match(await page.locator('#selection-title').innerText(),/선택 영역/);
    const selection=await page.locator('#selection-preview').evaluate(img=>[img.naturalWidth,img.naturalHeight]);
    assert.ok(Math.abs(selection[0]-848)<=2&&Math.abs(selection[1]-212)<=2);
    const handle=await page.locator('[data-handle="se"]').boundingBox();
    await page.mouse.move(handle.x+handle.width/2,handle.y+handle.height/2);await page.mouse.down();
    await page.mouse.move(...coord(944,402),{steps:4});await page.mouse.up();
    const resized=await page.locator('#selection-preview').evaluate(img=>[img.naturalWidth,img.naturalHeight]);
    assert.ok(resized[0]>selection[0]&&resized[1]>selection[1]);
    await page.locator('#undo').click();
    assert.deepEqual(await page.locator('#selection-preview').evaluate(img=>[img.naturalWidth,img.naturalHeight]),selection);
    response=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/recognize');
    await page.locator('#recognize').click();
    const recognition=await (await response).json();
    assert.equal(recognition.prediction,'y _ { 6 } + c + x');
    await page.waitForFunction(()=>!document.body.classList.contains('busy'));
    assert.ok(await page.locator('#formula-preview math').count());
    await page.screenshot({path:path.join(root,'ui_preview.png'),fullPage:true});
    checks.push('upload, native-coordinate crop, cropped inference, MathML preview');

    const pngDownload=page.waitForEvent('download');await page.locator('#download-image').click();
    assert.equal((await pngDownload).suggestedFilename(),'formula-edited.png');
    const texDownload=page.waitForEvent('download');await page.locator('#download-tex').click();
    assert.equal((await texDownload).suggestedFilename(),'formula.tex');
    await page.locator('#latex-output').fill('x+1');
    assert.match(await page.locator('#edit-hint').innerText(),/코드를 수정/);
    checks.push('edited PNG and TeX downloads, editable LaTeX');
    await page.locator('#apply-crop').click();
    assert.deepEqual(await image.evaluate(c=>[c.width,c.height]),selection);
    await page.locator('#reset').click();
    assert.equal(await image.evaluate(c=>c.toDataURL()),original);
    checks.push('crop commit and original restore');
    await page.setViewportSize({width:390,height:844});
    await page.waitForFunction(()=>document.getElementById('toast').hidden);
    await page.screenshot({path:path.join(root,'ui_preview_mobile.png'),fullPage:true});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
    const card=await page.locator('.result-card').boundingBox();assert.ok(card.width>350);
    checks.push('mobile layout without horizontal page overflow');
    if(await page.locator('#model-profile option[value="physics"]').count()){
      await page.setViewportSize({width:1440,height:1100});
      await page.locator('#model-profile').selectOption('physics');
      await page.waitForFunction(()=>document.getElementById('model-profile').value==='physics'&&!document.getElementById('model-profile').disabled);
      // Examples live in the empty-image view. Check the profile change first,
      // then reload to open a handwriting example through the visible UI.
      await page.reload({waitUntil:'networkidle'});
      await page.getByRole('button',{name:'그리스 문자 예제 열기'}).waitFor();
      await page.route('**/api/sample/handwriting_0?*',async route=>{
        await new Promise(resolve=>setTimeout(resolve,250));await route.continue();
      });
      await page.getByRole('button',{name:'그리스 문자 예제 열기'}).click();
      assert.ok(await page.locator('#recognize').isDisabled());
      assert.ok(await page.locator('#model-profile').isDisabled());
      await page.waitForFunction(()=>document.getElementById('image-info').textContent.includes('예제_handwriting_0'));
      await page.waitForFunction(()=>!document.getElementById('recognize').disabled);
      const pending=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/recognize');
      await page.locator('#recognize').click();
      const handwritten=await(await pending).json();
      assert.ok(handwritten.syntax_valid&&handwritten.terminated);
      await page.waitForFunction(()=>!document.body.classList.contains('busy'));
      const math=await page.locator('#formula-preview math').textContent();
      assert.match(math,/[\u0370-\u03ff]/);
      assert.match(await page.locator('#guide-copy').textContent(),/손글씨/);
      assert.match(await page.locator('#result-status').textContent(),/실험 결과.*확인 필요/);
      await page.screenshot({path:path.join(root,'ui_preview_handwriting.png'),fullPage:true});
      checks.push('handwriting profile, Greek sample recognition and native MathML rendering');
    }
    const galleryPath=path.join(root,'physics/handwriting_gallery.html');
    if(await page.locator('#model-profile option[value="segments"]').count()){
      const p=await browser.newPage({viewport:{width:1440,height:1100}});
      p.on('pageerror',error=>errors.push(error.message));
      await p.goto('http://127.0.0.1:8765/',{waitUntil:'networkidle'});
      await p.locator('#model-profile').selectOption('segments');
      await p.waitForFunction(()=>document.getElementById('model-profile').value==='segments'&&!document.getElementById('model-profile').disabled);
      await p.locator('[data-sample]').first().click();
      await p.waitForFunction(()=>document.getElementById('image-info').textContent.includes('segments_0')&&!document.getElementById('recognize').disabled);
      const pending=p.waitForResponse(r=>new URL(r.url()).pathname==='/api/recognize');await p.locator('#recognize').click();
      const result=await(await pending).json();assert.ok(result.boxes.length>0);
      await p.waitForFunction(()=>!document.body.classList.contains('busy'));
      await p.locator('#segments-details summary').click();
      assert.ok(await p.locator('#segments-image').isVisible());
      await p.screenshot({path:path.join(root,'ui_preview_segmentation.png'),fullPage:true});await p.close();
      checks.push('segmentation CNN profile, real predicted boxes and visual overlay');
    }
    if(fs.existsSync(galleryPath)){
      const gallery=await browser.newPage({viewport:{width:1440,height:1100}});
      gallery.on('pageerror',error=>errors.push(error.message));
      await gallery.goto(pathToFileURL(galleryPath).href,{waitUntil:'load'});
      assert.equal(await gallery.locator('article').count(),800);
      assert.match(await gallery.locator('article').first().locator('math').first().textContent(),/F\(y\)/);
      await gallery.screenshot({path:path.join(root,'physics/handwriting_gallery_preview.png')});
      await gallery.getByRole('button',{name:'오류만'}).click();
      const metrics=JSON.parse(fs.readFileSync(path.join(root,'physics/handwriting_metrics.json'),'utf8')).handwriting.test;
      assert.equal(await gallery.locator('article:not([hidden])').count(),800-Math.round(metrics.semantic_exact*800));
      await gallery.getByRole('button',{name:'그리스 문자',exact:true}).click();
      assert.equal(await gallery.locator('article:not([hidden])').count(),300);
      await gallery.close();
      checks.push('all 800 held-out examples rendered, error and Greek filters match measured counts');
    }
    const segmentationGallery=path.join(root,'segmentation/gallery.html');
    if(fs.existsSync(segmentationGallery)){
      const p=await browser.newPage({viewport:{width:1440,height:1100}});p.on('pageerror',error=>errors.push(error.message));
      await p.goto(pathToFileURL(segmentationGallery).href,{waitUntil:'load'});
      const metrics=JSON.parse(fs.readFileSync(path.join(root,'segmentation/test_metrics.json'),'utf8'));
      assert.equal(await p.locator('article').count(),metrics.n);
      await p.screenshot({path:path.join(root,'segmentation/gallery_preview.png')});
      await p.locator('#errors').click();
      assert.equal(await p.locator('article:not([hidden])').count(),metrics.n-metrics.modes[metrics.selected_automatic_method].correct);
      await p.close();checks.push('all segmentation test cases rendered with oracle and automatic predictions separated');
    }
    assert.deepEqual(errors,[]);
    const report={passed:true,checks,errors,fraction: fraction.prediction,cropped_prediction:recognition.prediction};
    fs.writeFileSync(path.join(root,'ui_qa.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
