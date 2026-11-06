import test from 'node:test';
import assert from 'node:assert/strict';
import {affineGeometry,defaults,boundedRect} from './editor.js';

test('quarter turns swap dimensions without adding padding or clipping',()=>{
  const m=affineGeometry(320,90,{...defaults(),quarter:90});
  assert.equal(m.width,90);assert.equal(m.height,320);
  assert.ok(Math.abs(m.e-90)<1e-7);assert.ok(Math.abs(m.f)<1e-7);
});
test('shear expands the image and scale preserves independent axes',()=>{
  const m=affineGeometry(100,40,{...defaults(),shearX:45});
  assert.equal(m.width,140);assert.equal(m.height,40);
  const n=affineGeometry(100,40,{...defaults(),scaleX:150,scaleY:50});
  assert.equal(n.width,150);assert.equal(n.height,20);
});
test('every transformed corner remains inside expanded bounds',()=>{
  for(const angle of [-45,-1,0,28,90,180])for(const shearX of [-30,0,30])for(const shearY of [-30,0,30]){
    const m=affineGeometry(317,91,{...defaults(),angle,shearX,shearY,scaleX:140,scaleY:75});
    for(const [x,y] of [[0,0],[317,0],[0,91],[317,91]]){
      const tx=m.a*x+m.c*y+m.e,ty=m.b*x+m.d*y+m.f;
      assert.ok(tx>=-1e-7&&tx<=m.width+1e-7);assert.ok(ty>=-1e-7&&ty<=m.height+1e-7);
    }
  }
});
test('crop uses image coordinates, clamps edges, and accepts reversed drags',()=>{
  assert.deepEqual(boundedRect(80.2,70,-20,10,100,60),{x:0,y:10,w:80,h:50});
  assert.deepEqual(boundedRect(10,20,90,40,100,60),{x:10,y:20,w:80,h:20});
});
