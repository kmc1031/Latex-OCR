// Coordinate and affine operations are independent of screen zoom and CSS pixels.
export const defaults = () => ({angle:0, quarter:0, shearX:0, shearY:0, scaleX:100, scaleY:100});
export function affineGeometry(width, height, p) {
  const r = (p.angle + p.quarter) * Math.PI / 180;
  const sx = p.scaleX / 100, sy = p.scaleY / 100;
  const kx = Math.tan(p.shearX * Math.PI / 180), ky = Math.tan(p.shearY * Math.PI / 180);
  const cos = Math.cos(r), sin = Math.sin(r);
  // R * shear * scale; Canvas matrix is [a c e; b d f; 0 0 1].
  const a = (cos - sin * ky) * sx, b = (sin + cos * ky) * sx;
  const c = (cos * kx - sin) * sy, d = (sin * kx + cos) * sy;
  const outW = Math.max(1, Math.ceil(Math.abs(a) * width + Math.abs(c) * height - 1e-8));
  const outH = Math.max(1, Math.ceil(Math.abs(b) * width + Math.abs(d) * height - 1e-8));
  return {a,b,c,d,width:outW,height:outH,e:outW/2-(a*width+c*height)/2,f:outH/2-(b*width+d*height)/2};
}
export function boundedRect(x0,y0,x1,y1,width,height) {
  const left=Math.round(Math.max(0,Math.min(x0,x1,width))), top=Math.round(Math.max(0,Math.min(y0,y1,height)));
  const right=Math.round(Math.max(0,Math.min(Math.max(x0,x1),width))), bottom=Math.round(Math.max(0,Math.min(Math.max(y0,y1),height)));
  return {x:left,y:top,w:right-left,h:bottom-top};
}
export function canvasOf(width,height) {
  const canvas=document.createElement('canvas'); canvas.width=width;canvas.height=height;
  const ctx=canvas.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,width,height);return canvas;
}
export function transformed(source,params) {
  const m=affineGeometry(source.width,source.height,params);
  const factor=Math.min(1,4096/Math.max(m.width,m.height),Math.sqrt(16000000/(m.width*m.height)));
  const canvas=canvasOf(Math.max(1,Math.ceil(m.width*factor)),Math.max(1,Math.ceil(m.height*factor))), ctx=canvas.getContext('2d');
  ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';
  ctx.setTransform(m.a*factor,m.b*factor,m.c*factor,m.d*factor,m.e*factor,m.f*factor);ctx.drawImage(source,0,0);return canvas;
}
export function extracted(source,rect) {
  if (!rect) return source;
  const canvas=canvasOf(rect.w,rect.h);
  canvas.getContext('2d').drawImage(source,rect.x,rect.y,rect.w,rect.h,0,0,rect.w,rect.h);return canvas;
}
