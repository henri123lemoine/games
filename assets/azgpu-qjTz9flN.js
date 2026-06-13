(function(){const s=document.createElement("link").relList;if(s&&s.supports&&s.supports("modulepreload"))return;for(const t of document.querySelectorAll('link[rel="modulepreload"]'))i(t);new MutationObserver(t=>{for(const o of t)if(o.type==="childList")for(const a of o.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&i(a)}).observe(document,{childList:!0,subtree:!0});function e(t){const o={};return t.integrity&&(o.integrity=t.integrity),t.referrerPolicy&&(o.referrerPolicy=t.referrerPolicy),t.crossOrigin==="use-credentials"?o.credentials="include":t.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function i(t){if(t.ep)return;t.ep=!0;const o=e(t);fetch(t.href,o)}})();const O=`
struct Params { c_in: u32, c_out: u32, k: u32, relu: u32, residual: u32, batch: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> res: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> P: Params;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let co = wg.x; let b = wg.y; let sq = li.x;
  if (b >= P.batch) { return; }
  let yy = i32(sq / 8u); let xx = i32(sq % 8u);
  var acc = bias[co];
  let half = i32(P.k) / 2;
  for (var ci = 0u; ci < P.c_in; ci++) {
    let xbase = (b * P.c_in + ci) * 64u;
    let wbase = (co * P.c_in + ci) * P.k * P.k;
    for (var dy = -half; dy <= half; dy++) {
      let sy = yy + dy;
      if (sy < 0 || sy > 7) { continue; }
      for (var dx = -half; dx <= half; dx++) {
        let sx = xx + dx;
        if (sx < 0 || sx > 7) { continue; }
        let wi = wbase + u32(dy + half) * P.k + u32(dx + half);
        acc += w[wi] * x[xbase + u32(sy * 8 + sx)];
      }
    }
  }
  let oi = (b * P.c_out + co) * 64u + sq;
  if (P.residual == 1u) { acc += res[oi]; }
  if (P.relu == 1u) { acc = max(acc, 0.0); }
  y[oi] = acc;
}`,C=`
struct Params { n_in: u32, n_out: u32, act: u32, batch: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gi: vec3<u32>) {
  let idx = gi.x;
  if (idx >= P.n_out * P.batch) { return; }
  let b = idx / P.n_out; let o = idx % P.n_out;
  var acc = bias[o];
  let xb = b * P.n_in; let wb = o * P.n_in;
  for (var i = 0u; i < P.n_in; i++) { acc += w[wb + i] * x[xb + i]; }
  if (P.act == 1u) { acc = max(acc, 0.0); }
  if (P.act == 2u) { acc = tanh(clamp(acc, -15.0, 15.0)); }
  y[(b * P.n_out) + o] = acc;
}`,G=18,B=4672,f=32;function M(g){const s=new DataView(g),e=new TextDecoder().decode(g.slice(0,8));if(e!=="AZWEB001")throw new Error("bad magic: "+e);const i=s.getUint32(8,!0),t=s.getUint32(12,!0);let o=16;const a=l=>{const c=new Float32Array(g,o,l);return o+=l*4,c},r=(l,c,h)=>({w:a(c*l*h*h),b:a(c),ci:l,co:c,k:h}),n=r(G,t,3),d=[];for(let l=0;l<i;l++)d.push([r(t,t,3),r(t,t,3)]);const P=r(t,t,1),x=r(t,73,1),m=r(t,8,1),v={w:a(256*512),b:a(256),ni:512,no:256},p={w:a(256),b:a(1),ni:256,no:1};if(o!==g.byteLength)throw new Error("trailing bytes: "+(g.byteLength-o));return{blocks:i,C:t,stem:n,tower:d,p1:P,p2:x,v1:m,vf1:v,vf2:p}}class A{model;dev;convPipe;linPipe;layers=[];inBuf;polBuf;v1out;stagePol;stageVal;batch=0;queue=Promise.resolve();get lost(){return this.dev.lost}destroy(){this.dev.destroy()}static async init(s){const e=new A,i=await navigator.gpu?.requestAdapter();if(!i)throw new Error("WebGPU unavailable");e.dev=await i.requestDevice(),e.model=M(s);const t=e.dev,o=e.model.C;e.convPipe=t.createComputePipeline({layout:"auto",compute:{module:t.createShaderModule({code:O}),entryPoint:"main"}}),e.linPipe=t.createComputePipeline({layout:"auto",compute:{module:t.createShaderModule({code:C}),entryPoint:"main"}});const a=u=>{const b=t.createBuffer({size:u.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});return t.queue.writeBuffer(b,0,u),b},r=u=>t.createBuffer({size:u*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}),n=r(f*o*64),d=r(f*o*64),P=r(f*o*64),x=r(16);e.inBuf=r(f*G*64),e.polBuf=r(f*73*64);const m=r(f*8*64),v=r(f*256);e.v1out=r(f),e.stagePol=t.createBuffer({size:f*73*64*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),e.stageVal=t.createBuffer({size:f*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const p=(u,b,_,U,y)=>{const w=t.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),L=t.createBindGroup({layout:e.convPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:b}},{binding:1,resource:{buffer:a(u.w)}},{binding:2,resource:{buffer:a(u.b)}},{binding:3,resource:{buffer:y??x}},{binding:4,resource:{buffer:_}},{binding:5,resource:{buffer:w}}]});return{kind:"conv",u:w,bg:L,cv:u,relu:U,residual:!!y}},l=(u,b,_,U)=>{const y=t.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),w=t.createBindGroup({layout:e.linPipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:b}},{binding:1,resource:{buffer:a(u.w)}},{binding:2,resource:{buffer:a(u.b)}},{binding:3,resource:{buffer:_}},{binding:4,resource:{buffer:y}}]});return{kind:"lin",u:y,bg:w,l:u,act:U}};e.layers.push(p(e.model.stem,e.inBuf,n,1,null));let c=n,h=d;for(const[u,b]of e.model.tower)e.layers.push(p(u,c,P,1,null)),e.layers.push(p(b,P,h,1,c)),[c,h]=[h,c];return e.layers.push(p(e.model.p1,c,P,1,null)),e.layers.push(p(e.model.p2,P,e.polBuf,0,null)),e.layers.push(p(e.model.v1,c,m,1,null)),e.layers.push(l(e.model.vf1,m,v,1)),e.layers.push(l(e.model.vf2,v,e.v1out,2)),e}setBatch(s){if(s!==this.batch){this.batch=s;for(const e of this.layers)e.kind==="conv"?this.dev.queue.writeBuffer(e.u,0,new Uint32Array([e.cv.ci,e.cv.co,e.cv.k,e.relu,e.residual?1:0,s])):this.dev.queue.writeBuffer(e.u,0,new Uint32Array([e.l.ni,e.l.no,e.act,s]))}}forward(s,e){const i=this.queue.then(()=>this.forwardNow(s,e));return this.queue=i.catch(()=>{}),i}async forwardNow(s,e){if(e<1||e>f)throw new Error(`batch ${e} out of range 1..${f}`);this.setBatch(e),this.dev.queue.writeBuffer(this.inBuf,0,s);const i=this.dev.createCommandEncoder();for(const r of this.layers){const n=i.beginComputePass();r.kind==="conv"?(n.setPipeline(this.convPipe),n.setBindGroup(0,r.bg),n.dispatchWorkgroups(r.cv.co,e)):(n.setPipeline(this.linPipe),n.setBindGroup(0,r.bg),n.dispatchWorkgroups(Math.ceil(r.l.no*e/64))),n.end()}i.copyBufferToBuffer(this.polBuf,0,this.stagePol,0,e*73*64*4),i.copyBufferToBuffer(this.v1out,0,this.stageVal,0,e*4),this.dev.queue.submit([i.finish()]);try{await Promise.all([this.stagePol.mapAsync(GPUMapMode.READ,0,e*73*64*4),this.stageVal.mapAsync(GPUMapMode.READ,0,e*4)])}catch(r){throw this.stagePol.unmap(),this.stageVal.unmap(),r}let t,o;try{t=new Float32Array(this.stagePol.getMappedRange(0,e*73*64*4).slice(0)),o=new Float32Array(this.stageVal.getMappedRange(0,e*4).slice(0))}finally{this.stagePol.unmap(),this.stageVal.unmap()}const a=new Float32Array(e*B);for(let r=0;r<e;r++)for(let n=0;n<73;n++)for(let d=0;d<64;d++)a[r*B+d*73+n]=t[(r*73+n)*64+d];return{logits:a,values:o}}}function q(g,s,e=0){const i=Array.from(s,r=>g[e+r]),t=Math.max(...i),o=i.map(r=>Math.exp(r-t)),a=o.reduce((r,n)=>r+n,0);return o.map(r=>r/a)}export{A,G as P,B as a,q as s};
