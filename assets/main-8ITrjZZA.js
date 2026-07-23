const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./index-mgQTu_g6.js","./assets-BX42eQBy.js"])))=>i.map(i=>d[i]);
import{a as et,s as yt,P as Be,d as At,g as Oe,l as Tt}from"./assets-BX42eQBy.js";import{A as He,a as _e}from"./azgpu-nL2OYv-D.js";import{p as Ge,G as Ne}from"./azgpu-DLGXD1iy.js";import{r as Fe}from"./conformance-CcSGGraw.js";const Re="modulepreload",Ie=function(n,t){return new URL(n,t).href},zt={},je=function(t,e,s){let o=Promise.resolve();if(e&&e.length>0){let a=function(d){return Promise.all(d.map(h=>Promise.resolve(h).then(p=>({status:"fulfilled",value:p}),p=>({status:"rejected",reason:p}))))};const r=document.getElementsByTagName("link"),l=document.querySelector("meta[property=csp-nonce]"),c=l?.nonce||l?.getAttribute("nonce");o=a(e.map(d=>{if(d=Ie(d,s),d in zt)return;zt[d]=!0;const h=d.endsWith(".css"),p=h?'[rel="stylesheet"]':"";if(!!s)for(let b=r.length-1;b>=0;b--){const u=r[b];if(u.href===d&&(!h||u.rel==="stylesheet"))return}else if(document.querySelector(`link[href="${d}"]${p}`))return;const f=document.createElement("link");if(f.rel=h?"stylesheet":Re,h||(f.as="script"),f.crossOrigin="",f.href=d,c&&f.setAttribute("nonce",c),document.head.appendChild(f),h)return new Promise((b,u)=>{f.addEventListener("load",b),f.addEventListener("error",()=>u(new Error(`Unable to preload CSS for ${d}`)))})}))}function i(a){const r=new Event("vite:preloadError",{cancelable:!0});if(r.payload=a,window.dispatchEvent(r),!r.defaultPrevented)throw a}return o.then(a=>{for(const r of a||[])r.status==="rejected"&&i(r.reason);return t().catch(i)})};class vt{worker;nextId=1;pending=new Map;constructor(){this.worker=new Worker(new URL(""+new URL("worker-D7W0AEA3.js",import.meta.url).href,import.meta.url),{type:"module"}),this.worker.onmessage=t=>{const e=this.pending.get(t.data.id);e&&(this.pending.delete(t.data.id),t.data.ok?e.resolve(t.data.data):e.reject(new Error(t.data.error)))},this.worker.onerror=t=>this.rejectAll(`engine worker error: ${t.message||"unknown"}`),this.worker.onmessageerror=()=>this.rejectAll("engine worker message error")}rejectAll(t){for(const e of this.pending.values())e.reject(new Error(t));this.pending.clear()}call(t){const e=this.nextId++;return new Promise((s,o)=>{this.pending.set(e,{resolve:s,reject:o}),this.worker.postMessage({...t,id:e})})}manifest(){return this.call({op:"manifest"})}create(t,e){return this.call({op:"create",game:t,opts:e})}step(){return this.call({op:"step"})}prepare(){return this.call({op:"prepare"})}state(){return this.call({op:"state"})}apply(t){return this.call({op:"apply",input:t})}artifact(t,e){return this.call({op:"artifact",key:t,bytes:e})}pairs(t,e,s,o,i,a,r){return this.call({op:"pairs",game:t,opts:e,a:s,b:o,seed:i,lo:a,hi:r})}field(t,e,s,o,i,a,r){return this.call({op:"field",game:t,opts:e,a:s,b:o,seed:i,lo:a,hi:r})}elo(t,e,s){return this.call({op:"elo",w:t,d:e,l:s})}fitElo(t){return this.call({op:"fitElo",records:t})}azNew(t,e,s,o){return this.call({op:"azNew",sims:t,leaves:e,seed:s,weights:o})}goNew(t,e,s,o,i){return this.call({op:"goNew",sims:t,leaves:e,seed:s,size:o,weights:i})}penteNew(t,e,s,o,i,a,r){return this.call({op:"penteNew",sims:t,leaves:e,seed:s,size:o,vcfDepth:i,vcfNodes:a,weights:r})}azPush(t){return this.call({op:"azPush",uci:t})}azAdvance(t,e){return this.call({op:"azAdvance",priors:t,values:e})}azPlayCpu(){return this.call({op:"azPlayCpu"})}azBest(){return this.call({op:"azBest"})}azFinalResult(){return this.call({op:"azFinalResult"})}goEval(){return this.call({op:"goEval"})}penteEval(){return this.call({op:"penteEval"})}chessEval(){return this.call({op:"chessEval"})}terminate(){this.worker.terminate(),this.rejectAll("engine terminated")}}function D(){return!("gpu"in navigator)}const _=1,F=[["Trivial",String(_)],["Light","16"]],wt=16;function kt(n,t){return`CPU FALLBACK ACTIVE: ${n}. AlphaZero is running on the CPU at ${t} ${t===1?"sim":"sims"}, so it will be much slower and weaker than WebGPU.`}let ut=null;function V(n){ut=n}function Ye(){return ut?ut():Promise.resolve(null)}function St(n){let t=null;return()=>(t??=(async()=>{const e=await fetch(n);if(!e.ok)throw new Error(`weights ${n} missing (HTTP ${e.status})`);return e.arrayBuffer()})(),t.catch(()=>{t=null}),t)}function Et(n,t){let e=null;return()=>(e??=(async()=>{const s=await n(await t());return s.lost.then(()=>{e=null}),s})(),e.catch(()=>{e=null}),e)}function P(n,t){const e=n[t],s=Number(e);if(!e||!Number.isInteger(s)||s<=0||s>4294967295)throw new Error(`client bot requires ${t}=1..4294967295, got '${e??""}'`);return s}function $t(n){return n instanceof Error?n.message:String(n)}const Dt=8,ft=St(et("azero/azero-chess.azweb")),We=Et(He.init,ft);class Ue{constructor(t,e){this.host=t,this.gpu=e}cancelled=!1;onMove(t){return this.host.azPush(t.label)}async chooseMove(t){let e=new Float32Array(0),s=new Float32Array(0);for(;;){if(this.cancelled)throw new Error("cancelled");const o=await this.host.azAdvance(e,s);if(o.n===0)break;if(this.cancelled)throw new Error("cancelled");const{logits:i,values:a}=await this.gpu.forward(o.features,o.n),r=[];for(let l=0;l<o.n;l++){const c=o.support.subarray(o.offsets[l],o.offsets[l+1]);r.push(...yt(i,c,l*_e))}e=Float32Array.from(r),s=a.slice(0,o.n)}return(await this.host.azBest()).uci}cancel(){this.cancelled=!0,V(null)}}class Ve{constructor(t,e){this.host=t,this.cpuFallback=e}cancelled=!1;onMove(t){return this.host.azPush(t.label)}async chooseMove(t){if(this.cancelled)throw new Error("cancelled");const{uci:e}=await this.host.azPlayCpu();if(this.cancelled)throw new Error("cancelled");return e}cancel(){this.cancelled=!0,V(null)}}async function Ke(n,t){const e=P(t,"seed"),s=P(t,"sims");let o="No compatible WebGPU device was detected";if(!D()){let a=null;try{a=await We()}catch(r){o=`WebGPU initialization failed: ${$t(r)}`}if(a)return await n.azNew(s,Dt,e,await ft()),V(()=>n.chessEval()),new Ue(n,a)}const i=Math.min(s,wt);return await n.azNew(i,Dt,e,await ft()),V(()=>n.chessEval()),new Ve(n,kt(o,i))}let gt=null;function K(n){gt=n}function Qe(){return gt?gt():Promise.resolve(null)}const Pt=8,Mt=St(et("azero/azero-go.azweb")),ke=Et(Ne.init,Mt),Bt=Mt,Xe=ke;class Ze{constructor(t,e,s){this.host=t,this.gpu=e,this.size=s,this.stride=Ge(s)}cancelled=!1;stride;onMove(t){return this.host.azPush(t.label)}async chooseMove(t){let e=new Float32Array(0),s=new Float32Array(0);for(;;){if(this.cancelled)throw new Error("cancelled");const o=await this.host.azAdvance(e,s);if(o.n===0)break;if(this.cancelled)throw new Error("cancelled");const{logits:i,values:a}=await this.gpu.forward(o.features,o.n,this.size),r=[];for(let l=0;l<o.n;l++){const c=o.support.subarray(o.offsets[l],o.offsets[l+1]);r.push(...yt(i,c,l*this.stride))}e=Float32Array.from(r),s=a.slice(0,o.n)}return(await this.host.azBest()).uci}finalResult(){return this.host.azFinalResult()}cancel(){this.cancelled=!0,K(null)}}class Je{constructor(t,e){this.host=t,this.cpuFallback=e}cancelled=!1;onMove(t){return this.host.azPush(t.label)}async chooseMove(t){if(this.cancelled)throw new Error("cancelled");const{uci:e}=await this.host.azPlayCpu();if(this.cancelled)throw new Error("cancelled");return e}finalResult(){return this.host.azFinalResult()}cancel(){this.cancelled=!0,K(null)}}async function ts(n,t){const e=P(t,"seed"),s=P(t,"sims"),o=P(t,"size");let i="No compatible WebGPU device was detected";if(!D()){let r=null;try{r=await Xe()}catch(l){i=`WebGPU initialization failed: ${$t(l)}`}if(r)return await n.goNew(s,Pt,e,o,await Bt()),K(()=>n.goEval()),new Ze(n,r,o)}const a=Math.min(s,wt);return await n.goNew(a,Pt,e,o,await Bt()),K(()=>n.goEval()),new Je(n,kt(i,a))}function es(n){return n*n+1}class qt extends Be{pgb;pfc;ppass;static async init(t){const e=new qt;return await e.boot(t),e}parseHead(t,e){const s=t.C,o=e.conv(s,s,1);this.pgb=e.linear(3*s,s),this.pfc=e.convNoBias(s,1,1),this.ppass=e.linear(3*s,1);const i=e.conv(s,s,1);return this.v1=e.linear(3*s,At),this.v2=e.linear(At,1),t.ownership&&e.convNoBias(s,1,1),{p1:o,v1:i}}heads(t,e,s,o){const i=this.arch.C,a=o+1,r=new Float32Array(s*a),l=new Float32Array(s);for(let c=0;c<s;c++){const d=c*i*o,h=Oe(t,d,i,o),p=Tt(this.pgb,h,!1),g=c*a;for(let f=0;f<o;f++){let b=0;for(let u=0;u<i;u++){let m=t[d+u*o+f]+p[u];m<0&&(m=0),b+=this.pfc.w[u]*m}r[g+f]=b}r[g+o]=Tt(this.ppass,h,!1)[0],l[c]=this.poolValue(e,d,o)}return{logits:r,values:l}}}let bt=null;function Q(n){bt=n}function ss(){return bt?bt():Promise.resolve(null)}const X=19,Ot=8,mt=St(et("azero/azero-pente.azweb")),os=Et(qt.init,mt);class is{constructor(t,e){this.host=t,this.gpu=e}cancelled=!1;stride=es(X);onMove(t){return this.host.azPush(t.label)}async chooseMove(t){let e=new Float32Array(0),s=new Float32Array(0);for(;;){if(this.cancelled)throw new Error("cancelled");const o=await this.host.azAdvance(e,s);if(o.n===0)break;if(this.cancelled)throw new Error("cancelled");const{logits:i,values:a}=await this.gpu.forward(o.features,o.n,X),r=[];for(let l=0;l<o.n;l++){const c=o.support.subarray(o.offsets[l],o.offsets[l+1]);r.push(...yt(i,c,l*this.stride))}e=Float32Array.from(r),s=a.slice(0,o.n)}return(await this.host.azBest()).uci}cancel(){this.cancelled=!0,Q(null)}}class as{constructor(t,e){this.host=t,this.cpuFallback=e}cancelled=!1;onMove(t){return this.host.azPush(t.label)}async chooseMove(t){if(this.cancelled)throw new Error("cancelled");const{uci:e}=await this.host.azPlayCpu();if(this.cancelled)throw new Error("cancelled");return e}cancel(){this.cancelled=!0,Q(null)}}async function ns(n,t){const e=P(t,"seed"),s=P(t,"sims"),o=P(t,"vcf-depth"),i=P(t,"vcf-nodes");let a="No compatible WebGPU device was detected";if(!D()){let l=null;try{l=await os()}catch(c){a=`WebGPU initialization failed: ${$t(c)}`}if(l)return await n.penteNew(s,Ot,e,X,o,i,await mt()),Q(()=>n.penteEval()),new is(n,l)}const r=Math.min(s,wt);return await n.penteNew(r,Ot,e,X,o,i,await mt()),Q(()=>n.penteEval()),new as(n,kt(a,r))}const rs=new Map([["chess/azero-gpu",Ke],["go/azero-gpu",ts],["pente/azero-gpu",ns]]);function Se(n,t){return t&&rs.get(`${n}/${t}`)||null}async function ls(n,t){if(t.length===0)return null;const e=[];try{for(const i of t){const a=Se(n,i.bot);if(!a)throw new Error(`no client-side driver for ${n}/${i.bot} at seat ${i.seat}`);const r=new vt;try{e.push({seat:i.seat,bot:await a(r,i.opts),host:r})}catch(l){throw r.terminate(),l}}}catch(i){for(const a of e)a.bot.cancel(),a.host.terminate();throw i}const s=new Map(e.map(i=>[i.seat,i.bot])),o=[...new Set(e.map(i=>i.bot.cpuFallback).filter(i=>!!i))];return{cpuFallback:o.length?o.join(" "):void 0,async onMove(i){await Promise.all(e.map(a=>a.bot.onMove(i)))},async chooseMove(i){const a=s.get(i.toAct);if(!a)throw new Error(`no client-side bot configured for seat ${i.toAct}`);return a.chooseMove(i)},async finalResult(){for(const i of e){const a=await i.bot.finalResult?.()??"";if(a)return a}return""},cancel(){for(const i of e)i.bot.cancel(),i.host.terminate()}}}function y(n){return new Promise(t=>setTimeout(t,n))}const cs=["q","r","b","n","p"],ds={q:1,r:2,b:2,n:2,p:8},hs={q:9,r:5,b:3,n:3,p:1},ps=["q","r","b","n"],Lt=/^[a-h][1-8][a-h][1-8][qrbn]?$/,us=240,fs=120,gs=160,bs={p:`<circle class="pcb" cx="22.5" cy="15.5" r="4.5"/>
<path class="pcb" d="M22.5 19.7c-3.3 0-5.4 2.4-5.4 5 0 1.8 0.9 3.3 2.3 4.3-2.9 1.6-4.9 4.2-4.9 6.5h16c0-2.3-2-4.9-4.9-6.5 1.4-1 2.3-2.5 2.3-4.3 0-2.6-2.1-5-5.4-5z"/>
<rect class="pcb" x="14" y="35.5" width="17" height="4.5" rx="2"/>`,n:`<path class="pcb" d="M14.5 35.5c0-7.5 1-11.5 4-14-2.5 0-6-1-7.5-4l0-2c0-1.5 1.5-3.2 3.5-3.5 2-0.4 3.4-2 4-5l2.1 3 2.4-3.5c1 1.5 1.6 2.7 1.6 3.8 4.4 1.7 8.9 6.7 8.9 13.7v11.5z"/>
<circle class="pcf" cx="16.2" cy="15.4" r="1"/>
<rect class="pcb" x="12" y="35.5" width="21.5" height="4.5" rx="2"/>`,b:`<circle class="pcb" cx="22.5" cy="9" r="1.9"/>
<path class="pcb" d="M22.5 11.5c3.4 2.5 5.5 5.6 5.5 8.9 0 2.3-1.1 4.4-2.9 5.7 3 2 5 6 5.4 9.4h-16c0.4-3.4 2.4-7.4 5.4-9.4-1.8-1.3-2.9-3.4-2.9-5.7 0-3.3 2.1-6.4 5.5-8.9z"/>
<path class="pcd" d="M22.5 15v6.4M19.6 18.2h5.8"/>
<rect class="pcb" x="12.5" y="35.5" width="20" height="4.5" rx="2"/>`,r:`<path class="pcb" d="M13.5 35.5v-4l2-2.5v-10l-2-2v-7h4v3h3v-3h4v3h3v-3h4v7l-2 2v10l2 2.5v4z"/>
<rect class="pcb" x="11.5" y="35.5" width="22" height="4.5" rx="2"/>`,q:`<path class="pcb" d="M14 21l-2.5-9.5 5 4.7 1.6-7.7 3 6.6 1.4-8.1 1.4 8.1 3-6.6 1.6 7.7 5-4.7-2.5 9.5c1 3-0.3 5.2-2.1 6.6 2.6 1.9 4.2 4.3 4.5 7.9h-21.8c0.3-3.6 1.9-6 4.5-7.9-1.8-1.4-3.1-3.6-2.1-6.6z"/>
<rect class="pcb" x="11.5" y="35.5" width="22" height="4.5" rx="2"/>`,k:`<path class="pcb" d="M21.3 4h2.4v3h2.9v2.4h-2.9v3h-2.4v-3h-2.9v-2.4h2.9z"/>
<path class="pcb" d="M22.5 12.8c5.3 0 9 3.3 9 7.4 0 2.5-1.4 4.8-3.5 6.2 3.2 2 5.3 5 5.6 9.1h-22.2c0.3-4.1 2.4-7.1 5.6-9.1-2.1-1.4-3.5-3.7-3.5-6.2 0-4.1 3.7-7.4 9-7.4z"/>
<path class="pcd" d="M16.2 21h12.6"/>
<rect class="pcb" x="11" y="35.5" width="23" height="4.5" rx="2"/>`};function it(n,t){const e=bs[n]??"";return`<svg class="chess-pc ${t?"chess-pc-w":"chess-pc-b"}" viewBox="0 0 45 45" aria-hidden="true">${e}</svg>`}function T(n){return(n.charCodeAt(1)-49)*8+(n.charCodeAt(0)-97)}function R(n,t){return n.charAt((7-Math.floor(t/8))*8+t%8)}function Ht(n){if(typeof n!="object"||n===null)return null;const t=n;return typeof t.board!="string"||t.board.length!==64?null:{board:t.board,turn:t.turn==="b"?"b":"w",check:t.check===!0}}function ms(n){if(typeof n!="object"||n===null)return null;const t=n;if(typeof t.from!="string"||typeof t.to!="string"||!Lt.test(t.from+t.to))return null;const e=s=>typeof s=="string"&&s.length===2?s:null;return{from:t.from,to:t.to,capturedSquare:e(t.capturedSquare),castleRookFrom:e(t.castleRookFrom),castleRookTo:e(t.castleRookTo)}}function xs(n){return Lt.test(n)?{from:n.slice(0,2),to:n.slice(2,4),capturedSquare:null,castleRookFrom:null,castleRookTo:null}:null}class ys{ctx;host;rootEl;boardEl;piecesEl;promoEl;bars;squareEls=[];pieceEls=new Map;flipped=!1;view=null;lastMove=null;gameOver=!1;moves=new Map;selected=null;inputArmed=!1;drag=null;skipSlide=!1;promoFromDrag=!1;unsubDebug=null;evalGen=0;mount(t,e){this.ctx=e,this.host=t,this.flipped=e.humanSeat===1,ws();const s=`
      <span class="chess-turn-dot"></span>
      <span class="chess-bar-name"></span>
      <span class="seat-slot"></span>
      <span class="chess-tray"></span>
      <span class="chess-score"></span>`;t.innerHTML=`
      <div class="chess-root">
        <div class="chess-bar chess-bar-top">${s}</div>
        <div class="chess-stage">
          <div class="chess-ranks"></div>
          <div class="chess-board">
            <div class="chess-squares"></div>
            <div class="chess-pieces"></div>
            <div class="chess-promo" hidden></div>
          </div>
          <div class="chess-files"></div>
        </div>
        <div class="chess-bar chess-bar-bottom">${s}</div>
      </div>`,this.rootEl=t.querySelector(".chess-root"),this.boardEl=t.querySelector(".chess-board"),this.piecesEl=t.querySelector(".chess-pieces"),this.promoEl=t.querySelector(".chess-promo");const o=t.querySelector(".chess-bar-top"),i=t.querySelector(".chess-bar-bottom"),a=d=>({root:d,tray:d.querySelector(".chess-tray"),score:d.querySelector(".chess-score")});this.bars=this.flipped?{w:a(o),b:a(i)}:{w:a(i),b:a(o)};for(const d of["w","b"]){const h=this.bars[d].root;h.querySelector(".chess-bar-name").textContent=d==="w"?"White":"Black",h.querySelector(".seat-slot").setAttribute("data-seat",d==="w"?"0":"1")}const r=t.querySelector(".chess-ranks"),l=t.querySelector(".chess-files");for(let d=0;d<8;d++){const h=this.flipped?d+1:8-d,p=this.flipped?7-d:d;r.insertAdjacentHTML("beforeend",`<span>${h}</span>`),l.insertAdjacentHTML("beforeend",`<span>${"abcdefgh"[p]}</span>`)}const c=t.querySelector(".chess-squares");this.squareEls=new Array(64);for(let d=0;d<8;d++)for(let h=0;h<8;h++){const p=this.flipped?7-h:h,g=this.flipped?d:7-d,f=g*8+p,b=document.createElement("div");b.className=`chess-sq ${(p+g)%2===1?"chess-sq-light":"chess-sq-dark"}`,b.dataset.sq=String(f),this.squareEls[f]=b,c.append(b)}this.boardEl.addEventListener("pointerdown",d=>this.onPointerDown(d)),this.boardEl.addEventListener("pointermove",d=>this.onPointerMove(d)),this.boardEl.addEventListener("pointerup",d=>this.onPointerUp(d)),this.boardEl.addEventListener("pointercancel",()=>this.cancelDrag(!0)),this.boardEl.addEventListener("contextmenu",d=>{this.drag&&d.preventDefault()}),this.promoEl.addEventListener("click",d=>{d.target===this.promoEl&&(this.closePromo(),this.select(null))}),this.unsubDebug=e.onDebugChange(d=>{d?this.refreshEval():this.ctx.setDebugReadout([])})}render(t){const e=Ht(t.viewData);e&&(this.view=e,this.gameOver=t.isOver,this.syncAll(),this.refreshEval())}async animate(t,e){const s=this.skipSlide;this.skipSlide=!1,this.disarm();const o=Ht(e.viewData);if(!o)return;const i=ms(t.data)??xs(t.label);i&&(this.lastMove={from:T(i.from),to:T(i.to)}),this.gameOver=e.isOver;const a=this.ctx.animationScale();i&&a>0&&!s&&await this.slide(i,a),this.view=o,this.syncAll(),this.refreshEval(),a>0&&!s&&await y(fs*a)}promptAction(t){if(!(this.ctx.humanSeat<0)){this.moves.clear();for(const e of t){if(!Lt.test(e))continue;const s=T(e.slice(0,2)),o=T(e.slice(2,4));let i=this.moves.get(s);i||(i=new Map,this.moves.set(s,i));const a=i.get(o);a?a.push(e):i.set(o,[e])}this.inputArmed=!0,this.rootEl.classList.add("chess-armed");for(const e of this.moves.keys())this.squareEls[e].classList.add("chess-sq-movable")}}refreshEval(){if(!this.ctx.debug())return;const t=++this.evalGen;Ye().then(e=>{if(t!==this.evalGen||!e){t===this.evalGen&&this.ctx.setDebugReadout([]);return}const s=Math.round(e.value*100);this.ctx.setDebugReadout([`Win: ${s}% (White)`])}).catch(()=>{t===this.evalGen&&this.ctx.setDebugReadout([])})}unmount(){this.unsubDebug?.(),this.unsubDebug=null,this.host.replaceChildren()}squareAt(t,e){const s=this.boardEl.getBoundingClientRect();if(t<s.left||t>=s.right||e<s.top||e>=s.bottom)return null;const o=Math.floor((t-s.left)/s.width*8),i=Math.floor((e-s.top)/s.height*8),a=this.flipped?7-o:o;return(this.flipped?i:7-i)*8+a}onPointerDown(t){if(!this.inputArmed||!this.promoEl.hidden||this.drag||t.pointerType==="mouse"&&t.button!==0)return;const e=this.squareAt(t.clientX,t.clientY);if(e===null)return;if(this.selected!==null){const a=this.moves.get(this.selected)?.get(e);if(a){a.length>1?this.openPromo(a):this.submitMove(a[0]);return}}if(!this.moves.has(e)){this.select(null);return}const s=this.pieceEls.get(e);if(!s)return;t.preventDefault();const o=this.selected===e;this.select(e);const i=s.cloneNode(!0);i.classList.add("chess-piece-ghost"),this.piecesEl.append(i),s.classList.add("chess-piece-drag"),this.boardEl.classList.add("chess-dragging"),this.drag={pointerId:t.pointerId,from:e,el:s,ghost:i,wasSelected:o,hover:null},this.boardEl.setPointerCapture(t.pointerId),this.moveDragTo(t.clientX,t.clientY)}onPointerMove(t){!this.drag||t.pointerId!==this.drag.pointerId||this.moveDragTo(t.clientX,t.clientY)}moveDragTo(t,e){if(!this.drag)return;const s=this.boardEl.getBoundingClientRect(),o=s.width/8,i=t-s.left-o/2,a=e-s.top-o/2;this.drag.el.style.transform=`translate(${i}px, ${a}px) scale(1.15)`;const r=this.squareAt(t,e);r!==this.drag.hover&&(this.drag.hover!==null&&this.squareEls[this.drag.hover].classList.remove("chess-sq-drop"),this.drag.hover=null,r!==null&&r!==this.drag.from&&this.moves.get(this.drag.from)?.has(r)&&(this.squareEls[r].classList.add("chess-sq-drop"),this.drag.hover=r))}onPointerUp(t){if(!this.drag||t.pointerId!==this.drag.pointerId)return;const e=this.drag;this.drag=null,this.endDragVisuals(e);const s=this.squareAt(t.clientX,t.clientY),o=s!==null&&s!==e.from?this.moves.get(e.from)?.get(s):void 0;if(s!==null&&o){this.settle(e.el,s),this.removeVictim(e.from,s),o.length>1?(this.promoFromDrag=!0,this.openPromo(o)):this.submitMove(o[0],!0);return}this.snapBack(e.el,e.from),s===e.from&&e.wasSelected&&this.select(null)}cancelDrag(t){if(!this.drag)return;const e=this.drag;this.drag=null,this.endDragVisuals(e),t?this.snapBack(e.el,e.from):(e.el.classList.remove("chess-piece-drag"),this.place(e.el,e.from))}endDragVisuals(t){t.hover!==null&&this.squareEls[t.hover].classList.remove("chess-sq-drop"),t.ghost.remove(),this.boardEl.classList.remove("chess-dragging")}settle(t,e){t.classList.remove("chess-piece-drag"),t.style.zIndex="5",this.place(t,e)}snapBack(t,e){t.classList.remove("chess-piece-drag");const s=gs*this.ctx.animationScale();if(s<=0){this.place(t,e);return}t.style.zIndex="5",t.style.transitionDuration=`${s}ms`,t.offsetWidth,this.place(t,e),window.setTimeout(()=>{t.style.transitionDuration="",t.style.zIndex=""},s+30)}removeVictim(t,e){const s=this.pieceEls.get(e);if(s){s.remove(),this.pieceEls.delete(e);return}if(!this.view||R(this.view.board,t).toLowerCase()!=="p"||t%8===e%8)return;const i=Math.floor(t/8)*8+e%8,a=this.pieceEls.get(i);a&&(a.remove(),this.pieceEls.delete(i))}select(t){if(this.clearSelection(),t===null)return;const e=this.moves.get(t);if(e){this.selected=t,this.squareEls[t].classList.add("chess-sq-selected");for(const s of e.keys())this.squareEls[s].classList.add(this.pieceEls.has(s)?"chess-sq-capture":"chess-sq-target")}}submitMove(t,e=!1){this.skipSlide=e,this.promoFromDrag=!1,this.disarm(),this.ctx.submit(t)}openPromo(t){const e=T(t[0].slice(0,2)),s=this.view?R(this.view.board,e):"P",o=s===s.toUpperCase(),i=this.promoFromDrag,a=document.createElement("div");a.className="chess-promo-panel";for(const r of ps){const l=t.find(d=>d.charAt(4)===r);if(!l)continue;const c=document.createElement("button");c.type="button",c.className="chess-promo-btn",c.innerHTML=it(r,o),c.onclick=()=>this.submitMove(l,i),a.append(c)}this.promoEl.replaceChildren(a),this.promoEl.hidden=!1}closePromo(){const t=this.promoFromDrag;this.promoFromDrag=!1,this.promoEl.hidden=!0,this.promoEl.replaceChildren(),t&&this.view&&this.syncPieces(this.view)}disarm(){this.cancelDrag(!1),this.inputArmed=!1,this.moves.clear(),this.clearSelection(),this.closePromo(),this.rootEl.classList.remove("chess-armed");for(const t of this.squareEls)t.classList.remove("chess-sq-movable")}clearSelection(){this.selected=null;for(const t of this.squareEls)t.classList.remove("chess-sq-selected","chess-sq-target","chess-sq-capture","chess-sq-drop")}syncAll(){this.view&&(this.clearSelection(),this.syncPieces(this.view),this.syncHighlights(this.view),this.syncBars(this.view))}syncPieces(t){this.pieceEls.clear();const e=document.createDocumentFragment();for(let s=0;s<64;s++){const o=R(t.board,s);if(o===".")continue;const i=document.createElement("div");i.className="chess-piece",i.innerHTML=it(o.toLowerCase(),o===o.toUpperCase()),this.place(i,s),this.pieceEls.set(s,i),e.append(i)}this.piecesEl.replaceChildren(e)}syncHighlights(t){for(const e of this.squareEls)e.classList.remove("chess-sq-last","chess-sq-check","chess-sq-mate");if(this.lastMove&&(this.squareEls[this.lastMove.from].classList.add("chess-sq-last"),this.squareEls[this.lastMove.to].classList.add("chess-sq-last")),t.check){const e=t.turn==="w"?"K":"k";for(let s=0;s<64;s++)R(t.board,s)===e&&(this.squareEls[s].classList.add("chess-sq-check"),this.gameOver&&this.squareEls[s].classList.add("chess-sq-mate"))}}syncBars(t){const e={};for(const a of t.board)a!=="."&&(e[a]=(e[a]??0)+1);const s=a=>{const r=[];let l=0;for(const c of cs){const d=e[a==="w"?c.toUpperCase():c]??0,h=Math.max(0,(ds[c]??0)-d);l+=h*(hs[c]??0);for(let p=0;p<h;p++)r.push(c)}return{pieces:r,pts:l}},o=s("w"),i=s("b");for(const a of["w","b"]){const r=a==="w"?i:o,l=a==="w"?i.pts-o.pts:o.pts-i.pts,c=this.bars[a];c.tray.replaceChildren(...r.pieces.map(d=>{const h=document.createElement("span");return h.className="chess-tray-piece",h.innerHTML=it(d,a==="b"),h})),c.score.textContent=l>0?`+${l}`:"",c.root.classList.toggle("chess-bar-active",!this.gameOver&&t.turn===a)}}place(t,e){const s=this.flipped?7-e%8:e%8,o=this.flipped?Math.floor(e/8):7-Math.floor(e/8);t.style.transform=`translate(${s*100}%, ${o*100}%)`}async slide(t,e){const s=us*e,o=T(t.from),i=T(t.to),a=this.pieceEls.get(o);if(!a)return;const r=t.capturedSquare!==null?T(t.capturedSquare):this.pieceEls.has(i)?i:null;if(r!==null&&r!==o){const c=this.pieceEls.get(r);c&&(c.style.transition=`opacity ${s}ms ease`,c.style.opacity="0")}const l=(c,d)=>{c.style.zIndex="3",c.style.transitionDuration=`${s}ms`,c.offsetWidth,this.place(c,d)};if(l(a,i),t.castleRookFrom!==null&&t.castleRookTo!==null){const c=this.pieceEls.get(T(t.castleRookFrom));c&&l(c,T(t.castleRookTo))}await y(s+30)}}function vs(){return new ys}const _t="chess-frontend-style";function ws(){if(document.getElementById(_t))return;const n=document.createElement("style");n.id=_t,n.textContent=ks,document.head.append(n)}const ks=`
.chess-root {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin: auto;
  width: min(100%, var(--board-fit));
}

.chess-bar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px 10px;
  min-height: 38px;
  padding: 6px 12px;
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  transition: border-color 0.2s ease;
}

.chess-bar-active {
  border-color: var(--accent);
}

.chess-turn-dot {
  flex: none;
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--border);
  transition: background 0.2s ease, box-shadow 0.2s ease;
}

.chess-bar-active .chess-turn-dot {
  background: var(--accent);
  box-shadow: 0 0 8px var(--accent);
}

.chess-bar-name {
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--text);
  white-space: nowrap;
}

.chess-tray {
  flex: 1;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0;
  min-height: 18px;
}

.chess-tray-piece {
  width: 17px;
  height: 17px;
  margin-left: -3px;
}

.chess-tray-piece:first-child {
  margin-left: 0;
}

.chess-score {
  color: var(--good);
  font-size: 0.85rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.chess-stage {
  display: grid;
  grid-template-areas: 'ranks board' '. files';
  grid-template-columns: auto minmax(0, 1fr);
  grid-template-rows: auto auto;
}

.chess-ranks {
  grid-area: ranks;
  display: flex;
  flex-direction: column;
  padding-right: 7px;
}

.chess-files {
  grid-area: files;
  display: flex;
  padding-top: 5px;
}

.chess-ranks span,
.chess-files span {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--mono);
  font-size: 0.62rem;
  letter-spacing: 0.05em;
  color: var(--text-dim);
  opacity: 0.8;
}

.chess-board {
  grid-area: board;
  position: relative;
  aspect-ratio: 1 / 1;
  border: 1px solid #30331f;
  border-radius: 2px;
  overflow: hidden;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.22);
  user-select: none;
  -webkit-user-select: none;
  touch-action: none;
}

.dark .chess-board {
  box-shadow: 0 1px 0 rgba(244, 238, 218, 0.05), 0 14px 30px rgba(5, 8, 3, 0.45);
}

.chess-squares {
  position: absolute;
  inset: 0;
  display: grid;
  grid-template-columns: repeat(8, 1fr);
  grid-template-rows: repeat(8, 1fr);
}

.chess-sq {
  position: relative;
}

.chess-sq-light {
  background: #e9ddbd;
}

.chess-sq-dark {
  background: #6f8a5d;
}

.chess-armed .chess-sq-movable {
  cursor: grab;
}

.chess-dragging,
.chess-dragging .chess-sq {
  cursor: grabbing;
}

.chess-armed .chess-sq-movable:hover::before {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(212, 169, 92, 0.18);
}

.chess-sq-last::before {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(212, 169, 92, 0.34);
}

.chess-sq-selected::before {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(212, 169, 92, 0.55);
}

.chess-sq-target,
.chess-sq-capture {
  cursor: pointer;
}

.chess-sq-target::after {
  content: '';
  position: absolute;
  inset: 0;
  margin: auto;
  width: 26%;
  height: 26%;
  border-radius: 50%;
  background: rgba(22, 24, 12, 0.32);
}

.chess-sq-capture::after {
  content: '';
  position: absolute;
  inset: 5%;
  border-radius: 50%;
  border: 3px solid rgba(22, 24, 12, 0.38);
}

.chess-sq-drop {
  box-shadow: inset 0 0 0 3px rgba(212, 169, 92, 0.95);
}

.chess-sq-check {
  background-image: radial-gradient(
    circle at 50% 50%,
    rgba(217, 106, 90, 0.62) 22%,
    rgba(217, 106, 90, 0.24) 50%,
    transparent 68%
  );
}

.chess-sq-mate {
  background-image: radial-gradient(
    circle at 50% 50%,
    rgba(217, 106, 90, 0.85) 26%,
    rgba(217, 106, 90, 0.35) 55%,
    transparent 75%
  );
}

.chess-pieces {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.chess-piece {
  position: absolute;
  top: 0;
  left: 0;
  width: 12.5%;
  height: 12.5%;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1;
  will-change: transform;
  transition: transform 0ms cubic-bezier(0.22, 0.85, 0.3, 1);
}

.chess-pc {
  display: block;
}

.chess-piece .chess-pc {
  width: 92%;
  height: 92%;
  filter: drop-shadow(0 2px 2px rgba(15, 14, 6, 0.32));
}

.chess-pc-w .pcb {
  fill: #f5efdc;
  stroke: #3a382c;
}

.chess-pc-w .pcd {
  stroke: #3a382c;
  fill: none;
}

.chess-pc-w .pcf {
  fill: #3a382c;
}

.chess-pc-b .pcb {
  fill: #33302a;
  stroke: #e9e2ca;
}

.chess-pc-b .pcd {
  stroke: #e9e2ca;
  fill: none;
}

.chess-pc-b .pcf {
  fill: #e9e2ca;
}

.chess-pc .pcb,
.chess-pc .pcd {
  stroke-width: 1.6;
  stroke-linejoin: round;
  stroke-linecap: round;
}

.chess-piece-drag {
  z-index: 7;
}

.chess-piece-drag .chess-pc {
  filter: drop-shadow(0 9px 12px rgba(10, 10, 4, 0.45));
}

.chess-piece-ghost {
  opacity: 0.35;
}

.chess-piece-ghost .chess-pc {
  filter: none;
}

.chess-tray-piece .chess-pc {
  width: 100%;
  height: 100%;
}

.chess-promo {
  position: absolute;
  inset: 0;
  z-index: 8;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(10, 12, 6, 0.55);
}

.chess-promo[hidden] {
  display: none;
}

.chess-promo-panel {
  display: flex;
  gap: 10px;
  padding: 12px;
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.25);
}

.dark .chess-promo-panel {
  box-shadow: 0 16px 48px rgba(5, 8, 3, 0.6);
}

.chess-promo-btn {
  width: 62px;
  height: 62px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 7px;
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) - 3px);
  cursor: pointer;
  transition: border-color 0.15s ease, transform 0.15s ease;
}

.chess-promo-btn .chess-pc {
  width: 100%;
  height: 100%;
}

.chess-promo-btn:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
}
`,S=7,M=6,Gt=["Red","Yellow"];function Nt(n){if(!n||typeof n!="object")return null;const t=n;return typeof t.cells=="string"&&t.cells.length===S*M?t:null}function Ss(n){if(!n||typeof n!="object")return null;const t=n;return typeof t.col=="number"&&typeof t.row=="number"&&typeof t.player=="number"?t:null}function Ft(n,t){return(M-1-t)*S+n}function Es(n,t){if(!n||!t)return null;for(let e=0;e<S*M;e++)if(n.cells[e]==="."&&t.cells[e]!==".")return{col:e%S,row:M-1-Math.floor(e/S),player:t.cells[e]==="x"?0:1};return null}const Rt="connect4-frontend-style",$s=`
.c4-root {
  align-self: center;
  width: min(100%, var(--board-fit));
  display: flex;
  flex-direction: column;
  gap: 14px;
  user-select: none;
}
.c4-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px 10px;
}
.c4-chip {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  padding: 6px 12px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--bg-inset);
  color: var(--text-dim);
  font-size: 0.88rem;
  white-space: nowrap;
  transition: border-color 0.25s, box-shadow 0.25s, color 0.25s;
}
.c4-chip.c4-active {
  border-color: var(--accent);
  color: var(--text);
  box-shadow: 0 0 12px rgba(88, 166, 255, 0.3);
}
.c4-swatch {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  flex: none;
}
.c4-chip-0 .c4-swatch {
  background: radial-gradient(circle at 35% 30%, #ff8d7e, #e23b3b 60%, #9c1f1f);
}
.c4-chip-1 .c4-swatch {
  background: radial-gradient(circle at 35% 30%, #ffeaa6, #f2c12e 60%, #b8860b);
}
.c4-msg {
  flex: 1;
  text-align: center;
  color: var(--text-dim);
  font-size: 0.92rem;
}
.c4-board {
  position: relative;
  aspect-ratio: ${S} / ${M};
  border-radius: calc(var(--radius) + 4px);
  overflow: hidden;
  background: #0b1020;
  box-shadow: 0 8px 22px rgba(0, 0, 0, 0.25), 0 0 0 2px rgba(10, 24, 64, 0.9);
}
.dark .c4-board {
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45), 0 0 0 2px rgba(10, 24, 64, 0.9);
}
.c4-layer {
  position: absolute;
  inset: 0;
}
.c4-backdrop {
  background:
    radial-gradient(circle closest-side at 50% 44%, #131a26 0 70%, #060910 100%)
    0 0 / calc(100% / ${S}) calc(100% / ${M});
}
.c4-frame {
  pointer-events: none;
  background:
    radial-gradient(circle closest-side at 50% 50%,
      transparent 0 77%,
      rgba(2, 6, 18, 0.65) 78% 84%,
      #2e63e9 85%,
      #1c46ba 99%,
      #1a41ad 100%)
    0 0 / calc(100% / ${S}) calc(100% / ${M});
}
.c4-hits {
  display: flex;
}
.c4-hit {
  flex: 1;
  height: 100%;
}
.c4-hits.c4-live .c4-hit {
  cursor: pointer;
}
.c4-hits.c4-live .c4-hit:hover {
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.02));
}
.c4-disc {
  position: absolute;
  width: calc(100% / ${S});
  height: calc(100% / ${M});
  will-change: transform;
}
.c4-disc::before {
  content: '';
  position: absolute;
  inset: 9%;
  border-radius: 50%;
  box-shadow:
    inset 0 -4px 7px rgba(0, 0, 0, 0.35),
    inset 0 4px 7px rgba(255, 255, 255, 0.16);
  transition: filter 0.35s;
}
.c4-disc::after {
  content: '';
  position: absolute;
  inset: 26%;
  border-radius: 50%;
  border: 2px solid rgba(0, 0, 0, 0.16);
}
.c4-p0::before {
  background: radial-gradient(circle at 35% 30%, #ff8d7e, #e23b3b 55%, #a32222 95%);
}
.c4-p1::before {
  background: radial-gradient(circle at 35% 30%, #ffeaa6, #f4c430 55%, #c2920c 95%);
}
.c4-ghost {
  opacity: 0.38;
}
.c4-dim::before {
  filter: brightness(0.45) saturate(0.6);
}
.c4-win::before {
  animation: c4-pulse 1.1s ease-in-out infinite;
}
@keyframes c4-pulse {
  0%, 100% {
    box-shadow:
      inset 0 -4px 7px rgba(0, 0, 0, 0.35),
      inset 0 4px 7px rgba(255, 255, 255, 0.16);
    filter: brightness(1);
  }
  50% {
    box-shadow:
      inset 0 -4px 7px rgba(0, 0, 0, 0.35),
      inset 0 4px 7px rgba(255, 255, 255, 0.16),
      0 0 18px 5px rgba(255, 255, 255, 0.4);
    filter: brightness(1.4);
  }
}
@media (prefers-reduced-motion: reduce) {
  .c4-win::before {
    animation: none;
    filter: brightness(1.3);
  }
}
.c4-fallback {
  display: none;
  margin: 0;
  font-family: ui-monospace, monospace;
  color: var(--text);
  white-space: pre;
}
.c4-root.c4-text-only .c4-bar,
.c4-root.c4-text-only .c4-board {
  display: none;
}
.c4-root.c4-text-only .c4-fallback {
  display: block;
}
`;function Ms(){if(document.getElementById(Rt))return;const n=document.createElement("style");n.id=Rt,n.textContent=$s,document.head.append(n)}class qs{ctx;rootEl;discsEl;hitsEl;msgEl;fallbackEl;chips=[];discs=new Map;view=null;colToAction=null;ghost=null;anims=new Set;mount(t,e){this.ctx=e,Ms(),t.innerHTML=`
      <div class="c4-root">
        <div class="c4-bar">
          <div class="c4-chip c4-chip-0"><span class="c4-swatch"></span><span class="c4-name"></span><span class="seat-slot" data-seat="0"></span></div>
          <div class="c4-msg"></div>
          <div class="c4-chip c4-chip-1"><span class="c4-swatch"></span><span class="c4-name"></span><span class="seat-slot" data-seat="1"></span></div>
        </div>
        <div class="c4-board">
          <div class="c4-layer c4-backdrop"></div>
          <div class="c4-layer c4-discs"></div>
          <div class="c4-layer c4-frame"></div>
          <div class="c4-layer c4-hits"></div>
        </div>
        <pre class="c4-fallback"></pre>
      </div>`,this.rootEl=t.querySelector(".c4-root"),this.discsEl=t.querySelector(".c4-discs"),this.hitsEl=t.querySelector(".c4-hits"),this.msgEl=t.querySelector(".c4-msg"),this.fallbackEl=t.querySelector(".c4-fallback"),this.chips=[t.querySelector(".c4-chip-0"),t.querySelector(".c4-chip-1")];for(let s=0;s<2;s++)this.chips[s].querySelector(".c4-name").textContent=Gt[s];for(let s=0;s<S;s++){const o=document.createElement("div");o.className="c4-hit",o.addEventListener("pointerenter",()=>this.showGhost(s)),o.addEventListener("pointerleave",()=>this.hideGhost()),o.addEventListener("click",()=>this.clickColumn(s)),this.hitsEl.append(o)}}render(t){this.disableInput();const e=Nt(t.viewData);if(this.view=e,!e){this.rootEl.classList.add("c4-text-only"),this.fallbackEl.textContent=t.view;return}this.rootEl.classList.remove("c4-text-only"),this.rebuildDiscs(e),this.decorateWin(e,!0);for(let s=0;s<2;s++)this.chips[s].classList.toggle("c4-active",!t.isOver&&e.turn===s);this.msgEl.textContent=t.isOver?e.winner!==null?`${Gt[e.winner]} connects four!`:"Draw — board full":""}async animate(t,e){const s=this.view,o=Nt(e.viewData),i=Ss(t.data)??Es(s,o);this.render(e);const a=this.ctx.animationScale();if(!o||!i||a<=0)return;const r=this.discs.get(Ft(i.col,i.row));if(!r)return;const l=o.winLine!==null;l&&this.decorateWin(o,!1);const c=M-i.row,d=(150+100*Math.sqrt(c))*a;await this.run(r.animate([{transform:`translateY(${-c*100-30}%)`,offset:0,easing:"cubic-bezier(0.5, 0, 0.9, 0.6)"},{transform:"translateY(0%)",offset:.62,easing:"cubic-bezier(0.1, 0.5, 0.5, 1)"},{transform:"translateY(-17%)",offset:.8,easing:"cubic-bezier(0.5, 0, 0.9, 0.6)"},{transform:"translateY(0%)",offset:1}],{duration:d/.62})),l&&(this.decorateWin(o,!0),await y(650*a))}promptAction(t){const e=new Map;t.forEach((s,o)=>{const i=/(\d+)/.exec(s);i&&e.set(Number(i[1])-1,o)}),this.colToAction=e,this.hitsEl.classList.add("c4-live")}unmount(){for(const t of this.anims)t.cancel();this.anims.clear()}rebuildDiscs(t){this.discsEl.replaceChildren(),this.discs.clear(),this.ghost=null;for(let e=0;e<S*M;e++){const s=t.cells[e];if(s===".")continue;const o=this.makeDisc(s==="x"?0:1,e%S,Math.floor(e/S));this.discs.set(e,o),this.discsEl.append(o)}}makeDisc(t,e,s){const o=document.createElement("div");return o.className=`c4-disc c4-p${t}`,o.style.left=`${e*100/S}%`,o.style.top=`${s*100/M}%`,o}decorateWin(t,e){if(!t.winLine)return;const s=new Set(t.winLine);for(const[o,i]of this.discs)i.classList.toggle("c4-win",e&&s.has(o)),i.classList.toggle("c4-dim",e&&!s.has(o))}showGhost(t){this.hideGhost();const e=this.view;if(!(!e||!this.colToAction?.has(t)||this.ctx.humanSeat<0))for(let s=0;s<M;s++){const o=Ft(t,s);if(e.cells[o]==="."){this.ghost=this.makeDisc(this.ctx.humanSeat,t,Math.floor(o/S)),this.ghost.classList.add("c4-ghost"),this.discsEl.append(this.ghost);return}}}hideGhost(){this.ghost?.remove(),this.ghost=null}clickColumn(t){const e=this.colToAction?.get(t);e!==void 0&&(this.disableInput(),this.ctx.submit(String(e)))}disableInput(){this.colToAction=null,this.hideGhost(),this.hitsEl.classList.remove("c4-live")}async run(t){this.anims.add(t);try{await t.finished}catch{}finally{this.anims.delete(t)}}}function Ls(){return new qs}class Cs{ctx;viewEl;actionsEl;mount(t,e){this.ctx=e,t.innerHTML=`
      <div class="generic">
        <pre class="generic-view"></pre>
        <div class="generic-actions"></div>
      </div>`,this.viewEl=t.querySelector(".generic-view"),this.actionsEl=t.querySelector(".generic-actions")}render(t){this.viewEl.textContent=t.view,t.toAct!==t.humanSeat&&this.actionsEl.replaceChildren()}async animate(t,e){this.render(e),await y(250*this.ctx.animationScale())}promptAction(t){const e=t.map((s,o)=>{const i=document.createElement("button");return i.className="action-btn",i.textContent=s,i.onclick=()=>this.ctx.submit(String(o)),i});this.actionsEl.replaceChildren(...e)}unmount(){}}function It(n){if(!n||typeof n!="object")return null;const t=n;return typeof t.size=="number"&&typeof t.cells=="string"&&t.cells.length===t.size*t.size&&Array.isArray(t.captures)?t:null}const jt="go-frontend-style",L=1;function Ee(n){return String.fromCharCode(97+n+(n>=8?1:0))}function As(n,t){return`${Ee(n%t)}${Math.floor(n/t)+1}`}function Ts(n,t){const e=n.charCodeAt(0)-97;if(e<0||e>25||n[0]==="i")return null;const s=e>8?e-1:e,o=parseInt(n.slice(1),10);return!Number.isFinite(o)||s>=t||o<1||o>t?null:(o-1)*t+s}function zs(n){const t=[],e=L+n-1;for(let s=0;s<n;s++){const o=L+s;t.push(`M ${o} ${L} L ${o} ${e}`,`M ${L} ${o} L ${e} ${o}`)}return t.join(" ")}function Ds(n){const t=[],e=n>=13?3:2;if(n>=7)for(const s of[e,n-1-e])for(const o of[e,n-1-e])t.push(s*n+o);if(n%2===1&&n>=5){const s=(n-1)/2;t.push(s*n+s),n>=15&&(t.push(e*n+s,(n-1-e)*n+s),t.push(s*n+e,s*n+(n-1-e)))}return t}const Ps=`
.go { display: flex; flex-direction: column; gap: 14px; width: min(100%, var(--board-fit)); margin: 0 auto; }
.go-hud { display: grid; grid-template-columns: 1fr auto 1fr; align-items: stretch; gap: 10px; }
.go-player { display: flex; align-items: center; gap: 10px; padding: 8px 12px; min-width: 0;
  border-radius: var(--radius); background: var(--bg-raised); border: 1px solid var(--border);
  transition: border-color .2s, box-shadow .2s; }
.go-player.go-active { border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent), 0 0 18px rgba(88, 166, 255, .22); }
.go-stone-icon { width: 22px; height: 22px; border-radius: 50%; flex: none;
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, .22), 0 1px 3px rgba(0, 0, 0, .45); }
.go-stone-icon-b { background: radial-gradient(circle at 35% 30%, #7c8088, #33343a 42%, #0a0a0d); }
.go-stone-icon-w { background: radial-gradient(circle at 35% 30%, #ffffff, #f0eee4 60%, #c4c0ae); }
.go-pinfo { display: flex; flex-direction: column; min-width: 0; }
.go-pname { font-weight: 600; line-height: 1.2; }
.go-psub { font-size: 12px; color: var(--text-dim); white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.go-pcaps { margin-left: auto; text-align: right; font-size: 11px; color: var(--text-dim);
  line-height: 1.25; white-space: nowrap; }
.go-pcaps b { display: block; font-size: 16px; color: var(--text); }
.go-turn-chip { align-self: center; display: flex; align-items: center; gap: 8px; padding: 7px 14px;
  border-radius: 999px; background: var(--bg-inset); border: 1px solid var(--border);
  font-size: 13px; color: var(--text-dim); white-space: nowrap; }
.go-turn-chip:has(.go-turn-text:empty) { padding: 7px 11px; }
.go-turn-dot { width: 13px; height: 13px; border-radius: 50%; flex: none;
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, .25), 0 1px 2px rgba(0, 0, 0, .4); }
.go-board-wrap { position: relative; width: 100%; margin: 0 auto; }
.go-svg { display: block; width: 100%; height: auto; border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, .22), 0 2px 6px rgba(0, 0, 0, .16); }
.dark .go-svg { box-shadow: 0 14px 40px rgba(0, 0, 0, .5), 0 2px 8px rgba(0, 0, 0, .4); }
.go-hit { fill: transparent; }
.go-hit-on { cursor: pointer; }
.go-ghost, .go-marker { pointer-events: none; }
.go-drop { transform-box: fill-box; transform-origin: center;
  animation: go-drop .28s cubic-bezier(.2, .85, .35, 1.25) backwards; }
@keyframes go-drop {
  from { transform: scale(.45) translateY(-22%); opacity: 0; }
  70% { opacity: 1; }
  to { transform: none; opacity: 1; }
}
.go-die { transform-box: fill-box; transform-origin: center;
  animation: go-die .34s ease-in forwards; }
@keyframes go-die {
  to { transform: scale(.65) translateY(-40%); opacity: 0; }
}
.go-controls { display: flex; justify-content: center; min-height: 42px; }
.go-pass { padding: 9px 28px; border-radius: 999px; border: 1px solid var(--border);
  background: var(--bg-raised); color: var(--text); font-weight: 600; letter-spacing: .05em;
  transition: border-color .15s, filter .15s; }
.go-pass:not(:disabled):hover { border-color: var(--accent); filter: brightness(1.18); }
.go-pass:disabled { opacity: .35; cursor: default; }
.go-toast { position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
  background: rgba(1, 4, 9, .8); border: 1px solid rgba(230, 237, 243, .2); color: #e6edf3;
  padding: 6px 16px; border-radius: 999px; font-size: 13px; white-space: nowrap;
  opacity: 0; pointer-events: none; transition: opacity .2s; }
.go-toast-show { opacity: 1; }
@media (max-width: 560px) {
  .go-hud { grid-template-columns: 1fr 1fr; }
  .go-turn-chip { order: 3; grid-column: 1 / -1; justify-self: center; }
}
`;function Bs(){if(document.getElementById(jt))return;const n=document.createElement("style");n.id=jt,n.textContent=Ps,document.head.append(n)}class Os{ctx;svg;stonesG;fxG;ghostEl;markerEl;toastEl;passBtn;turnChip;plaques=[];capEls=[];size=0;view=null;lastMove=null;interactive=!1;labelIndex=new Map;legalPoints=new Set;stoneEls=new Map;unsubDebug=null;evalGen=0;mount(t,e){this.ctx=e,Bs(),t.innerHTML=`
      <div class="go">
        <div class="go-hud">
          <div class="go-player" data-seat="0">
            <span class="go-stone-icon go-stone-icon-b"></span>
            <span class="go-pinfo"><span class="go-pname">Black</span><span class="seat-slot" data-seat="0"></span></span>
            <span class="go-pcaps"><b>0</b>captures</span>
          </div>
          <div class="go-turn-chip"><span class="go-turn-dot"></span><span class="go-turn-text"></span></div>
          <div class="go-player" data-seat="1">
            <span class="go-stone-icon go-stone-icon-w"></span>
            <span class="go-pinfo"><span class="go-pname">White</span><span class="seat-slot" data-seat="1"></span></span>
            <span class="go-pcaps"><b>0</b>captures</span>
          </div>
        </div>
        <div class="go-board-wrap">
          <svg class="go-svg" role="img" aria-label="Go board"></svg>
          <div class="go-toast"></div>
        </div>
        <div class="go-controls">
          <button type="button" class="go-pass" disabled>Pass</button>
        </div>
      </div>`,this.svg=t.querySelector(".go-svg"),this.toastEl=t.querySelector(".go-toast"),this.passBtn=t.querySelector(".go-pass"),this.turnChip=t.querySelector(".go-turn-chip"),this.plaques=[...t.querySelectorAll(".go-player")],this.capEls=this.plaques.map(s=>s.querySelector(".go-pcaps b")),e.humanSeat<0&&(this.passBtn.style.display="none"),this.passBtn.onclick=()=>{const s=this.labelIndex.get("pass");!this.interactive||s===void 0||(this.setInteractive(!1),this.ctx.submit(String(s)))},this.unsubDebug=e.onDebugChange(s=>{s?this.refreshEval():this.ctx.setDebugReadout([])})}xy(t){return{x:L+t%this.size,y:L+(this.size-1-Math.floor(t/this.size))}}buildBoard(t){this.size=t;const e=t-1+2*L;this.svg.setAttribute("viewBox",`0 0 ${e} ${e}`);const s=Ds(t).map(l=>{const{x:c,y:d}=this.xy(l);return`<circle cx="${c}" cy="${d}" r="${t>13?.08:.1}" fill="rgba(40,24,8,.78)"/>`}).join(""),o=[];for(let l=0;l<t;l++)o.push(`<text x="${L+l}" y="${L+t-1+.72}">${Ee(l)}</text>`,`<text x="${L-.66}" y="${L+(t-1-l)+.11}">${l+1}</text>`);const i=[];for(let l=0;l<t*t;l++){const{x:c,y:d}=this.xy(l);i.push(`<rect class="go-hit" data-p="${l}" x="${c-.5}" y="${d-.5}" width="1" height="1"/>`)}this.svg.innerHTML=`
      <defs>
        <linearGradient id="go-wood" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#e8bd7a"/>
          <stop offset="0.35" stop-color="#d9a85f"/>
          <stop offset="0.65" stop-color="#cf9a4f"/>
          <stop offset="1" stop-color="#bf8943"/>
        </linearGradient>
        <radialGradient id="go-sheen" cx="0.5" cy="0.2" r="1.1">
          <stop offset="0" stop-color="rgba(255,244,214,.5)"/>
          <stop offset="0.55" stop-color="rgba(255,244,214,0)"/>
          <stop offset="1" stop-color="rgba(60,30,5,.28)"/>
        </radialGradient>
        <radialGradient id="go-stone-b" cx="0.36" cy="0.3" r="0.95">
          <stop offset="0" stop-color="#7c8088"/>
          <stop offset="0.4" stop-color="#33343a"/>
          <stop offset="1" stop-color="#0a0a0d"/>
        </radialGradient>
        <radialGradient id="go-stone-w" cx="0.36" cy="0.3" r="0.95">
          <stop offset="0" stop-color="#ffffff"/>
          <stop offset="0.6" stop-color="#f0eee4"/>
          <stop offset="1" stop-color="#c4c0ae"/>
        </radialGradient>
        <filter id="go-shadow" x="-30%" y="-30%" width="160%" height="160%">
          <feDropShadow dx="0.015" dy="0.05" stdDeviation="0.045" flood-color="#000" flood-opacity="0.45"/>
        </filter>
      </defs>
      <rect width="${e}" height="${e}" rx="0.32" fill="url(#go-wood)"/>
      <rect width="${e}" height="${e}" rx="0.32" fill="url(#go-sheen)"/>
      <path d="${zs(t)}" stroke="rgba(46,28,8,.8)" stroke-width="0.032" fill="none" stroke-linecap="square"/>
      ${s}
      <g fill="rgba(46,28,8,.55)" font-size="0.32" text-anchor="middle" font-family="inherit">${o.join("")}</g>
      <g class="go-stones" filter="url(#go-shadow)"></g>
      <g class="go-fx"></g>
      <circle class="go-marker" r="0.17" fill="none" stroke-width="0.07" opacity="0"/>
      <circle class="go-ghost" r="0.45" opacity="0"/>
      <g class="go-hits"></g>`,this.stonesG=this.svg.querySelector(".go-stones"),this.fxG=this.svg.querySelector(".go-fx"),this.markerEl=this.svg.querySelector(".go-marker"),this.ghostEl=this.svg.querySelector(".go-ghost");const a=this.svg.querySelector(".go-hits");a.innerHTML=i.join("");const r=l=>{const c=l.target.getAttribute?.("data-p");return c==null?null:Number(c)};a.addEventListener("click",l=>{const c=r(l);c!==null&&this.tryPlay(c)}),a.addEventListener("pointerover",l=>this.showGhost(r(l))),a.addEventListener("pointerout",()=>this.showGhost(null))}tryPlay(t){if(!this.interactive||!this.legalPoints.has(t))return;const e=this.labelIndex.get(As(t,this.size));e!==void 0&&(this.setInteractive(!1),this.ctx.submit(String(e)))}showGhost(t){if(t===null||!this.interactive||!this.legalPoints.has(t)||this.view?.cells[t]!=="."){this.ghostEl.setAttribute("opacity","0");return}const{x:e,y:s}=this.xy(t);this.ghostEl.setAttribute("cx",String(e)),this.ghostEl.setAttribute("cy",String(s)),this.ghostEl.setAttribute("fill",this.ctx.humanSeat===1?"rgba(250,248,238,.62)":"rgba(12,12,16,.55)"),this.ghostEl.setAttribute("opacity","1")}setInteractive(t){this.interactive=t,this.passBtn.disabled=!t||!this.labelIndex.has("pass"),t||this.ghostEl.setAttribute("opacity","0"),this.svg.querySelectorAll(".go-hit").forEach(e=>e.classList.toggle("go-hit-on",t&&this.legalPoints.has(Number(e.getAttribute("data-p")))))}drawStones(t){this.stoneEls.clear(),this.stonesG.replaceChildren();for(let e=0;e<t.cells.length;e++){const s=t.cells[e];s!=="b"&&s!=="w"||this.stonesG.append(this.makeStone(e,s==="b"?0:1))}if(this.lastMove!==null&&t.cells[this.lastMove]!=="."){const{x:e,y:s}=this.xy(this.lastMove);this.markerEl.setAttribute("cx",String(e)),this.markerEl.setAttribute("cy",String(s)),this.markerEl.setAttribute("stroke",t.cells[this.lastMove]==="b"?"#f2f0e4":"#1c1c20"),this.markerEl.setAttribute("opacity","1")}else this.markerEl.setAttribute("opacity","0")}makeStone(t,e){const{x:s,y:o}=this.xy(t),i=document.createElementNS("http://www.w3.org/2000/svg","circle");return i.setAttribute("cx",String(s)),i.setAttribute("cy",String(o)),i.setAttribute("r","0.47"),i.setAttribute("fill",e===0?"url(#go-stone-b)":"url(#go-stone-w)"),this.stoneEls.set(t,i),i}render(t){const e=It(t.viewData);if(!e)return;e.size!==this.size&&this.buildBoard(e.size),this.view=e,this.drawStones(e),this.capEls[0].textContent=String(e.captures[0]),this.capEls[1].textContent=String(e.captures[1]);const s=this.turnChip.querySelector(".go-turn-dot"),o=this.turnChip.querySelector(".go-turn-text");t.isOver?(o.textContent="Game over",s.style.display="none",this.plaques.forEach(i=>i.classList.remove("go-active"))):(o.textContent="",s.style.display="",s.style.background=e.turn===0?"radial-gradient(circle at 35% 30%, #7c8088, #0a0a0d)":"radial-gradient(circle at 35% 30%, #ffffff, #c4c0ae)",this.plaques.forEach((i,a)=>i.classList.toggle("go-active",a===e.turn))),t.toAct!==t.humanSeat&&this.setInteractive(!1),this.refreshEval()}async animate(t,e){const s=t.data??null,o=this.ctx.animationScale(),i=It(e.viewData);if(i&&i.size!==this.size&&this.buildBoard(i.size),s&&typeof s.point=="number"){if(this.lastMove=s.point,this.render(e),o>0){const a=this.stoneEls.get(s.point);a&&(a.style.animationDuration=`${280*o}ms`,a.classList.add("go-drop"));const r=s.captured??[];for(const l of r){const c=this.makeStone(l,s.seat^1);this.stoneEls.delete(l),c.style.animationDuration=`${340*o}ms`,c.style.animationDelay=`${120*o}ms`,c.classList.add("go-die"),this.fxG.append(c)}await y((r.length>0?500:300)*o),this.fxG.replaceChildren()}}else s&&s.move==="pass"?(this.lastMove=null,this.render(e),o>0&&(this.toastEl.textContent=`${s.seat===0?"Black":"White"} passes`,this.toastEl.classList.add("go-toast-show"),await y(650*o),this.toastEl.classList.remove("go-toast-show"))):(this.render(e),await y(200*o))}refreshEval(){if(!this.ctx.debug())return;const t=this.view?.komi??7.5,e=++this.evalGen;Qe().then(s=>{if(e!==this.evalGen)return;if(!s){this.ctx.setDebugReadout([`Komi: ${t}`]);return}const o=s.scoreLead,i=o>=0?`B+${o.toFixed(1)}`:`W+${(-o).toFixed(1)}`,a=Math.round(s.value*100);this.ctx.setDebugReadout([`Score: ${i}`,`Win: ${a}% (Black)`,`Komi: ${t}`])}).catch(()=>{e===this.evalGen&&this.ctx.setDebugReadout([`Komi: ${t}`])})}promptAction(t){this.labelIndex=new Map(t.map((e,s)=>[e,s])),this.legalPoints=new Set(t.map(e=>Ts(e,this.size)).filter(e=>e!==null)),this.setInteractive(!0)}unmount(){this.unsubDebug?.(),this.unsubDebug=null}}function Hs(){return new Os}const Yt="liars-dice-frontend-style",_s=`
.ld-root {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 100%;
  max-width: 920px;
  margin: 0 auto;
  user-select: none;
}

/* ---------- the table ---------- */

.ld-table {
  position: relative;
  width: 100%;
  height: clamp(360px, 56vh, 540px);
}

/* Layered felt: a lamp-light sheen, woven-grain noise, and the green pile
 * sit on the padding box; the mahogany rail is painted on the border box. */
.ld-felt {
  position: absolute;
  inset: 5% 2%;
  border-radius: 50% / 46%;
  border: 10px solid transparent;
  background:
    radial-gradient(ellipse 60% 44% at 50% 30%, rgba(255, 252, 230, 0.07), transparent 70%)
      padding-box,
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url(%23n)' opacity='0.05'/%3E%3C/svg%3E")
      padding-box,
    radial-gradient(ellipse at 50% 38%, #2e5e40 0%, #224830 55%, #142c1d 100%) padding-box,
    linear-gradient(155deg, #7a5530 0%, #46301b 40%, #5d3e22 70%, #33210f 100%) border-box;
  box-shadow:
    inset 0 0 80px rgba(0, 0, 0, 0.55),
    inset 0 3px 8px rgba(0, 0, 0, 0.5),
    0 0 0 1px rgba(0, 0, 0, 0.6),
    0 1px 0 rgba(255, 255, 255, 0.06),
    0 16px 40px rgba(0, 0, 0, 0.55);
}

.ld-felt::after {
  content: '';
  position: absolute;
  inset: 7%;
  border-radius: inherit;
  border: 1px solid rgba(212, 169, 92, 0.16);
}

/* ---------- center: bid, ladder, round ---------- */

.ld-center {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  text-align: center;
  z-index: 2;
  max-width: 46%;
  pointer-events: none;
}

.ld-round {
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: rgba(234, 230, 216, 0.55);
}

.ld-bid-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  min-height: 60px;
}

.ld-bid-main {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 34px;
  font-weight: 700;
  color: #eae6d8;
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.6);
}

.ld-bid-main .ld-die {
  width: 36px;
  height: 36px;
  rotate: -3deg;
}

.ld-x {
  color: #9da28e;
  font-size: 20px;
  font-weight: 600;
}

.ld-tally-n {
  color: #d4a95c;
  min-width: 1.1em;
  text-align: right;
}

.ld-bid-sub {
  font-size: 12px;
  color: rgba(234, 230, 216, 0.65);
}

.ld-open-hint {
  font-size: 14px;
  font-style: italic;
  color: rgba(234, 230, 216, 0.7);
}

.ld-win-text {
  font-size: 24px;
  font-weight: 700;
  color: #d4a95c;
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.6);
}

.ld-ladder {
  display: flex;
  flex-direction: column;
  gap: 2px;
  align-items: center;
}

.ld-rung {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: rgba(234, 230, 216, 0.45);
}

.ld-rung .ld-die {
  width: 13px;
  height: 13px;
}

.ld-rung-now {
  color: #eae6d8;
  font-weight: 600;
}

/* ---------- seats ---------- */

.ld-seats {
  position: absolute;
  inset: 0;
}

.ld-seat {
  position: absolute;
  transform: translate(-50%, -50%);
  z-index: 3;
  transition: opacity 0.4s;
}

.ld-pod {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  min-width: 108px;
  padding: 8px 12px;
  background: rgba(10, 13, 9, 0.6);
  border: 1px solid #2d352c;
  border-radius: 14px;
  backdrop-filter: blur(3px);
  transition: box-shadow 0.3s, border-color 0.3s;
}

.ld-turn .ld-pod {
  border-color: #d4a95c;
  animation: ld-glow 1.6s ease-in-out infinite;
}

@keyframes ld-glow {
  0%, 100% { box-shadow: 0 0 0 1px #d4a95c, 0 0 12px rgba(212, 169, 92, 0.25); }
  50% { box-shadow: 0 0 0 1px #d4a95c, 0 0 24px rgba(212, 169, 92, 0.55); }
}

.ld-out {
  opacity: 0.35;
  filter: grayscale(0.9);
}

.ld-out-mark {
  font-size: 20px;
  line-height: 30px;
  color: #9da28e;
}

.ld-out-tag {
  font-size: 10px;
  letter-spacing: 1px;
  color: #d96a5a;
  font-weight: 700;
}

.ld-name {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-weight: 600;
  color: #eae6d8;
  white-space: nowrap;
}

.ld-tag {
  font-size: 10px;
  font-weight: 400;
  color: #9da28e;
}

.ld-crown {
  color: #d4a95c;
}

.ld-hand {
  display: flex;
  gap: 4px;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  max-width: 144px;
  min-height: 34px;
}

.ld-bubble {
  position: absolute;
  top: -13px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 4px;
  background: #0b0e0a;
  border: 1px solid #d4a95c;
  border-radius: 999px;
  padding: 2px 9px;
  font-size: 11px;
  font-weight: 700;
  color: #eae6d8;
  white-space: nowrap;
  z-index: 2;
}

.ld-bubble .ld-die {
  width: 14px;
  height: 14px;
}

/* ---------- dice ---------- */

/* Pips are absolutely positioned with percentage offsets, which resolve
 * against the die's own box — the pattern stays correct at every size the
 * die is used (hand, bid, ladder, buttons, fly chip). */
.ld-die {
  position: relative;
  width: 24px;
  height: 24px;
  border-radius: 20%;
  background: linear-gradient(145deg, #f9f4e2 0%, #ece4ca 55%, #d6cbab 100%);
  box-shadow:
    inset 0 1px 1px rgba(255, 255, 255, 0.75),
    inset 0 -2px 3px rgba(94, 78, 48, 0.3),
    0 2px 4px rgba(0, 0, 0, 0.45);
  flex: none;
}

.ld-die i {
  position: absolute;
  width: 22%;
  height: 22%;
  border-radius: 50%;
  background: radial-gradient(circle at 36% 30%, #51463a, #221b12 75%);
  transform: translate(-50%, -50%);
}

.ld-pip-nw { left: 26%; top: 26%; }
.ld-pip-n  { left: 50%; top: 26%; }
.ld-pip-ne { left: 74%; top: 26%; }
.ld-pip-w  { left: 26%; top: 50%; }
.ld-pip-c  { left: 50%; top: 50%; }
.ld-pip-e  { left: 74%; top: 50%; }
.ld-pip-sw { left: 26%; top: 74%; }
.ld-pip-s  { left: 50%; top: 74%; }
.ld-pip-se { left: 74%; top: 74%; }

.ld-die-num {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 70%;
  font-weight: 800;
  color: #221b12;
}

/* A thrown hand, not a stamped row: each die settles at its own angle. */
.ld-hand .ld-die:nth-child(2n) { rotate: 2.5deg; }
.ld-hand .ld-die:nth-child(3n) { rotate: -2deg; }
.ld-hand .ld-die:nth-child(4n + 1) { rotate: -1.4deg; }
.ld-hand .ld-die:nth-child(5n + 2) { rotate: 1.8deg; }

.ld-die.ld-hit {
  background: linear-gradient(145deg, #ffedb9, #e9cb74);
  box-shadow: 0 0 0 2px #d4a95c, 0 0 10px rgba(212, 169, 92, 0.6);
}

.ld-flip {
  animation: ld-flip 0.35s ease;
}

@keyframes ld-flip {
  from { transform: rotateX(90deg) scale(0.6); opacity: 0; }
  to { transform: rotateX(0) scale(1); opacity: 1; }
}

/* ---------- cups ---------- */

.ld-cup {
  position: relative;
  width: 36px;
  height: 34px;
  flex: none;
}

.ld-cup::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(160deg, #9a6b3d, #5d3c20 70%);
  clip-path: polygon(15% 0, 85% 0, 100% 88%, 0 88%);
  border-radius: 3px;
}

.ld-cup::after {
  content: '';
  position: absolute;
  left: -4%;
  right: -4%;
  bottom: 0;
  height: 14%;
  background: #3f2814;
  border-radius: 3px;
}

.ld-cup-count {
  position: absolute;
  top: -7px;
  right: -9px;
  z-index: 1;
  min-width: 17px;
  height: 17px;
  border-radius: 999px;
  background: #0b0e0a;
  border: 1px solid #2d352c;
  color: #eae6d8;
  font-size: 10px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 3px;
}

.ld-roll .ld-cup {
  animation: ld-shake 0.45s ease-in-out infinite;
}

@keyframes ld-shake {
  0%, 100% { transform: rotate(0); }
  25% { transform: rotate(-7deg) translateY(-2px); }
  75% { transform: rotate(7deg) translateY(-1px); }
}

/* ---------- reveal choreography ---------- */

.ld-lose .ld-pod {
  border-color: #d96a5a;
  animation: ld-lose-flash 0.4s ease 3;
}

@keyframes ld-lose-flash {
  0%, 100% { box-shadow: 0 0 0 1px #d96a5a; }
  50% {
    box-shadow: 0 0 0 3px #d96a5a, 0 0 26px rgba(217, 106, 90, 0.7);
    background: rgba(217, 106, 90, 0.18);
  }
}

.ld-safe .ld-pod {
  border-color: #8fae6e;
  box-shadow: 0 0 0 1px #8fae6e, 0 0 18px rgba(143, 174, 110, 0.5);
}

.ld-winner .ld-pod {
  border-color: #d4a95c;
  box-shadow: 0 0 0 1px #d4a95c, 0 0 26px rgba(212, 169, 92, 0.55);
}

.ld-float {
  position: absolute;
  left: 50%;
  top: -6px;
  color: #d96a5a;
  font-weight: 800;
  font-size: 15px;
  text-shadow: 0 1px 4px rgba(0, 0, 0, 0.8);
  animation: ld-float 1s ease-out forwards;
  pointer-events: none;
  z-index: 4;
  white-space: nowrap;
}

@keyframes ld-float {
  from { opacity: 1; transform: translate(-50%, 0); }
  to { opacity: 0; transform: translate(-50%, -28px); }
}

.ld-banner {
  position: absolute;
  left: 50%;
  top: 13%;
  transform: translate(-50%, -50%) scale(0.85);
  z-index: 6;
  background: #0b0e0a;
  border: 1px solid #2d352c;
  border-radius: 999px;
  padding: 9px 22px;
  font-size: 15px;
  font-weight: 800;
  letter-spacing: 0.4px;
  white-space: nowrap;
  opacity: 0;
  transition: opacity 0.22s ease, transform 0.22s ease;
  pointer-events: none;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.5);
  color: #eae6d8;
}

.ld-banner.ld-show {
  opacity: 1;
  transform: translate(-50%, -50%) scale(1);
}

.ld-banner-liar { color: #d96a5a; border-color: #d96a5a; }
.ld-banner-exact { color: #8fae6e; border-color: #8fae6e; }
.ld-banner-good { color: #8fae6e; border-color: #8fae6e; }

.ld-fly {
  position: absolute;
  z-index: 5;
  transform: translate(-50%, -50%);
  display: flex;
  align-items: center;
  gap: 6px;
  background: #0b0e0a;
  border: 1px solid #d4a95c;
  border-radius: 999px;
  padding: 4px 12px;
  font-weight: 800;
  font-size: 14px;
  color: #eae6d8;
  pointer-events: none;
}

.ld-fly .ld-die {
  width: 18px;
  height: 18px;
}

/* ---------- controls ---------- */

/* The player's rail: a leather strip below the felt that the bid controls
 * sit on, so they read as part of the table rather than a floating toolbar. */
.ld-controls {
  display: flex;
  gap: 10px;
  justify-content: center;
  align-items: center;
  flex-wrap: wrap;
  min-height: 64px;
  padding: 9px 14px;
  background: linear-gradient(180deg, rgba(40, 56, 38, 0.5), rgba(21, 31, 20, 0.5));
  border: 1px solid rgba(212, 169, 92, 0.14);
  border-radius: 16px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), inset 0 0 24px rgba(0, 0, 0, 0.25);
}

.ld-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 18px;
  background: linear-gradient(180deg, #233527 0%, #162417 100%);
  border: 1px solid #3a4a38;
  border-radius: var(--radius);
  color: #eae6d8;
  font: inherit;
  font-weight: 700;
  cursor: pointer;
  transition: border-color 0.15s, transform 0.15s, box-shadow 0.15s;
}

.ld-btn:hover:not(:disabled) {
  border-color: #d4a95c;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}

.ld-btn:disabled {
  opacity: 0.45;
  cursor: default;
}

.ld-btn .ld-die {
  width: 20px;
  height: 20px;
}

.ld-btn-liar {
  color: #d96a5a;
  border-color: rgba(217, 106, 90, 0.55);
  letter-spacing: 1px;
}

.ld-btn-liar:hover:not(:disabled) {
  border-color: #d96a5a;
  background: rgba(217, 106, 90, 0.12);
  box-shadow: 0 0 14px rgba(217, 106, 90, 0.35);
}

.ld-btn-exact {
  color: #8fae6e;
  border-color: rgba(143, 174, 110, 0.55);
  letter-spacing: 1px;
}

.ld-btn-exact:hover:not(:disabled) {
  border-color: #8fae6e;
  background: rgba(143, 174, 110, 0.12);
  box-shadow: 0 0 14px rgba(143, 174, 110, 0.3);
}

.ld-open {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  justify-content: center;
}

.ld-open-label {
  font-size: 11px;
  color: #9da28e;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.ld-qty {
  display: flex;
  align-items: center;
  gap: 8px;
}

.ld-qty-n {
  font-size: 23px;
  font-weight: 700;
  min-width: 2ch;
  text-align: center;
}

.ld-step {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  border: 1px solid #2d352c;
  background: #1a211a;
  color: #eae6d8;
  font-size: 16px;
  font-weight: 700;
  cursor: pointer;
  line-height: 1;
}

.ld-step:hover:not(:disabled) {
  border-color: #d4a95c;
}

.ld-step:disabled {
  opacity: 0.4;
  cursor: default;
}

.ld-faces {
  display: flex;
  gap: 6px;
}

.ld-face-btn {
  padding: 3px;
  border-radius: 8px;
  border: 2px solid transparent;
  background: none;
  cursor: pointer;
  display: flex;
}

.ld-face-btn:hover {
  border-color: #2d352c;
}

.ld-face-btn.ld-sel {
  border-color: #d4a95c;
  box-shadow: 0 0 10px rgba(212, 169, 92, 0.35);
}

.ld-fallback {
  font-family: ui-monospace, monospace;
  white-space: pre-wrap;
  color: var(--text);
  padding: 12px;
}

/* The shell's opponent + difficulty picker, tucked into the pod right under the
   player's name. Compact and full-width so it reads as part of the box. */
.ld-pod .seat-slot {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 3px;
  width: 100%;
  margin: 4px 0 1px;
}
.ld-pod .seat-select,
.ld-pod .seat-level {
  width: 100%;
  font-size: 10px;
  padding: 2px 4px;
  border-radius: 5px;
}

@media (prefers-reduced-motion: reduce) {
  .ld-turn .ld-pod,
  .ld-roll .ld-cup,
  .ld-flip,
  .ld-lose .ld-pod {
    animation: none;
  }
}
`;function Wt(n){return typeof n=="object"&&n!==null&&Array.isArray(n.hands)}function Gs(n){if(typeof n!="object"||n===null)return!1;const t=n.kind;return t==="liar"||t==="exact"}const Ns={1:["c"],2:["ne","sw"],3:["ne","c","sw"],4:["nw","ne","sw","se"],5:["nw","ne","c","sw","se"],6:["nw","ne","w","e","sw","se"],7:["nw","ne","w","c","e","sw","se"],8:["nw","n","ne","w","e","sw","s","se"],9:["nw","n","ne","w","c","e","sw","s","se"]};function q(n,t=""){const e=Ns[n],s=e?e.map(o=>`<i class="ld-pip-${o}"></i>`).join(""):`<b class="ld-die-num">${n}</b>`;return`<span class="ld-die${t}" data-v="${n}">${s}</span>`}function Fs(n){return`<span class="ld-cup"><span class="ld-cup-count">${n}</span></span>`}function Rs(n,t){const e=Math.PI/180*(90+360*n/t);return{x:50+39*Math.cos(e),y:50+36*Math.sin(e)}}class Is{ctx;tableEl;seatsEl;centerEl;bannerEl;controlsEl;view=null;ladder=[];ladderRound=-1;dead=!1;seatsBuilt=!1;openQty=1;openFace=1;mount(t,e){if(this.ctx=e,!document.getElementById(Yt)){const s=document.createElement("style");s.id=Yt,s.textContent=_s,document.head.append(s)}t.innerHTML=`
      <div class="ld-root">
        <div class="ld-table">
          <div class="ld-felt"></div>
          <div class="ld-center">
            <div class="ld-round"></div>
            <div class="ld-bid-box"></div>
            <div class="ld-ladder"></div>
          </div>
          <div class="ld-seats"></div>
          <div class="ld-banner"></div>
        </div>
        <div class="ld-controls"></div>
      </div>`,this.tableEl=t.querySelector(".ld-table"),this.seatsEl=t.querySelector(".ld-seats"),this.centerEl=t.querySelector(".ld-center"),this.bannerEl=t.querySelector(".ld-banner"),this.controlsEl=t.querySelector(".ld-controls")}render(t){if(!Wt(t.viewData)){const s=document.createElement("pre");s.className="ld-fallback",s.textContent=t.view,this.seatsEl.replaceChildren(s);return}const e=t.viewData;this.view=e,this.syncLadder(e),this.renderSeats(e),this.renderCenter(e),(t.toAct!==t.humanSeat||t.isOver)&&this.controlsEl.replaceChildren()}async animate(t,e){const s=this.ctx.animationScale();if(Gs(t.data)){s>0&&!this.dead&&await this.playReveal(t.data,s),this.render(e);return}if(s>0&&!this.dead&&await this.animateBid(t.seat,e,s),this.render(e),s>0&&!this.dead){this.centerEl.querySelector(".ld-bid-main")?.animate([{transform:"scale(0.8)"},{transform:"scale(1.12)"},{transform:"scale(1)"}],{duration:300*s,easing:"ease-out"});const o=t.seat!==this.ctx.humanSeat?2:1;await y(150*o*s)}}promptAction(t){this.ctx.humanSeat<0||(t.some(e=>e.startsWith("open "))?this.renderOpenControls(t):this.renderResponseControls(t))}unmount(){this.dead=!0}name(t){return t===this.ctx.humanSeat?"You":`Player ${t}`}syncLadder(t){if(t.round!==this.ladderRound)this.ladderRound=t.round,this.ladder=[...t.history];else if(t.history.length>this.ladder.length)this.ladder=[...t.history];else if(t.bid&&!t.bid.forced){const e=this.ladder[this.ladder.length-1];(!e||e.qty!==t.bid.qty||e.face!==t.bid.face)&&this.ladder.push({seat:t.bid.by,qty:t.bid.qty,face:t.bid.face})}}handHtml(t){return t.alive?t.dice===null||t.dice.length===0?Fs(t.count):t.dice.map(e=>q(e)).join(""):'<span class="ld-out-mark">×</span>'}buildSeats(t){if(this.seatsBuilt)return;this.seatsBuilt=!0;const e=t.players,s=this.ctx.humanSeat>=0?this.ctx.humanSeat:0;this.seatsEl.innerHTML=t.hands.map(o=>{const i=Rs((o.seat-s+e)%e,e);return`
        <div class="ld-seat" data-seat="${o.seat}"
             style="left:${i.x.toFixed(2)}%;top:${i.y.toFixed(2)}%">
          <div class="ld-pod">
            <span class="ld-bubble-slot"></span>
            <div class="ld-hand"></div>
            <div class="ld-name"><span class="ld-crown-slot"></span>${this.name(o.seat)}<span class="ld-tag-slot"></span></div>
            <span class="seat-slot" data-seat="${o.seat}"></span>
          </div>
        </div>`}).join("")}renderSeats(t){this.buildSeats(t);for(const e of t.hands){const s=this.seatsEl.querySelector(`.ld-seat[data-seat="${e.seat}"]`);if(!s)continue;const o=t.phase==="over"&&t.winner===e.seat,i=["ld-seat"];e.alive||i.push("ld-out"),e.alive&&t.phase==="bidding"&&t.turn===e.seat&&i.push("ld-turn"),e.alive&&t.phase==="rolling"&&i.push("ld-roll"),o&&i.push("ld-winner"),s.className=i.join(" "),s.querySelector(".ld-bubble-slot").innerHTML=t.bid&&!t.bid.forced&&t.phase==="bidding"&&t.bid.by===e.seat?`<span class="ld-bubble">${t.bid.qty}×${q(t.bid.face)}</span>`:"",s.querySelector(".ld-hand").innerHTML=this.handHtml(e),s.querySelector(".ld-crown-slot").innerHTML=o?'<span class="ld-crown">★</span>':"",s.querySelector(".ld-tag-slot").innerHTML=e.alive?` <span class="ld-tag">${e.count} ${e.count===1?"die":"dice"}</span>`:' <span class="ld-out-tag">OUT</span>'}}renderCenter(t){const e=this.centerEl.querySelector(".ld-round"),s=this.centerEl.querySelector(".ld-bid-box"),o=this.centerEl.querySelector(".ld-ladder");if(e.textContent=`round ${t.round} · ${t.totalDice} dice in play`,t.phase==="over"&&t.winner!==null){const a=t.winner===this.ctx.humanSeat?"win":"wins";s.innerHTML=`<span class="ld-win-text">★ ${this.name(t.winner)} ${a}</span>`}else if(t.phase==="rolling")s.innerHTML='<span class="ld-open-hint">shaking the cups…</span>';else if(t.bid){const a=t.bid.by===this.ctx.humanSeat?"bid":"bids",r=t.bid.forced?"forced opening bid":`${this.name(t.bid.by)} ${a}`;s.innerHTML=`
        <div class="ld-bid-main">${t.bid.qty}<span class="ld-x">×</span>${q(t.bid.face)}</div>
        <div class="ld-bid-sub">${r}</div>`}else{const a=t.turn===this.ctx.humanSeat?"open":"opens";s.innerHTML=`<span class="ld-open-hint">${this.name(t.turn)} ${a} the round…</span>`}const i=this.ladder.slice(-6);o.innerHTML=i.map((a,r)=>{const l=a.seat===this.ctx.humanSeat?"you":`P${a.seat}`;return`<div class="ld-rung${r===i.length-1?" ld-rung-now":""}"><span>${l}</span> ${a.qty}×${q(a.face)}</div>`}).join("")}submit(t){for(const e of this.controlsEl.querySelectorAll("button"))e.disabled=!0;this.ctx.submit(String(t))}renderResponseControls(t){const e=this.view?.bid,s=this.view?.faces??6,o=t.map((i,a)=>{const r=document.createElement("button");if(r.type="button",r.className="ld-btn",i==="raise quantity"&&e)r.innerHTML=`Raise to ${e.qty+1}×${q(e.face)}`;else if(i==="raise face"&&e){const[l,c]=e.face<s?[e.qty,e.face+1]:[e.qty+1,1];r.innerHTML=`Raise to ${l}×${q(c)}`}else i==="call LIAR"?(r.classList.add("ld-btn-liar"),r.textContent="LIAR!"):i==="call EXACT"?(r.classList.add("ld-btn-exact"),r.textContent="EXACT"):r.textContent=i;return r.onclick=()=>this.submit(a),r});this.controlsEl.replaceChildren(...o)}renderOpenControls(t){const e=new Map;let s=1;for(const[u,m]of t.entries()){const x=/^open (\d+)x(\d+)$/.exec(m);x&&(e.set(`${x[1]}x${x[2]}`,u),s=Math.max(s,Number(x[1])))}const o=this.view?.faces??6,i=this.view?.hands.find(u=>u.seat===this.ctx.humanSeat)?.dice??[],a=new Array(o+1).fill(0);for(const u of i)a[u]++;let r=1;for(let u=1;u<=o;u++)a[u]>=a[r]&&(r=u);this.openFace=r,this.openQty=Math.min(s,Math.max(1,a[r]));const l=document.createElement("div");l.className="ld-open",l.innerHTML=`
      <span class="ld-open-label">open the round</span>
      <div class="ld-qty">
        <button type="button" class="ld-step ld-minus">−</button>
        <span class="ld-qty-n"></span>
        <button type="button" class="ld-step ld-plus">+</button>
      </div>
      <div class="ld-faces"></div>
      <button type="button" class="ld-btn ld-go"></button>`;const c=l.querySelector(".ld-qty-n"),d=l.querySelector(".ld-faces"),h=l.querySelector(".ld-go"),p=l.querySelector(".ld-minus"),g=l.querySelector(".ld-plus"),f=[],b=()=>{c.textContent=String(this.openQty),p.disabled=this.openQty<=1,g.disabled=this.openQty>=s,f.forEach((u,m)=>u.classList.toggle("ld-sel",m+1===this.openFace)),h.innerHTML=`Bid ${this.openQty}×${q(this.openFace)}`,h.disabled=!e.has(`${this.openQty}x${this.openFace}`)};for(let u=1;u<=o;u++){const m=document.createElement("button");m.type="button",m.className="ld-face-btn",m.innerHTML=q(u),m.onclick=()=>{this.openFace=u,b()},f.push(m),d.append(m)}p.onclick=()=>{this.openQty=Math.max(1,this.openQty-1),b()},g.onclick=()=>{this.openQty=Math.min(s,this.openQty+1),b()},h.onclick=()=>{const u=e.get(`${this.openQty}x${this.openFace}`);u!==void 0&&this.submit(u)},b(),this.controlsEl.replaceChildren(l)}showBanner(t,e){this.bannerEl.textContent=t,this.bannerEl.className=`ld-banner ld-banner-${e} ld-show`}hideBanner(){this.bannerEl.classList.remove("ld-show")}async animateBid(t,e,s){const o=Wt(e.viewData)?e.viewData.bid:null,i=this.seatsEl.querySelector(`[data-seat="${t}"]`);if(!o||!i)return;const a=document.createElement("div");a.className="ld-fly",a.innerHTML=`${o.qty}×${q(o.face)}`,a.style.left=i.style.left,a.style.top=i.style.top,this.tableEl.append(a);const r=this.tableEl.getBoundingClientRect(),l=(50-parseFloat(i.style.left))/100*r.width,c=(46-parseFloat(i.style.top))/100*r.height;await a.animate([{transform:"translate(-50%, -50%)",opacity:1},{transform:`translate(calc(-50% + ${l}px), calc(-50% + ${c}px))`,opacity:.15}],{duration:480*s,easing:"cubic-bezier(0.3, 0.7, 0.4, 1)",fill:"forwards"}).finished.catch(()=>{}),a.remove()}setTally(t,e,s){const o=this.centerEl.querySelector(".ld-bid-box");o&&(o.innerHTML=`
      <div class="ld-bid-main">
        <span class="ld-tally-n">${t??"?"}</span><span class="ld-x">/</span>${s}<span class="ld-x">×</span>${q(e)}
      </div>
      <div class="ld-bid-sub">counting ${e}s across the table…</div>`)}async playReveal(t,e){const s=p=>p*e,o=t.hands.length,i=t.bid.face,a=this.ctx.humanSeat,r=t.caller===a?"call":"calls",l=t.kind==="liar"?"LIAR":"EXACT";if(this.showBanner(`${this.name(t.caller)} ${r} ${l} on ${t.bid.qty}×${i}!`,t.kind==="liar"?"liar":"exact"),this.setTally(null,i,t.bid.qty),await y(s(900)),this.dead)return;let c=0;for(let p=0;p<o;p++){const g=(t.caller+p)%o,f=t.hands[g];if(!f.length)continue;const b=this.seatsEl.querySelector(`[data-seat="${g}"] .ld-hand`);if(b&&(b.innerHTML=f.map(u=>q(u,u===i?" ld-hit ld-flip":" ld-flip")).join("")),c+=f.filter(u=>u===i).length,this.setTally(c,i,t.bid.qty),await y(s(380)),this.dead)return}if(await y(s(250)),this.dead)return;const d=t.loser===null?"":` ${this.name(t.loser)} ${t.loser===a?"lose":"loses"} a die.`;let h;if(t.kind==="liar"?h=t.actual<t.bid.qty?`A lie — only ${t.actual}!${d}`:`The bid was good — ${t.actual} on the table.${d}`:h=t.loser===null?`EXACT — dead on ${t.actual}! Nobody loses a die.`:`Not exact — ${t.actual}, not ${t.bid.qty}.${d}`,this.showBanner(h,t.loser===null?"good":"liar"),t.loser!==null){const p=this.seatsEl.querySelector(`[data-seat="${t.loser}"]`);p?.classList.add("ld-lose");const g=document.createElement("span");g.className="ld-float",g.textContent="−1 die",p?.querySelector(".ld-pod")?.append(g)}else this.seatsEl.querySelector(`[data-seat="${t.caller}"]`)?.classList.add("ld-safe");if(await y(s(1100)),!this.dead&&!(t.loser!==null&&t.diceLeft[t.loser]===0&&!t.gameOver&&(this.showBanner(`${this.name(t.loser)} ${t.loser===a?"are":"is"} out of the game!`,"liar"),await y(s(900)),this.dead))){if(t.gameOver&&t.winner!==null){this.seatsEl.querySelector(`[data-seat="${t.winner}"]`)?.classList.add("ld-winner");const p=t.adjudicated?" on dice count (round cap reached)":"";this.showBanner(`★ ${this.name(t.winner)} ${t.winner===a?"win":"wins"} the game${p}!`,"good"),await y(s(1200))}this.hideBanner()}}}function js(){return new Is}const E=8,at=["Black","White"];function Ut(n){if(!n||typeof n!="object")return null;const t=n;return typeof t.cells=="string"&&t.cells.length===E*E&&Array.isArray(t.counts)&&Array.isArray(t.legal)?t:null}function Ys(n){if(!n||typeof n!="object")return null;const t=n;return typeof t.move=="string"&&typeof t.player=="number"&&Array.isArray(t.flipped)?t:null}function Vt(n){return/^[a-h][1-8]$/.test(n)?(n.charCodeAt(1)-49)*E+(n.charCodeAt(0)-97):null}function Ws(n,t){return Math.max(Math.abs(Math.floor(n/E)-Math.floor(t/E)),Math.abs(n%E-t%E))}function Us(n,t){if(!n||!t)return null;let e=null;const s=[];for(let o=0;o<E*E;o++)n.cells[o]!==t.cells[o]&&(n.cells[o]==="."?e=o:s.push(o));return e===null?{move:"pass",player:n.turn,placed:null,flipped:[]}:{move:"place",player:t.cells[e]==="b"?0:1,placed:e,flipped:s}}const Kt="othello-frontend-style",Vs=`
.ot-root {
  align-self: center;
  width: min(100%, var(--board-fit));
  display: flex;
  flex-direction: column;
  gap: 14px;
  user-select: none;
}
.ot-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px 10px;
}
.ot-score {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--bg-inset);
  color: var(--text-dim);
  font-size: 0.88rem;
  white-space: nowrap;
  transition: border-color 0.25s, box-shadow 0.25s, color 0.25s;
}
.ot-score.ot-active {
  border-color: var(--accent);
  color: var(--text);
  box-shadow: 0 0 12px rgba(88, 166, 255, 0.3);
}
.ot-mini {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  flex: none;
}
.ot-mini-b {
  background: radial-gradient(circle at 35% 30%, #59636e, #11151b 75%);
  box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.25);
}
.ot-mini-w {
  background: radial-gradient(circle at 35% 30%, #ffffff, #c2cad4 80%);
  box-shadow: inset 0 -1px 1px rgba(0, 0, 0, 0.2);
}
.ot-count {
  font-weight: 700;
  color: var(--text);
  min-width: 1.4em;
  text-align: center;
}
.ot-msg {
  flex: 1;
  text-align: center;
  color: var(--text-dim);
  font-size: 0.92rem;
}
.ot-board {
  position: relative;
  display: grid;
  grid-template-columns: repeat(${E}, 1fr);
  grid-template-rows: repeat(${E}, 1fr);
  aspect-ratio: 1;
  border: 10px solid #18221b;
  border-radius: var(--radius);
  background:
    repeating-linear-gradient(48deg, rgba(255, 255, 255, 0.02) 0 2px, transparent 2px 5px),
    linear-gradient(158deg, #31894e, #1d5c31);
  box-shadow: 0 8px 22px rgba(0, 0, 0, 0.25), inset 0 0 24px rgba(0, 0, 0, 0.28);
}
.dark .ot-board {
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45), inset 0 0 24px rgba(0, 0, 0, 0.28);
}
.ot-cell {
  position: relative;
  box-shadow: inset 0 0 0 1px rgba(4, 28, 13, 0.55);
}
.ot-star {
  position: absolute;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: rgba(4, 28, 13, 0.6);
  transform: translate(-50%, -50%);
  pointer-events: none;
}
.ot-cell.ot-legal::after,
.ot-cell.ot-hint::after {
  content: '';
  position: absolute;
  inset: 38%;
  border-radius: 50%;
  background: rgba(4, 28, 13, 0.45);
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.4);
}
.ot-cell.ot-hint::after {
  inset: 43%;
  background: rgba(4, 28, 13, 0.32);
}
.ot-board.ot-live .ot-cell.ot-legal {
  cursor: pointer;
}
.ot-board.ot-live.ot-human-b .ot-cell.ot-legal:hover::after {
  inset: 12%;
  background: radial-gradient(circle at 35% 30%, rgba(89, 99, 110, 0.8), rgba(17, 21, 27, 0.8) 75%);
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.35);
}
.ot-board.ot-live.ot-human-w .ot-cell.ot-legal:hover::after {
  inset: 12%;
  background: radial-gradient(circle at 35% 30%, rgba(255, 255, 255, 0.85), rgba(194, 202, 212, 0.85) 80%);
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.35);
}
.ot-disc {
  position: absolute;
  inset: 11%;
  perspective: 240px;
  pointer-events: none;
}
.ot-flip {
  position: absolute;
  inset: 0;
  transform-style: preserve-3d;
  will-change: transform;
}
.ot-disc.ot-w .ot-flip {
  transform: rotateY(180deg);
}
.ot-face {
  position: absolute;
  inset: 0;
  border-radius: 50%;
  backface-visibility: hidden;
  -webkit-backface-visibility: hidden;
}
.ot-face-b {
  background: radial-gradient(circle at 35% 28%, #6b7684, #2a313b 45%, #0d1117 85%);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.5), inset 0 1px 2px rgba(255, 255, 255, 0.25);
}
.ot-face-w {
  transform: rotateY(180deg);
  background: radial-gradient(circle at 35% 28%, #ffffff, #dde3ea 55%, #a9b2be 92%);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.5), inset 0 -2px 3px rgba(0, 0, 0, 0.18);
}
.ot-toast {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  padding: 10px 20px;
  background: rgba(1, 4, 9, 0.88);
  border: 1px solid rgba(230, 237, 243, 0.2);
  border-radius: var(--radius);
  color: #e6edf3;
  font-weight: 600;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.2s;
  z-index: 2;
}
.ot-toast.ot-show {
  opacity: 1;
}
.ot-pass {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  display: none;
  padding: 10px 26px;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  border: none;
  border-radius: 999px;
  color: #fff;
  font-weight: 700;
  font-size: 1rem;
  cursor: pointer;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4);
  z-index: 2;
}
.ot-pass.ot-show {
  display: block;
}
.ot-fallback {
  display: none;
  margin: 0;
  font-family: ui-monospace, monospace;
  color: var(--text);
  white-space: pre;
}
.ot-root.ot-text-only .ot-bar,
.ot-root.ot-text-only .ot-board {
  display: none;
}
.ot-root.ot-text-only .ot-fallback {
  display: block;
}
`;function Ks(){if(document.getElementById(Kt))return;const n=document.createElement("style");n.id=Kt,n.textContent=Vs,document.head.append(n)}class Qs{ctx;rootEl;boardEl;msgEl;toastEl;passEl;fallbackEl;scoreEls=[];countEls=[];cells=[];discs=new Map;view=null;actionBySq=null;anims=new Set;mount(t,e){this.ctx=e,Ks();const s=o=>`
      <div class="ot-score ot-score-${o}">
        <span class="ot-mini ot-mini-${o===0?"b":"w"}"></span>
        <span>${at[o]}</span>
        <span class="seat-slot" data-seat="${o}"></span>
        <span class="ot-count">0</span>
      </div>`;t.innerHTML=`
      <div class="ot-root">
        <div class="ot-bar">${s(0)}<div class="ot-msg"></div>${s(1)}</div>
        <div class="ot-board">
          ${'<div class="ot-cell"></div>'.repeat(E*E)}
          <div class="ot-toast"></div>
          <button type="button" class="ot-pass">Pass</button>
        </div>
        <pre class="ot-fallback"></pre>
      </div>`,this.rootEl=t.querySelector(".ot-root"),this.boardEl=t.querySelector(".ot-board"),this.msgEl=t.querySelector(".ot-msg"),this.toastEl=t.querySelector(".ot-toast"),this.passEl=t.querySelector(".ot-pass"),this.fallbackEl=t.querySelector(".ot-fallback"),this.scoreEls=[t.querySelector(".ot-score-0"),t.querySelector(".ot-score-1")],this.countEls=this.scoreEls.map(o=>o.querySelector(".ot-count")),this.cells=[...this.boardEl.querySelectorAll(".ot-cell")];for(const o of[25,75])for(const i of[25,75]){const a=document.createElement("div");a.className="ot-star",a.style.left=`${o}%`,a.style.top=`${i}%`,this.boardEl.append(a)}this.boardEl.addEventListener("click",o=>{const i=o.target.closest(".ot-cell");i&&this.clickSquare(this.cells.indexOf(i))})}render(t){this.disableInput();const e=Ut(t.viewData);if(this.view=e,!e){this.rootEl.classList.add("ot-text-only"),this.fallbackEl.textContent=t.view;return}this.rootEl.classList.remove("ot-text-only"),this.rebuildDiscs(e);for(let i=0;i<2;i++)this.countEls[i].textContent=String(e.counts[i]),this.scoreEls[i].classList.toggle("ot-active",!t.isOver&&e.turn===i);if(this.ctx.humanSeat<0&&!t.isOver)for(const i of e.legal){const a=Vt(i);a!==null&&this.cells[a].classList.add("ot-hint")}const[s,o]=e.counts;this.msgEl.textContent=t.isOver?s===o?`Draw, ${s}–${o}`:`${at[s>o?0:1]} wins ${Math.max(s,o)}–${Math.min(s,o)}`:""}async animate(t,e){const s=this.view,o=Ut(e.viewData),i=Ys(t.data)??Us(s,o),a=this.ctx.animationScale();if(i?.move==="pass"||i?.placed==null){a>0&&i&&(this.toastEl.textContent=`${at[i.player]} passes`,this.toastEl.classList.add("ot-show"),await y(800*a),this.toastEl.classList.remove("ot-show")),this.render(e);return}if(this.render(e),!o||a<=0)return;const r=[],l=this.discs.get(i.placed);l&&r.push(this.run(l.animate([{transform:"scale(0.2)",opacity:.4,offset:0},{transform:"scale(1.14)",opacity:1,offset:.7},{transform:"scale(1)",offset:1}],{duration:240*a,easing:"ease-out"})));for(const c of i.flipped){const d=this.discs.get(c)?.querySelector(".ot-flip");if(!d)continue;const h=o.cells[c]==="w",[p,g,f]=h?[0,90,180]:[180,270,360];r.push(this.run(d.animate([{transform:`rotateY(${p}deg) scale(1)`},{transform:`rotateY(${g}deg) scale(1.18)`},{transform:`rotateY(${f}deg) scale(1)`}],{duration:340*a,delay:(110+85*(Ws(i.placed,c)-1))*a,easing:"ease-in-out",fill:"backwards"})))}await Promise.all(r),await y(70*a)}promptAction(t){const e=t.indexOf("pass");if(e>=0&&t.length===1){this.passEl.classList.add("ot-show"),this.passEl.onclick=()=>{this.disableInput(),this.ctx.submit(String(e))};return}const s=new Map;t.forEach((o,i)=>{const a=Vt(o);a!==null&&(s.set(a,i),this.cells[a].classList.add("ot-legal"))}),this.actionBySq=s,this.boardEl.classList.add("ot-live",this.ctx.humanSeat===1?"ot-human-w":"ot-human-b")}unmount(){for(const t of this.anims)t.cancel();this.anims.clear()}rebuildDiscs(t){this.discs.clear();for(let e=0;e<E*E;e++){const s=t.cells[e];if(s==="."){this.cells[e].replaceChildren();continue}const o=document.createElement("div");o.className=`ot-disc ${s==="b"?"ot-b":"ot-w"}`,o.innerHTML='<div class="ot-flip"><div class="ot-face ot-face-b"></div><div class="ot-face ot-face-w"></div></div>',this.cells[e].replaceChildren(o),this.discs.set(e,o)}}clickSquare(t){const e=this.actionBySq?.get(t);e!==void 0&&(this.disableInput(),this.ctx.submit(String(e)))}disableInput(){this.actionBySq=null,this.passEl.classList.remove("ot-show"),this.passEl.onclick=null,this.boardEl.classList.remove("ot-live","ot-human-b","ot-human-w");for(const t of this.cells)t.classList.remove("ot-legal","ot-hint")}async run(t){this.anims.add(t);try{await t.finished}catch{}finally{this.anims.delete(t)}}}function Xs(){return new Qs}const Qt=5,Zs=5;function Xt(n){if(!n||typeof n!="object")return null;const t=n;return typeof t.size=="number"&&typeof t.cells=="string"&&t.cells.length===t.size*t.size&&Array.isArray(t.pairs)?t:null}const Zt="pente-frontend-style",C=1;function $e(n){return String.fromCharCode(97+n+(n>=8?1:0))}function Js(n,t){return`${$e(n%t)}${Math.floor(n/t)+1}`}function to(n,t){const e=n.charCodeAt(0)-97;if(e<0||e>25||n[0]==="i")return null;const s=e>8?e-1:e,o=parseInt(n.slice(1),10);return!Number.isFinite(o)||s>=t||o<1||o>t?null:(o-1)*t+s}function eo(n){const t=[],e=C+n-1;for(let s=0;s<n;s++){const o=C+s;t.push(`M ${o} ${C} L ${o} ${e}`,`M ${C} ${o} L ${e} ${o}`)}return t.join(" ")}function so(n){const t=[],e=n>=13?3:2;if(n>=7)for(const s of[e,n-1-e])for(const o of[e,n-1-e])t.push(s*n+o);if(n%2===1&&n>=5){const s=(n-1)/2;t.push(s*n+s)}return t}function oo(n,t,e){const s=[[0,1],[1,0],[1,1],[1,-1]];for(let o=0;o<n.length;o++){if(n[o]!==e)continue;const i=Math.floor(o/t),a=o%t;for(const[r,l]of s){const c=[o];let d=i+r,h=a+l;for(;d>=0&&h>=0&&d<t&&h<t&&n[d*t+h]===e;)c.push(d*t+h),d+=r,h+=l;if(c.length>=Zs)return c}}return null}const io=`
.pente { display: flex; flex-direction: column; gap: 14px; width: min(100%, var(--board-fit)); margin: 0 auto; }
.pente-hud { display: grid; grid-template-columns: 1fr auto 1fr; align-items: stretch; gap: 10px; }
.pente-player { display: flex; align-items: center; gap: 11px; padding: 9px 14px; min-width: 0;
  border-radius: var(--radius); background: var(--bg-raised); border: 1px solid var(--border);
  transition: border-color .2s, box-shadow .2s; }
.pente-player.pente-active { border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent), 0 0 18px rgba(99, 102, 241, .26); }
.pente-stone-icon { width: 22px; height: 22px; border-radius: 50%; flex: none;
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, .25), 0 1px 3px rgba(0, 0, 0, .5); }
.pente-stone-icon-b { background: radial-gradient(circle at 34% 28%, #5b6478, #24293a 44%, #070910); }
.pente-stone-icon-w { background: radial-gradient(circle at 34% 28%, #ffffff, #e7eaf4 58%, #b9bfd2); }
.pente-pinfo { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
.pente-pname { font-weight: 600; line-height: 1.2; }
.pente-pcaps { margin-left: auto; display: flex; align-items: center; gap: 5px;
  color: var(--text-dim); line-height: 1; white-space: nowrap; }
.pente-pips { display: inline-flex; gap: 3px; align-self: center; }
.pente-pip { width: 7px; height: 7px; border-radius: 50%; background: var(--border);
  transition: background .25s, box-shadow .25s; }
.pente-pip.pente-pip-on { background: var(--accent);
  box-shadow: 0 0 6px rgba(99, 102, 241, .7); }
.pente-turn-chip { align-self: center; display: flex; align-items: center; gap: 8px; padding: 7px 14px;
  border-radius: 999px; background: var(--bg-inset); border: 1px solid var(--border);
  font-size: 13px; color: var(--text-dim); white-space: nowrap;
  transition: opacity .2s; }
.pente-turn-chip.pente-chip-hidden { opacity: 0; }
.pente-turn-dot { width: 11px; height: 11px; border-radius: 50%; flex: none;
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, .25), 0 1px 2px rgba(0, 0, 0, .4); }
.pente-board-wrap { position: relative; width: 100%; margin: 0 auto; }
.pente-svg { display: block; width: 100%; height: auto; border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, .26), 0 2px 6px rgba(0, 0, 0, .18); }
.dark .pente-svg { box-shadow: 0 14px 40px rgba(0, 0, 0, .55), 0 2px 8px rgba(0, 0, 0, .42); }
.pente-hit { fill: transparent; }
.pente-hit-on { cursor: pointer; }
.pente-ghost, .pente-marker, .pente-winline { pointer-events: none; }
.pente-drop { transform-box: fill-box; transform-origin: center;
  animation: pente-drop .26s cubic-bezier(.2, .85, .35, 1.2) backwards; }
@keyframes pente-drop {
  from { transform: scale(.4); opacity: 0; }
  70% { opacity: 1; }
  to { transform: none; opacity: 1; }
}
.pente-cap { transform-box: fill-box; transform-origin: center;
  animation: pente-cap .36s ease-in forwards; }
@keyframes pente-cap {
  40% { transform: scale(1.18); }
  to { transform: scale(.2); opacity: 0; }
}
.pente-win-stone { animation: pente-pulse 1.1s ease-in-out infinite; }
@keyframes pente-pulse {
  0%, 100% { filter: brightness(1); }
  50% { filter: brightness(1.55) drop-shadow(0 0 .12px #fff); }
}
.pente-toast { position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
  background: rgba(2, 3, 12, .82); border: 1px solid rgba(180, 186, 220, .25); color: #eef0fb;
  padding: 6px 16px; border-radius: 999px; font-size: 13px; white-space: nowrap;
  opacity: 0; pointer-events: none; transition: opacity .2s; }
.pente-toast-show { opacity: 1; }
@media (prefers-reduced-motion: reduce) {
  .pente-win-stone { animation: none; filter: brightness(1.4); }
}
@media (max-width: 560px) {
  .pente-hud { grid-template-columns: 1fr 1fr; }
  .pente-turn-chip { order: 3; grid-column: 1 / -1; justify-self: center; }
  .pente-turn-chip.pente-chip-hidden { display: none; }
  /* Narrow plaque: the dropdown takes the width, so stack the pips under the
     name row. */
  .pente-player { align-items: flex-start; }
  .pente-stone-icon { margin-top: 1px; }
  .pente-pcaps { align-self: flex-start; }
}
`;function ao(){if(document.getElementById(Zt))return;const n=document.createElement("style");n.id=Zt,n.textContent=io,document.head.append(n)}class no{ctx;svg;stonesG;fxG;ghostEl;markerEl;winLineEl;toastEl;turnChip;plaques=[];pipRows=[];size=0;view=null;lastMove=null;interactive=!1;labelIndex=new Map;legalPoints=new Set;stoneEls=new Map;unsubDebug=null;evalGen=0;mount(t,e){this.ctx=e,ao(),t.innerHTML=`
      <div class="pente">
        <div class="pente-hud">
          <div class="pente-player" data-seat="0">
            <span class="pente-stone-icon pente-stone-icon-b"></span>
            <span class="pente-pinfo"><span class="pente-pname">Black</span><span class="seat-slot" data-seat="0"></span></span>
            <span class="pente-pcaps"><span class="pente-pips" data-seat="0"></span></span>
          </div>
          <div class="pente-turn-chip"><span class="pente-turn-dot"></span><span class="pente-turn-text"></span></div>
          <div class="pente-player" data-seat="1">
            <span class="pente-stone-icon pente-stone-icon-w"></span>
            <span class="pente-pinfo"><span class="pente-pname">White</span><span class="seat-slot" data-seat="1"></span></span>
            <span class="pente-pcaps"><span class="pente-pips" data-seat="1"></span></span>
          </div>
        </div>
        <div class="pente-board-wrap">
          <svg class="pente-svg" role="img" aria-label="Pente board"></svg>
          <div class="pente-toast"></div>
        </div>
      </div>`,this.svg=t.querySelector(".pente-svg"),this.toastEl=t.querySelector(".pente-toast"),this.turnChip=t.querySelector(".pente-turn-chip"),this.plaques=[...t.querySelectorAll(".pente-player")],this.pipRows=[...t.querySelectorAll(".pente-pips")];for(const s of this.pipRows)for(let o=0;o<Qt;o++){const i=document.createElement("span");i.className="pente-pip",s.append(i)}this.unsubDebug=e.onDebugChange(s=>{s?this.refreshEval():this.ctx.setDebugReadout([])})}xy(t){return{x:C+t%this.size,y:C+(this.size-1-Math.floor(t/this.size))}}buildBoard(t){this.size=t;const e=t-1+2*C;this.svg.setAttribute("viewBox",`0 0 ${e} ${e}`);const s=so(t).map(l=>{const{x:c,y:d}=this.xy(l);return`<circle cx="${c}" cy="${d}" r="${t>13?.08:.1}" fill="rgba(150,160,210,.5)"/>`}).join(""),o=[];for(let l=0;l<t;l++)o.push(`<text x="${C+l}" y="${C+t-1+.72}">${$e(l)}</text>`,`<text x="${C-.66}" y="${C+(t-1-l)+.11}">${l+1}</text>`);const i=[];for(let l=0;l<t*t;l++){const{x:c,y:d}=this.xy(l);i.push(`<rect class="pente-hit" data-p="${l}" x="${c-.5}" y="${d-.5}" width="1" height="1"/>`)}this.svg.innerHTML=`
      <defs>
        <linearGradient id="pente-board" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#2b3350"/>
          <stop offset="0.4" stop-color="#222942"/>
          <stop offset="1" stop-color="#171c30"/>
        </linearGradient>
        <radialGradient id="pente-sheen" cx="0.5" cy="0.18" r="1.1">
          <stop offset="0" stop-color="rgba(180,190,235,.22)"/>
          <stop offset="0.55" stop-color="rgba(180,190,235,0)"/>
          <stop offset="1" stop-color="rgba(4,6,16,.32)"/>
        </radialGradient>
        <radialGradient id="pente-stone-b" cx="0.36" cy="0.3" r="0.95">
          <stop offset="0" stop-color="#5b6478"/>
          <stop offset="0.42" stop-color="#24293a"/>
          <stop offset="1" stop-color="#070910"/>
        </radialGradient>
        <radialGradient id="pente-stone-w" cx="0.36" cy="0.3" r="0.95">
          <stop offset="0" stop-color="#ffffff"/>
          <stop offset="0.6" stop-color="#e7eaf4"/>
          <stop offset="1" stop-color="#b9bfd2"/>
        </radialGradient>
        <filter id="pente-shadow" x="-30%" y="-30%" width="160%" height="160%">
          <feDropShadow dx="0.015" dy="0.05" stdDeviation="0.045" flood-color="#000" flood-opacity="0.5"/>
        </filter>
      </defs>
      <rect width="${e}" height="${e}" rx="0.32" fill="url(#pente-board)"/>
      <rect width="${e}" height="${e}" rx="0.32" fill="url(#pente-sheen)"/>
      <path d="${eo(t)}" stroke="rgba(150,162,210,.4)" stroke-width="0.028" fill="none" stroke-linecap="square"/>
      ${s}
      <g fill="rgba(170,180,222,.5)" font-size="0.32" text-anchor="middle" font-family="inherit">${o.join("")}</g>
      <g class="pente-stones" filter="url(#pente-shadow)"></g>
      <g class="pente-fx"></g>
      <path class="pente-winline" fill="none" stroke="rgba(129,140,248,.9)" stroke-width="0.12" stroke-linecap="round" opacity="0"/>
      <circle class="pente-marker" r="0.17" fill="none" stroke-width="0.07" opacity="0"/>
      <circle class="pente-ghost" r="0.45" opacity="0"/>
      <g class="pente-hits"></g>`,this.stonesG=this.svg.querySelector(".pente-stones"),this.fxG=this.svg.querySelector(".pente-fx"),this.winLineEl=this.svg.querySelector(".pente-winline"),this.markerEl=this.svg.querySelector(".pente-marker"),this.ghostEl=this.svg.querySelector(".pente-ghost");const a=this.svg.querySelector(".pente-hits");a.innerHTML=i.join("");const r=l=>{const c=l.target.getAttribute?.("data-p");return c==null?null:Number(c)};a.addEventListener("click",l=>{const c=r(l);c!==null&&this.tryPlay(c)}),a.addEventListener("pointerover",l=>this.showGhost(r(l))),a.addEventListener("pointerout",()=>this.showGhost(null))}tryPlay(t){if(!this.interactive||!this.legalPoints.has(t))return;const e=this.labelIndex.get(Js(t,this.size));e!==void 0&&(this.setInteractive(!1),this.ctx.submit(String(e)))}showGhost(t){if(t===null||!this.interactive||!this.legalPoints.has(t)||this.view?.cells[t]!=="."){this.ghostEl.setAttribute("opacity","0");return}const{x:e,y:s}=this.xy(t);this.ghostEl.setAttribute("cx",String(e)),this.ghostEl.setAttribute("cy",String(s)),this.ghostEl.setAttribute("fill",this.ctx.humanSeat===1?"rgba(250,250,255,.6)":"rgba(14,16,28,.6)"),this.ghostEl.setAttribute("opacity","1")}setInteractive(t){this.interactive=t,t||this.ghostEl.setAttribute("opacity","0"),this.svg.querySelectorAll(".pente-hit").forEach(e=>e.classList.toggle("pente-hit-on",t&&this.legalPoints.has(Number(e.getAttribute("data-p")))))}drawStones(t){this.stoneEls.clear(),this.stonesG.replaceChildren();for(let e=0;e<t.cells.length;e++){const s=t.cells[e];s!=="b"&&s!=="w"||this.stonesG.append(this.makeStone(e,s==="b"?0:1))}if(this.lastMove!==null&&t.cells[this.lastMove]!=="."){const{x:e,y:s}=this.xy(this.lastMove);this.markerEl.setAttribute("cx",String(e)),this.markerEl.setAttribute("cy",String(s)),this.markerEl.setAttribute("stroke",t.cells[this.lastMove]==="b"?"#eef0fb":"#1a1c2c"),this.markerEl.setAttribute("opacity","1")}else this.markerEl.setAttribute("opacity","0")}makeStone(t,e){const{x:s,y:o}=this.xy(t),i=document.createElementNS("http://www.w3.org/2000/svg","circle");return i.setAttribute("cx",String(s)),i.setAttribute("cy",String(o)),i.setAttribute("r","0.46"),i.setAttribute("fill",e===0?"url(#pente-stone-b)":"url(#pente-stone-w)"),this.stoneEls.set(t,i),i}showWin(t){if(t.winner===null||t.pairs[t.winner]>=Qt)return;const e=oo(t.cells,t.size,t.winner===0?"b":"w");if(!e)return;const s=e.map((o,i)=>{const{x:a,y:r}=this.xy(o);return`${i===0?"M":"L"} ${a} ${r}`}).join(" ");this.winLineEl.setAttribute("d",s),this.winLineEl.setAttribute("opacity","1");for(const o of e)this.stoneEls.get(o)?.classList.add("pente-win-stone")}render(t){const e=Xt(t.viewData);if(!e)return;e.size!==this.size&&this.buildBoard(e.size),this.view=e,this.winLineEl.setAttribute("opacity","0"),this.drawStones(e);for(let i=0;i<2;i++){const a=this.pipRows[i].children;for(let r=0;r<a.length;r++)a[r].classList.toggle("pente-pip-on",r<e.pairs[i])}const s=this.turnChip.querySelector(".pente-turn-dot"),o=this.turnChip.querySelector(".pente-turn-text");if(t.isOver)this.showWin(e),o.textContent=e.winner===null?"Draw — board full":`${e.winner===0?"Black":"White"} wins`,s.style.background="var(--text-dim)",this.plaques.forEach(i=>i.classList.remove("pente-active"));else{const i=e.cells.split("").every(a=>a===".");o.textContent=i?"Black opens at the center":"",s.style.background=e.turn===0?"radial-gradient(circle at 35% 30%, #5b6478, #070910)":"radial-gradient(circle at 35% 30%, #ffffff, #b9bfd2)",this.plaques.forEach((a,r)=>a.classList.toggle("pente-active",r===e.turn))}this.turnChip.classList.toggle("pente-chip-hidden",o.textContent===""),t.toAct!==t.humanSeat&&this.setInteractive(!1),this.refreshEval()}async animate(t,e){const s=t.data??null,o=this.ctx.animationScale(),i=Xt(e.viewData);if(i&&i.size!==this.size&&this.buildBoard(i.size),s&&typeof s.point=="number"){if(this.lastMove=s.point,this.render(e),o>0){const a=this.stoneEls.get(s.point);a&&(a.style.animationDuration=`${260*o}ms`,a.classList.add("pente-drop"));const r=s.captured??[];for(const l of r){const c=this.makeStone(l,s.seat^1);this.stoneEls.delete(l),c.style.animationDuration=`${360*o}ms`,c.style.animationDelay=`${110*o}ms`,c.classList.add("pente-cap"),this.fxG.append(c)}if(r.length>0&&!e.isOver){const l=r.length/2;this.toastEl.textContent=`${s.seat===0?"Black":"White"} captures ${l} pair${l===1?"":"s"}`,this.toastEl.classList.add("pente-toast-show")}await y((r.length>0?520:300)*o),this.fxG.replaceChildren(),this.toastEl.classList.remove("pente-toast-show")}}else this.render(e),await y(200*o)}refreshEval(){if(!this.ctx.debug())return;const t=++this.evalGen;ss().then(e=>{if(t!==this.evalGen)return;if(!e){this.ctx.setDebugReadout([]);return}const s=Math.round(e.value*100);this.ctx.setDebugReadout([`AlphaZero: Black ${s}% · captures ${e.pairs[0]}–${e.pairs[1]}`])}).catch(()=>{t===this.evalGen&&this.ctx.setDebugReadout([])})}promptAction(t){this.labelIndex=new Map(t.map((e,s)=>[e,s])),this.legalPoints=new Set(t.map(e=>to(e,this.size)).filter(e=>e!==null)),this.setInteractive(!0)}unmount(){this.unsubDebug?.(),this.unsubDebug=null}}function ro(){return new no}const Jt="poker-frontend-style",lo=`
.pk-root {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 100%;
  max-width: 940px;
  margin: 0 auto;
  user-select: none;
  --card-w: clamp(30px, 4.2vw, 44px);
}

/* ---------- the table ---------- */

.pk-table {
  position: relative;
  width: 100%;
  height: clamp(380px, 58vh, 560px);
}

.pk-felt {
  position: absolute;
  inset: 6% 2%;
  border-radius: 46% / 50%;
  border: 11px solid transparent;
  background:
    radial-gradient(ellipse 58% 42% at 50% 32%, rgba(255, 252, 230, 0.08), transparent 70%)
      padding-box,
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url(%23n)' opacity='0.05'/%3E%3C/svg%3E")
      padding-box,
    radial-gradient(ellipse at 50% 40%, #2f6043 0%, #224a31 55%, #122a1c 100%) padding-box,
    linear-gradient(155deg, #6f4c2c 0%, #3f2b18 42%, #543820 72%, #2c1c0d 100%) border-box;
  box-shadow:
    inset 0 0 90px rgba(0, 0, 0, 0.55),
    inset 0 3px 8px rgba(0, 0, 0, 0.5),
    0 0 0 1px rgba(0, 0, 0, 0.6),
    0 1px 0 rgba(255, 255, 255, 0.06),
    0 18px 44px rgba(0, 0, 0, 0.55);
}

.pk-felt::after {
  content: '';
  position: absolute;
  inset: 6%;
  border-radius: inherit;
  border: 1px solid rgba(212, 169, 92, 0.18);
}

/* ---------- center: board + pot ---------- */

.pk-center {
  position: absolute;
  left: 50%;
  top: 42%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  width: 64%;
  pointer-events: none;
}

.pk-board {
  display: flex;
  gap: 6px;
  min-height: calc(var(--card-w) * 1.4);
  align-items: center;
}

.pk-pot {
  font: 600 13px/1 ui-monospace, 'SF Mono', Menlo, monospace;
  color: #f4e3b8;
  letter-spacing: 0.04em;
  background: rgba(8, 22, 14, 0.62);
  border: 1px solid rgba(212, 169, 92, 0.4);
  border-radius: 999px;
  padding: 5px 14px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.45);
  white-space: nowrap;
}
.pk-pot b { color: #fff; }
.pk-street {
  font: 600 10px/1 system-ui, sans-serif;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: rgba(244, 227, 184, 0.65);
}

/* ---------- cards ---------- */

.pk-card {
  width: var(--card-w);
  height: calc(var(--card-w) * 1.4);
  border-radius: 5px;
  background: linear-gradient(160deg, #fff 0%, #f1f1ec 100%);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.45), inset 0 0 0 1px rgba(0, 0, 0, 0.08);
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 3px 4px;
  font: 700 calc(var(--card-w) * 0.42) / 1 'Georgia', serif;
  color: #1b1b1b;
}
.pk-card.red { color: #c01f2e; }
.pk-card .pk-rank { line-height: 0.9; }
.pk-card .pk-suit { font-size: calc(var(--card-w) * 0.5); align-self: flex-end; line-height: 0.8; }
.pk-card.deal-in { animation: pk-deal 0.32s ease-out backwards; }
@keyframes pk-deal {
  from { opacity: 0; transform: translateY(-18px) rotate(-6deg) scale(0.9); }
  to { opacity: 1; transform: none; }
}

.pk-card.back {
  background:
    repeating-linear-gradient(45deg, #243f7a 0 6px, #1d3260 6px 12px),
    #1d3260;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.45), inset 0 0 0 2px rgba(255, 255, 255, 0.14);
}
.pk-card.muck { opacity: 0.32; filter: grayscale(0.6); }
.pk-card.win-card { box-shadow: 0 0 0 2px #f4d35e, 0 3px 12px rgba(244, 211, 94, 0.6); }

/* ---------- seats ---------- */

.pk-seats { position: absolute; inset: 0; }

.pk-seat {
  position: absolute;
  transform: translate(-50%, -50%);
  width: clamp(130px, 17vw, 162px);
}
.pk-pod {
  position: relative;
  background: linear-gradient(180deg, rgba(20, 28, 22, 0.92), rgba(10, 16, 12, 0.92));
  border: 1px solid rgba(212, 169, 92, 0.28);
  border-radius: 12px;
  padding: 7px 8px 6px;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.5);
  text-align: center;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.pk-seat.turn .pk-pod {
  border-color: #f4d35e;
  box-shadow: 0 0 0 1px #f4d35e, 0 0 18px rgba(244, 211, 94, 0.4), 0 6px 18px rgba(0, 0, 0, 0.5);
}
.pk-seat.folded .pk-pod { opacity: 0.46; }
.pk-seat.winner .pk-pod {
  border-color: #66e08a;
  box-shadow: 0 0 0 1px #66e08a, 0 0 22px rgba(102, 224, 138, 0.5);
}

.pk-holes { display: flex; gap: 3px; justify-content: center; margin-bottom: 4px; min-height: calc(var(--card-w) * 1.4); }
.pk-seat .pk-card { --card-w: clamp(26px, 3.4vw, 36px); }

.pk-name {
  font: 600 12px/1.2 system-ui, sans-serif;
  color: #f1e6cb;
  display: flex; align-items: center; justify-content: center; gap: 4px;
}
.pk-stack {
  font: 600 11px/1.3 ui-monospace, Menlo, monospace;
  color: #bfe9cf;
}
.pk-stack .pk-bust { color: #e88; }
.pk-badge {
  display: inline-block;
  font: 700 8px/1 system-ui;
  background: #d4a95c; color: #201400;
  border-radius: 3px; padding: 2px 3px; margin-left: 2px;
  vertical-align: middle;
}
.pk-tag {
  position: absolute; top: -8px; right: -6px;
  font: 700 8px/1 system-ui; letter-spacing: 0.06em;
  padding: 2px 5px; border-radius: 999px;
}
.pk-tag.allin { background: #c0392b; color: #fff; }
.pk-tag.folded { background: #555; color: #ddd; }

/* a seat's current bet chips, pushed toward the pot */
.pk-bet {
  position: absolute;
  left: 50%; transform: translateX(-50%);
  font: 700 10px/1 ui-monospace, Menlo, monospace;
  color: #1a1208;
  background: #f4d35e;
  border: 1px solid #b8901f;
  border-radius: 999px;
  padding: 2px 7px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.4);
  white-space: nowrap;
}
.pk-seat.below .pk-bet { top: -16px; }
.pk-seat:not(.below) .pk-bet { bottom: -16px; }
.pk-bet:empty { display: none; }
/* The decoration holder doesn't lay out — its tag/dealer position off the pod. */
.pk-deco { display: contents; }

.pk-dealer {
  position: absolute;
  width: 18px; height: 18px; border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, #fff, #d8d2c4 70%, #b7b1a3);
  color: #222; font: 800 9px/18px system-ui; text-align: center;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(0,0,0,0.25);
}

/* float a chip delta over a seat */
.pk-float {
  position: absolute; left: 50%; top: 50%;
  transform: translate(-50%, -50%);
  font: 800 14px/1 ui-monospace, Menlo, monospace;
  pointer-events: none;
  animation: pk-rise 1.1s ease-out forwards;
}
.pk-float.win { color: #66e08a; text-shadow: 0 1px 4px rgba(0,0,0,0.7); }
.pk-float.lose { color: #ff7a7a; text-shadow: 0 1px 4px rgba(0,0,0,0.7); }
@keyframes pk-rise {
  0% { opacity: 0; transform: translate(-50%, -30%); }
  20% { opacity: 1; }
  100% { opacity: 0; transform: translate(-50%, -130%); }
}

/* ---------- banner ---------- */

.pk-banner {
  position: absolute; left: 50%; top: 14%;
  transform: translate(-50%, -50%) scale(0.9);
  background: rgba(8, 16, 11, 0.92);
  border: 1px solid rgba(212, 169, 92, 0.5);
  border-radius: 10px;
  padding: 8px 18px;
  font: 700 14px/1.2 system-ui, sans-serif;
  color: #f6eccf; text-align: center;
  opacity: 0; pointer-events: none;
  transition: opacity 0.22s, transform 0.22s;
  z-index: 6; max-width: 70%;
}
.pk-banner.show { opacity: 1; transform: translate(-50%, -50%) scale(1); }
.pk-banner.good { border-color: #66e08a; color: #d6ffe3; }
.pk-banner.bad { border-color: #e0664f; color: #ffdcd2; }

/* ---------- controls ---------- */

.pk-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  justify-content: center;
  min-height: 46px;
}
.pk-btn {
  font: 600 14px/1 system-ui, sans-serif;
  padding: 11px 18px;
  border-radius: 9px;
  border: 1px solid rgba(0, 0, 0, 0.25);
  background: linear-gradient(180deg, #f4f4f0, #e1e1d8);
  color: #1b1b1b;
  cursor: pointer;
  transition: transform 0.08s, filter 0.15s;
}
.pk-btn:hover:not(:disabled) { filter: brightness(1.05); }
.pk-btn:active:not(:disabled) { transform: translateY(1px); }
.pk-btn:disabled { opacity: 0.5; cursor: default; }
.pk-btn.fold { background: linear-gradient(180deg, #e9b0a6, #d98e80); color: #3a120a; }
.pk-btn.call { background: linear-gradient(180deg, #a9d9b6, #7cc18f); color: #0c2c16; }
.pk-btn.raise { background: linear-gradient(180deg, #f6dd8a, #e8c45a); color: #3a2a06; }

.pk-raiser {
  display: flex; align-items: center; gap: 8px;
  background: rgba(8, 16, 11, 0.06);
  border: 1px solid rgba(0,0,0,0.12);
  border-radius: 10px; padding: 6px 10px;
}
.pk-raiser input[type=range] { width: clamp(90px, 18vw, 180px); accent-color: #c79a3a; }
.pk-raiser .pk-amt {
  font: 700 13px/1 ui-monospace, Menlo, monospace;
  min-width: 46px; text-align: right; color: inherit;
}
.pk-quick { display: flex; gap: 4px; }
.pk-quick button {
  font: 600 11px/1 system-ui; padding: 5px 7px; border-radius: 6px;
  border: 1px solid rgba(0,0,0,0.18); background: #efe9da; cursor: pointer; color: #1b1b1b;
}
.pk-quick button:hover { background: #f7f1e2; }

.pk-fallback {
  white-space: pre-wrap;
  font: 13px/1.5 ui-monospace, Menlo, monospace;
  color: var(--fg, #222);
  padding: 12px;
}

/* The shell's opponent + difficulty picker, tucked into the pod right under the
   player's name. Compact and full-width so it reads as part of the box. */
.pk-pod .seat-slot {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 3px;
  width: 100%;
  margin: 3px 0 1px;
}
.pk-pod .seat-select,
.pk-pod .seat-level {
  width: 100%;
  font-size: 10px;
  padding: 2px 4px;
  border-radius: 5px;
}
`,co={c:"♣",d:"♦",h:"♥",s:"♠"},ho=new Set(["d","h"]);function te(n){return typeof n=="object"&&n!==null&&Array.isArray(n.players)}function nt(n){return typeof n=="object"&&n!==null&&typeof n.kind=="string"}function rt(n,t=""){const e=n[0]==="T"?"10":n[0],s=n[1];return`<div class="pk-card${ho.has(s)?" red":""}${t}">
    <span class="pk-rank">${e}</span>
    <span class="pk-suit">${co[s]??s}</span>
  </div>`}function I(n=""){return`<div class="pk-card back${n}"></div>`}function po(n,t){const e=Math.PI/180*(90+360*n/t),s=50+42*Math.cos(e),o=50+40*Math.sin(e);return{x:s,y:o,below:o>52}}function ee(n){return`pk-bank-${n.gameId}-${n.numSeats}`}class uo{ctx;seatsEl;centerEl;bannerEl;controlsEl;bankEl;view=null;boardShown=0;dead=!1;bank=0;seatsBuilt=!1;mount(t,e){if(this.ctx=e,!document.getElementById(Jt)){const s=document.createElement("style");s.id=Jt,s.textContent=lo,document.head.append(s)}this.bank=Number(sessionStorage.getItem(ee(e))??"0")||0,t.innerHTML=`
      <div class="pk-root">
        <div class="pk-table">
          <div class="pk-felt"></div>
          <div class="pk-center">
            <div class="pk-street"></div>
            <div class="pk-board"></div>
            <div class="pk-pot"></div>
          </div>
          <div class="pk-seats"></div>
          <div class="pk-banner"></div>
        </div>
        <div class="pk-bank"></div>
        <div class="pk-controls"></div>
      </div>`,this.seatsEl=t.querySelector(".pk-seats"),this.centerEl=t.querySelector(".pk-center"),this.bannerEl=t.querySelector(".pk-banner"),this.controlsEl=t.querySelector(".pk-controls"),this.bankEl=t.querySelector(".pk-bank"),this.renderBank()}render(t){if(!te(t.viewData)){const e=document.createElement("pre");e.className="pk-fallback",e.textContent=t.view,this.seatsEl.replaceChildren(e);return}this.view=t.viewData,this.boardShown=this.view.board.length,this.renderSeats(this.view),this.renderCenter(this.view),(t.toAct!==t.humanSeat||t.isOver)&&this.controlsEl.replaceChildren()}async animate(t,e){const s=this.ctx.animationScale(),o=t.data;if(nt(o)&&o.gameOver&&o.showdown){this.render(e),s>0&&!this.dead&&await this.playShowdown(o,s);return}const i=this.boardShown;this.render(e),s>0&&!this.dead&&te(e.viewData)&&(e.viewData.board.length>i?await this.dealBoard(i,s):nt(o)&&await this.flashAction(o,s)),nt(o)&&o.gameOver&&(this.render(e),s>0&&!this.dead&&await this.playShowdown(o,s))}promptAction(t){this.ctx.humanSeat<0||!this.view||this.renderControls(t)}unmount(){this.dead=!0}name(t){return t===this.ctx.humanSeat?"You":`Bot ${t+1}`}renderBank(){const t=this.bank,e=t>0?"+":"",s=t>0?"#3a8a52":t<0?"#b04a3a":"inherit";this.bankEl.innerHTML=`<span style="font:600 12px ui-monospace,Menlo,monospace;color:${s}">session: ${e}${t.toFixed(1)} bb</span>`}renderCenter(t){const e=this.centerEl.querySelector(".pk-street"),s=this.centerEl.querySelector(".pk-board"),o=this.centerEl.querySelector(".pk-pot");e.textContent=t.phase==="over"?"showdown":t.phase;const i=t.phase==="over"?this.winnerSeats(t):new Set,a=new Set;if(i.size){for(const r of t.players)if(i.has(r.seat)&&r.hole)for(const l of r.hole)a.add(l)}s.innerHTML=t.board.map(r=>rt(r,a.has(r)?" win-card":"")).join(""),o.innerHTML=`pot <b>${t.pot}</b>`}winnerSeats(t){const e=t.players.filter(o=>!o.folded&&o.net!==null),s=Math.max(...e.map(o=>o.net??-1/0));return new Set(e.filter(o=>(o.net??-1/0)>=0&&(o.net??0)===s).map(o=>o.seat))}buildSeats(t){if(this.seatsBuilt)return;this.seatsBuilt=!0;const e=t.seats,s=this.ctx.humanSeat>=0?this.ctx.humanSeat:0;this.seatsEl.innerHTML=t.players.map(o=>{const i=po((o.seat-s+e)%e,e);return`
        <div class="pk-seat${i.below?" below":""}" data-seat="${o.seat}"
             style="left:${i.x.toFixed(2)}%;top:${i.y.toFixed(2)}%">
          <div class="pk-pod">
            <span class="pk-deco"></span>
            <div class="pk-holes"></div>
            <div class="pk-name">${this.name(o.seat)}</div>
            <span class="seat-slot" data-seat="${o.seat}"></span>
            <div class="pk-stack"></div>
          </div>
          <span class="pk-bet"></span>
        </div>`}).join("")}renderSeats(t){this.buildSeats(t);const e=t.phase==="over"?this.winnerSeats(t):new Set;for(const s of t.players){const o=this.seatsEl.querySelector(`.pk-seat[data-seat="${s.seat}"]`);if(!o)continue;o.classList.toggle("folded",s.folded),o.classList.toggle("turn",s.toAct),o.classList.toggle("winner",e.has(s.seat));const i=s.folded?'<span class="pk-tag folded">FOLD</span>':s.allIn?'<span class="pk-tag allin">ALL-IN</span>':"",a=s.seat===t.button?'<span class="pk-dealer" style="right:-4px;bottom:-4px">D</span>':"";o.querySelector(".pk-deco").innerHTML=i+a,o.querySelector(".pk-holes").innerHTML=this.holesHtml(t,s);const r=s.stack<=0&&!s.allIn?" pk-bust":"";o.querySelector(".pk-stack").innerHTML=`<span class="${r}">${s.stack} bb</span>`,o.querySelector(".pk-bet").textContent=s.streetBet>0?String(s.streetBet):""}}holesHtml(t,e){return e.folded&&t.phase!=="over"?e.seat===t.viewer&&e.hole?e.hole.map(s=>rt(s," muck")).join(""):`${I(" muck")}${I(" muck")}`:e.hole?e.hole.map(s=>rt(s)).join(""):e.folded?"":`${I()}${I()}`}async dealBoard(t,e){const s=this.centerEl.querySelectorAll(".pk-board .pk-card");for(let o=t;o<s.length;o++)if(s[o].classList.add("deal-in"),await y(120*e),this.dead)return;await y(160*e)}async flashAction(t,e){const s=this.seatsEl.querySelector(`[data-seat="${t.seat}"] .pk-pod`);if(!s)return;const o=t.kind==="fold"?"folds":t.kind==="check"?"checks":t.kind==="call"?`calls ${t.amount}`:t.kind==="allin"?`all-in ${t.amount}`:`raises ${t.amount}`;this.banner(`${this.name(t.seat)} ${o}`,"info",!1),s.animate([{transform:"scale(1)"},{transform:"scale(1.05)"},{transform:"scale(1)"}],{duration:240*e,easing:"ease-out"});const i=t.seat!==this.ctx.humanSeat?2:1;await y((t.kind==="fold"||t.kind==="check"?240:380)*i*e),this.hideBanner()}async playShowdown(t,e){if(!t.showdown)return;const s=this.ctx.humanSeat;if(t.showdown.length>1&&(this.banner("Showdown","info",!0),await y(700*e),this.dead))return;for(const i of this.view?.players??[]){const a=i.net??0;if(Math.abs(a)<.001)continue;const r=this.seatsEl.querySelector(`[data-seat="${i.seat}"] .pk-pod`);if(!r)continue;const l=document.createElement("span");l.className=`pk-float ${a>0?"win":"lose"}`,l.textContent=`${a>0?"+":""}${a.toFixed(a%1===0?0:1)}`,r.append(l)}const o=this.view?.players.find(i=>i.seat===s)?.net??0;if(s>=0){const i=o>.001?`You win ${o.toFixed(1)} bb`:o<-.001?`You lose ${Math.abs(o).toFixed(1)} bb`:"You break even";this.banner(i,o>=0?"good":"bad",!0)}else{const i=Math.max(...this.view?.players.map(r=>r.net??0)??[0]),a=this.view?.players.find(r=>(r.net??0)===i);this.banner(`${this.name(a?.seat??0)} takes ${(i||0).toFixed(0)} bb`,"good",!0)}s>=0&&(this.bank+=o,sessionStorage.setItem(ee(this.ctx),String(this.bank)),this.renderBank()),await y(1500*e),!this.dead&&this.hideBanner()}banner(t,e,s){this.bannerEl.textContent=t,this.bannerEl.className=`pk-banner show ${e==="info"?"":e}`}hideBanner(){this.bannerEl.classList.remove("show")}submit(t){for(const e of this.controlsEl.querySelectorAll("button"))e.disabled=!0;this.ctx.submit(String(t))}renderControls(t){const e=[],s=[];let o=-1,i=0;t.forEach((a,r)=>{if(a==="fold")e.push(this.btn("Fold","fold",r));else if(a==="check")e.push(this.btn("Check","call",r));else if(a.startsWith("call")){const l=a.split(" ")[1]??"";e.push(this.btn(`Call ${l}`,"call",r))}else if(a.startsWith("raise to")){const l=Number(a.replace(/[^0-9]/g,""));s.push({idx:r,to:l,label:a})}else a.startsWith("all-in")&&(o=r,i=Number(a.replace(/[^0-9]/g,"")))}),this.controlsEl.replaceChildren(...e),(s.length||o>=0)&&this.controlsEl.append(this.raiserWidget(s,o,i))}btn(t,e,s){const o=document.createElement("button");return o.type="button",o.className=`pk-btn ${e}`,o.textContent=t,o.onclick=()=>this.submit(s),o}raiserWidget(t,e,s){const o=[...t].sort((p,g)=>p.to-g.to);if(e>=0&&!o.some(p=>p.to===s))o.push({idx:e,to:s,label:"all-in"});else if(e>=0){const p=o.findIndex(g=>g.to===s);o[p]={idx:e,to:s,label:"all-in"}}o.sort((p,g)=>p.to-g.to);const i=document.createElement("div");i.className="pk-raiser";const a=document.createElement("input");a.type="range",a.min="0",a.max=String(o.length-1),a.step="1",a.value=String(Math.min(o.length-1,Math.max(0,Math.floor(o.length/2))));const r=document.createElement("span");r.className="pk-amt";const l=document.createElement("button");l.type="button",l.className="pk-btn raise";const c=document.createElement("div");c.className="pk-quick";const d=p=>{a.value=String(p),h()};[["min",0],["½",Math.max(0,Math.round((o.length-1)*.25))],["pot",Math.max(0,Math.round((o.length-1)*.6))],["max",o.length-1]].forEach(([p,g])=>{const f=document.createElement("button");f.type="button",f.textContent=String(p),f.onclick=()=>d(g),c.append(f)});const h=()=>{const p=o[Number(a.value)],g=p.label==="all-in";r.textContent=`${p.to}`,l.textContent=g?`All-in ${p.to}`:`Raise to ${p.to}`};return a.oninput=h,l.onclick=()=>this.submit(o[Number(a.value)].idx),h(),i.append(c,a,r,l),i}}function fo(){return new uo}const lt=100,go={ArrowUp:"n",ArrowRight:"e",ArrowDown:"s",ArrowLeft:"w",w:"n",d:"e",s:"s",a:"w",W:"n",D:"e",S:"s",A:"w"},bo={n:"s",s:"n",e:"w",w:"e"},Me={n:[0,-1],e:[1,0],s:[0,1],w:[-1,0]},se={n:"up",e:"right",s:"down",w:"left"},mo=0,oe=170,xo=2,k=[{body:"#21c46a",bodyHi:"#5cf0a0",bodyLo:"#127a43",head:"#9bffc7",rim:"#063a22",glow:"rgba(45, 220, 120, 0.55)"},{body:"#3d8bff",bodyHi:"#7ec0ff",bodyLo:"#1f4fae",head:"#c4e2ff",rim:"#061634",glow:"rgba(70, 150, 255, 0.55)"},{body:"#ef9f27",bodyHi:"#ffd37a",bodyLo:"#9a5510",head:"#ffe6a8",rim:"#3b2105",glow:"rgba(255, 174, 55, 0.55)"},{body:"#b16cea",bodyHi:"#d8a7ff",bodyLo:"#67329b",head:"#ecd2ff",rim:"#26103a",glow:"rgba(190, 105, 255, 0.55)"}],ie=["Green","Blue","Amber","Violet"];function ae(n){if(!n||typeof n!="object")return null;const t=n;if(typeof t.side!="number"||t.coordinateSystem!=="battlesnake"||t.simultaneous!==!0||!Array.isArray(t.snakes)||t.snakes.length<2||t.snakes.length>4||!Array.isArray(t.food)||!Array.isArray(t.hazards))return null;for(const s of t.snakes)if(!s||!Array.isArray(s.cells)||s.alive&&s.cells.length===0)return null;const e=([s,o])=>[s,t.side-1-o];return{side:t.side,snakes:t.snakes.map(s=>({...s,cells:s.cells.filter(([o,i])=>o>=0&&o<t.side&&i>=0&&i<t.side).map(e)})),food:t.food.map(e),hazards:t.hazards.map(e),turn:t.turn??0,outcome:t.outcome??"ongoing"}}function yo(n,t){if(!n)return!1;for(let e=0;e<t.snakes.length;e++){if(n.snakes[e]?.alive!==t.snakes[e].alive)return!0;const[s,o]=n.snakes[e].cells[0]??[],[i,a]=t.snakes[e].cells[0]??[];if(!(s===void 0||o===void 0||i===void 0||a===void 0)&&(s!==i||o!==a))return!0}return!1}const ne="snake-frontend-style",vo=`
.snk-root {
  align-self: center;
  width: min(100%, 560px);
  display: flex;
  flex-direction: column;
  gap: 14px;
  user-select: none;
}
.snk-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: stretch;
  gap: 10px;
}
.snk-chip {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 8px 13px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--bg-inset);
  color: var(--text-dim);
  font-size: 0.86rem;
  white-space: nowrap;
  transition: border-color 0.25s, box-shadow 0.25s, color 0.25s, opacity 0.25s;
}
.snk-chip.snk-dead {
  opacity: 0.42;
  filter: grayscale(0.5);
}
.snk-dot {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  flex: none;
}
.snk-chip-0 .snk-dot {
  background: radial-gradient(circle at 35% 30%, ${k[0].bodyHi}, ${k[0].body} 70%, ${k[0].bodyLo});
  box-shadow: 0 0 9px ${k[0].glow};
}
.snk-chip-1 .snk-dot {
  background: radial-gradient(circle at 35% 30%, ${k[1].bodyHi}, ${k[1].body} 70%, ${k[1].bodyLo});
  box-shadow: 0 0 9px ${k[1].glow};
}
.snk-chip .snk-len {
  margin-left: auto;
  font-variant-numeric: tabular-nums;
  color: var(--text);
  font-weight: 700;
}
.snk-hp {
  position: relative;
  flex: none;
  width: 48px;
  height: 6px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.12);
  overflow: hidden;
}
.snk-hp-fill {
  position: absolute;
  inset: 0 auto 0 0;
  width: 100%;
  border-radius: 999px;
  transition: width 0.2s linear, background-color 0.3s;
}
.snk-chip-0 .snk-hp-fill { background: ${k[0].body}; }
.snk-chip-1 .snk-hp-fill { background: ${k[1].body}; }
.snk-hp.snk-hp-low .snk-hp-fill { background: #f85149; }
.snk-stage {
  position: relative;
  aspect-ratio: 1 / 1;
  border-radius: 16px;
  overflow: hidden;
  background:
    radial-gradient(120% 100% at 50% 0%, #14304f 0%, #0a1c30 45%, #050d18 100%);
  border: 1px solid rgba(120, 180, 255, 0.14);
  box-shadow:
    inset 0 1px 0 rgba(180, 220, 255, 0.07),
    inset 0 0 60px rgba(0, 0, 0, 0.55),
    0 10px 30px rgba(0, 0, 0, 0.35);
}
.snk-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  display: block;
}
.snk-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  background: rgba(4, 10, 20, 0.62);
  backdrop-filter: blur(2px);
  color: #eaf2ff;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.35s;
  text-align: center;
  padding: 12px;
}
.snk-overlay.snk-show {
  opacity: 1;
}
.snk-overlay.snk-start-gate {
  transition: none;
}
.snk-overlay b {
  font-size: 1.6rem;
  letter-spacing: 0.03em;
}
.snk-overlay small {
  color: rgba(200, 215, 235, 0.8);
}
.snk-hint {
  text-align: center;
  color: var(--text-dim);
  font-size: 0.8rem;
  min-height: 1.1em;
}
.snk-debug {
  position: absolute;
  top: 8px;
  left: 8px;
  padding: 6px 9px;
  border-radius: 8px;
  background: rgba(4, 10, 20, 0.62);
  border: 1px solid rgba(120, 180, 255, 0.18);
  color: #d7e6ff;
  font-family: var(--mono);
  font-size: 11px;
  line-height: 1.5;
  letter-spacing: 0.02em;
  white-space: pre;
  pointer-events: none;
  display: none;
}
.snk-debug.snk-debug-on {
  display: block;
}
`;function wo(){if(document.getElementById(ne))return;const n=document.createElement("style");n.id=ne,n.textContent=vo,document.head.append(n)}function ko(){try{return new URLSearchParams(window.location.search).has("snakeDebug")?!0:window.localStorage.getItem("snakeDebug")==="1"}catch{return!1}}class So{ctx;canvas;c2d;chips=[];lenEls=[];hpEls=[];hpFillEls=[];overlayEl;overlayTitleEl;overlaySubEl;hintEl;view=null;glide=null;side=11;cssSize=0;rafId=0;resizeObs=null;pendingLabels=null;turnBuffer=[];tickTimer=0;awaitingStart=!1;acknowledgedDir=null;mySeat=-1;wrapped=!1;foodPops=new Map;flashes=[];deathOrbs=[];deadSeats=new Set;prevScores=[];debugEl;showDebug=!1;mount(t,e){this.ctx=e,this.mySeat=e.humanSeat,this.awaitingStart=this.mySeat>=0,this.wrapped=["wrapped","wrapped-constrictor"].includes(String(e.opts.mode??"standard")),wo();const s=Array.from({length:e.numSeats},(i,a)=>`
          <div class="snk-chip snk-chip-${a}">
            <span class="snk-dot" style="background:${k[a].body}"></span>
            <span class="seat-slot" data-seat="${a}"></span>
            <span class="snk-hp"><span class="snk-hp-fill" style="background:${k[a].body}"></span></span>
            <span class="snk-len">3</span>
          </div>`).join("");t.innerHTML=`
      <div class="snk-root">
        <div class="snk-bar">${s}</div>
        <div class="snk-stage">
          <canvas class="snk-canvas"></canvas>
          <div class="snk-debug"></div>
          <div class="snk-overlay"><b></b><small></small></div>
        </div>
        <div class="snk-hint"></div>
      </div>`,this.canvas=t.querySelector(".snk-canvas"),this.c2d=this.canvas.getContext("2d"),this.chips=Array.from(t.querySelectorAll(".snk-chip")),this.lenEls=this.chips.map(i=>i.querySelector(".snk-len")),this.hpEls=this.chips.map(i=>i.querySelector(".snk-hp")),this.hpFillEls=this.chips.map(i=>i.querySelector(".snk-hp-fill")),this.prevScores=Array(e.numSeats).fill(0),this.overlayEl=t.querySelector(".snk-overlay"),this.overlayTitleEl=this.overlayEl.querySelector("b"),this.overlaySubEl=this.overlayEl.querySelector("small"),this.hintEl=t.querySelector(".snk-hint"),this.debugEl=t.querySelector(".snk-debug"),this.showDebug=ko(),this.debugEl.classList.toggle("snk-debug-on",this.showDebug);const o=t.querySelector(".snk-stage");this.mySeat>=0?(window.addEventListener("keydown",this.onKey,!0),o.addEventListener("touchstart",this.onTouchStart,{passive:!0}),o.addEventListener("touchmove",this.onTouchMove,{passive:!1}),o.addEventListener("touchend",this.onTouchEnd),this.hintEl.textContent="Choose a direction to start · arrow keys / WASD / swipe"):this.hintEl.textContent="Watching the bots play",this.resizeObs=new ResizeObserver(()=>this.resize(o)),this.resizeObs.observe(o),this.resize(o),this.loop(performance.now())}render(t){const e=ae(t.viewData);e&&(this.side=e.side,this.glide||(this.view=e),this.syncJuice(e),this.updateBar(e,t),this.updateOverlay(e,t))}async animate(t,e){const s=ae(e.viewData);if(!s)return;const o=this.latestKnown();this.syncJuice(s),this.updateBar(s,e),this.updateOverlay(s,e);const i=this.ctx.animationScale();if(this.side=s.side,i<=0){this.view=s,this.glide=null,this.acknowledgedDir=this.turnBuffer.at(-1)??null,await y(oe);return}if(!o||!yo(o,s)){this.acknowledgedDir=this.turnBuffer.at(-1)??null,this.view=s,this.glide=null;return}const a=performance.now();this.acknowledgedDir=this.turnBuffer.at(-1)??null;let r=s,l;const c=Eo(t.data);for(let d=0;d<s.snakes.length;d++){if(!o.snakes[d]?.alive||s.snakes[d].alive)continue;const h=c[d]??s.snakes[d].dir,p=$o(o,d,h,this.wrapped);r===s&&(r=qe(s)),r.snakes[d]={...r.snakes[d],cells:p.snakes[d].cells,dir:h},l=s,d===this.mySeat&&(this.acknowledgedDir=null)}this.glide={from:o,to:r,commitTo:l,start:a,dur:oe*i},await y(this.glide.dur),this.advanceGlide(performance.now())}latestKnown(){return this.glide?this.glide.to:this.view}promptAction(t){this.pendingLabels=t,!(this.mySeat<0)&&(this.awaitingStart||this.armTick())}armTick(){this.tickTimer||!this.pendingLabels||this.awaitingStart||(this.tickTimer=window.setTimeout(()=>{this.tickTimer=0,this.fireTick()},mo))}unmount(){cancelAnimationFrame(this.rafId),this.tickTimer&&clearTimeout(this.tickTimer),this.tickTimer=0,window.removeEventListener("keydown",this.onKey,!0),this.resizeObs?.disconnect(),this.resizeObs=null}fireTick(){if(!this.pendingLabels||this.mySeat<0)return;const t=this.currentHeading(),e=this.turnBuffer.shift()??null,s=e??t;this.acknowledgedDir=this.turnBuffer.at(-1)??(e?s:null);const o=se[s],i=this.pendingLabels.indexOf(o),a=this.pendingLabels;this.pendingLabels=null;const r=a.indexOf(se[t]);this.ctx.submit(String(i>=0?i:r>=0?r:0))}currentHeading(){return(this.glide?this.glide.to:this.view)?.snakes[this.mySeat]?.dir??"e"}onKey=t=>{if(this.mySeat<0||t.metaKey||t.ctrlKey||t.altKey)return;const e=t.target;if(e&&(e.tagName==="INPUT"||e.tagName==="TEXTAREA"||e.tagName==="SELECT"||e.tagName==="BUTTON"||e.isContentEditable))return;const s=go[t.key];s&&(t.preventDefault(),this.steer(s))};touchStart=null;onTouchStart=t=>{const e=t.changedTouches[0];this.touchStart={x:e.clientX,y:e.clientY}};onTouchMove=t=>{this.touchStart&&t.preventDefault()};onTouchEnd=t=>{if(!this.touchStart)return;const e=t.changedTouches[0],s=e.clientX-this.touchStart.x,o=e.clientY-this.touchStart.y;this.touchStart=null,!(Math.max(Math.abs(s),Math.abs(o))<22)&&this.steer(Math.abs(s)>Math.abs(o)?s>0?"e":"w":o>0?"s":"n")};steer(t){if(this.mySeat<0)return;const e=this.turnBuffer.at(-1)??this.currentHeading();if(t===e){this.awaitingStart&&(this.turnBuffer.push(t),this.acknowledgeInput(t),this.beginHumanPlay());return}const s=this.latestKnown()?.snakes[this.mySeat];!(s&&s.cells.length>0&&s.cells.every(([i,a])=>i===s.cells[0][0]&&a===s.cells[0][1])&&this.turnBuffer.length===0)&&t===bo[e]||this.turnBuffer.length>=xo||(this.turnBuffer.push(t),this.acknowledgeInput(t),this.awaitingStart&&this.beginHumanPlay(),this.armTick())}beginHumanPlay(){this.awaitingStart=!1,this.hintEl.textContent="Arrow keys / WASD / swipe to steer",this.updateStartOverlay(),this.armTick()}acknowledgeInput(t){this.acknowledgedDir=t}updateBar(t,e){for(let s=0;s<t.snakes.length;s++){const o=t.snakes[s];this.lenEls[s].textContent=String(o.score);const i=Math.max(0,Math.min(lt,o.health??lt)),a=o.alive?i/lt*100:0;this.hpFillEls[s].style.width=`${a}%`,this.hpEls[s].classList.toggle("snk-hp-low",o.alive&&i<=25),this.hpEls[s].title=`health ${i}`,this.chips[s].classList.toggle("snk-dead",!o.alive)}}updateOverlay(t,e){if(!e.isOver){this.updateStartOverlay();return}let s="Draw";const o=/^win(\d+)$/.exec(t.outcome)?.[1];o!==void 0&&(s=`${ie[Number(o)]} wins`),this.mySeat>=0&&t.outcome!=="draw"&&(s=t.outcome===`win${this.mySeat}`?"You win!":"You lose"),this.overlayTitleEl.textContent=s,this.overlaySubEl.textContent=`${t.snakes.map((i,a)=>`${ie[a]} ${i.score}`).join(" · ")} · turn ${t.turn}`,this.overlayEl.classList.add("snk-show")}updateStartOverlay(){if(!this.awaitingStart||this.mySeat<0){this.overlayEl.classList.remove("snk-show"),this.overlayEl.classList.contains("snk-start-gate")&&requestAnimationFrame(()=>this.overlayEl.classList.remove("snk-start-gate"));return}this.overlayTitleEl.textContent="Choose your first move",this.overlaySubEl.textContent="Arrow keys, WASD, or swipe to start",this.overlayEl.classList.add("snk-start-gate"),this.overlayEl.classList.add("snk-show")}syncJuice(t){const e=performance.now(),s=new Set(t.food.map(([o,i])=>`${o},${i}`));for(const o of s)this.foodPops.has(o)||this.foodPops.set(o,e);for(const o of this.foodPops.keys())s.has(o)||this.foodPops.delete(o);for(let o=0;o<t.snakes.length;o++){const i=t.snakes[o];if(i.score>this.prevScores[o]){const[a,r]=i.cells[0];this.flashes.push({x:a,y:r,born:e,dur:360,color:k[o].bodyHi})}this.prevScores[o]=i.score,!i.alive&&!this.deadSeats.has(o)&&(this.deadSeats.add(o),this.spawnDeath(o,i)),i.alive&&this.deadSeats.delete(o)}}spawnDeath(t,e){const s=performance.now(),o=k[t],i=Math.max(1,Math.floor(e.cells.length/22));for(let a=0;a<e.cells.length;a+=i){const[r,l]=e.cells[a],c=Math.random()*Math.PI*2,d=.4+Math.random()*1.4;this.deathOrbs.push({x:r,y:l,vx:Math.cos(c)*d,vy:Math.sin(c)*d,born:s+a*4,color:(a&4)===0?o.bodyHi:o.body})}}resize(t){const e=t.getBoundingClientRect(),s=Math.max(1,Math.round(Math.min(e.width,e.height))),o=Math.min(window.devicePixelRatio||1,2);this.cssSize=s,this.canvas.width=s*o,this.canvas.height=s*o,this.c2d.setTransform(o,0,0,o,0,0)}loop=t=>{this.draw(t),this.showDebug&&this.paintDebug(),this.rafId=requestAnimationFrame(this.loop)};paintDebug(){const t=`turn  ${this.view?.turn??0}`;this.debugEl.textContent!==t&&(this.debugEl.textContent=t)}draw(t){const e=this.c2d,s=this.cssSize;if(s<=0)return;const o=s/this.side;e.clearRect(0,0,s,s),this.drawGrid(o),this.advanceGlide(t);const i=this.glide?this.glide.to:this.view;if(i){this.drawHazards(i.hazards,o);for(const a of i.food)this.drawFood(a,o,t)}if(i)for(let a=0;a<i.snakes.length;a++)this.drawSnake(a,this.glideProgress(t),o);this.drawFlashes(o,t),this.drawDeathOrbs(o,t)}advanceGlide(t){this.glide&&t-this.glide.start>=this.glide.dur&&(this.view=this.glide.commitTo??this.glide.to,this.glide=null)}glideProgress(t){return this.glide?Math.min(1,(t-this.glide.start)/this.glide.dur):1}drawGrid(t){const e=this.c2d;e.save(),e.strokeStyle="rgba(130, 190, 255, 0.05)",e.lineWidth=1,e.beginPath();for(let s=1;s<this.side;s++){const o=Math.round(s*t)+.5;e.moveTo(o,0),e.lineTo(o,this.cssSize),e.moveTo(0,o),e.lineTo(this.cssSize,o)}e.stroke(),e.restore()}drawHazards(t,e){if(t.length===0)return;const s=this.c2d;s.save();for(const[o,i]of t){const a=o*e,r=i*e,l=s.createRadialGradient(a+e*.5,r+e*.5,0,a+e*.5,r+e*.5,e*.8);l.addColorStop(0,"rgba(173, 100, 255, 0.26)"),l.addColorStop(1,"rgba(82, 27, 126, 0.42)"),s.fillStyle=l,s.fillRect(a,r,e,e),s.strokeStyle="rgba(225, 184, 255, 0.22)",s.lineWidth=Math.max(1,e*.04),s.beginPath(),s.moveTo(a,r+e*.72),s.lineTo(a+e*.72,r),s.moveTo(a+e*.28,r+e),s.lineTo(a+e,r+e*.28),s.stroke()}s.restore()}drawFood(t,e,s){const o=this.c2d,i=(t[0]+.5)*e,a=(t[1]+.5)*e,r=(s-(this.foodPops.get(`${t[0]},${t[1]}`)??s))/240,l=r<1?.55+.45*le(r):1,c=1+.07*Math.sin(s/360),d=e*.32*l*c;o.save();const h=o.createRadialGradient(i,a,d*.4,i,a,d*2.6);h.addColorStop(0,"rgba(255, 120, 100, 0.4)"),h.addColorStop(1,"rgba(255, 80, 70, 0)"),o.fillStyle=h,o.beginPath(),o.arc(i,a,d*2.6,0,Math.PI*2),o.fill();const p=o.createRadialGradient(i-d*.32,a-d*.34,d*.1,i,a,d);p.addColorStop(0,"#fff2ec"),p.addColorStop(.4,"#ff9d86"),p.addColorStop(1,"#f0463c"),o.fillStyle=p,o.shadowColor="rgba(248, 81, 73, 0.85)",o.shadowBlur=e*.7,o.beginPath(),o.arc(i,a,d,0,Math.PI*2),o.fill(),o.shadowBlur=0,o.fillStyle="rgba(255, 240, 220, 0.9)";for(let g=0;g<2;g++){const f=s/600+g*Math.PI,b=i+Math.cos(f)*d*1.5,u=a+Math.sin(f)*d*1.5;o.beginPath(),o.arc(b,u,e*.045,0,Math.PI*2),o.fill()}o.restore()}drawSnake(t,e,s){const o=k[t],i=(this.glide?this.glide.to:this.view).snakes[t],a=this.glide?this.glide.from.snakes[t]:i;if(!i.alive&&!this.glide)return;const r=i.dir,l=t===this.mySeat?this.acknowledgedDir??r:i.dir,c=Mo(a.cells,i.cells,e,this.side,this.wrapped),d=this.wrapped?qo(c,this.side):c;if(d.length===0)return;if(this.showDebug){const f=t===0?"__snakeHead0":"__snakeHead1",b=window;b[f]={t:performance.now(),x:this.wrapped?J(d[0][0],this.side):d[0][0],y:this.wrapped?J(d[0][1],this.side):d[0][1],maxLink:d.slice(1).reduce((u,m,x)=>{const v=d[x];return Math.max(u,Math.hypot(m[0]-v[0],m[1]-v[1]))},0),dir:r,lookDir:l,alive:i.alive,len:i.cells.length}}const h=d.map(([f,b])=>[(f+.5)*s,(b+.5)*s]),p=s*.74,g=this.wrapped?Lo(d,this.side,this.cssSize):[[0,0]];for(const[f,b]of g){const u=h.map(([m,x])=>[m+f,x+b]);this.drawSnakeCopy(u,l,s,p,o,i.alive)}}drawSnakeCopy(t,e,s,o,i,a){const r=this.c2d;if(r.save(),r.globalAlpha=a?1:.4,r.lineJoin="round",r.lineCap="round",a&&(r.save(),r.shadowColor=i.glow,r.shadowBlur=s*.6,r.strokeStyle=i.glow,r.lineWidth=o,j(r,t,1),r.restore()),r.strokeStyle=i.rim,r.lineWidth=o+Math.max(1.5,s*.1),j(r,t,1),t.length>=2){const l=r.createLinearGradient(t[0][0],t[0][1],t[t.length-1][0],t[t.length-1][1]);l.addColorStop(0,i.bodyHi),l.addColorStop(.5,i.body),l.addColorStop(1,i.bodyLo),r.strokeStyle=l}else r.strokeStyle=i.body;r.lineWidth=o,j(r,t,.82),r.strokeStyle="rgba(255, 255, 255, 0.22)",r.lineWidth=Math.max(1,o*.3),j(r,t,1,-o*.22),r.restore(),this.drawHead(t[0],e,s,i,a)}drawHead(t,e,s,o,i){const a=this.c2d,[r,l]=t,c=s*.42,[d,h]=Me[e];a.save(),a.globalAlpha=i?1:.4,a.fillStyle=o.rim,a.beginPath(),a.arc(r,l,c+Math.max(1,s*.05),0,Math.PI*2),a.fill();const p=a.createRadialGradient(r-c*.34,l-c*.4,c*.1,r,l,c);p.addColorStop(0,o.head),p.addColorStop(.55,o.body),p.addColorStop(1,o.bodyLo),a.fillStyle=p,a.beginPath(),a.arc(r,l,c,0,Math.PI*2),a.fill();const g=s*.14,f=s*.18,b=s*.13,u=s*.07;for(const m of[-1,1]){const x=r+d*g+h*f*m,v=l+h*g+d*f*m;a.fillStyle="#f4f9ff",a.beginPath(),a.arc(x,v,b,0,Math.PI*2),a.fill(),a.fillStyle="#0a1424",a.beginPath(),a.arc(x+d*b*.4,v+h*b*.4,u,0,Math.PI*2),a.fill()}a.restore()}drawFlashes(t,e){if(this.flashes.length===0)return;const s=this.c2d;s.save(),s.globalCompositeOperation="lighter",this.flashes=this.flashes.filter(o=>{const i=(e-o.born)/o.dur;if(i>=1)return!1;const a=(o.x+.5)*t,r=(o.y+.5)*t,l=t*(.3+1.1*le(i));return s.globalAlpha=(1-i)*.7,s.strokeStyle=o.color,s.lineWidth=t*.12*(1-i),s.beginPath(),s.arc(a,r,l,0,Math.PI*2),s.stroke(),!0}),s.restore()}drawDeathOrbs(t,e){if(this.deathOrbs.length===0)return;const s=this.c2d;s.save(),s.globalCompositeOperation="lighter",this.deathOrbs=this.deathOrbs.filter(o=>{const i=e-o.born;if(i<0)return!0;const a=i/720;if(a>=1)return!1;const r=(o.x+.5+o.vx*a)*t,l=(o.y+.5+o.vy*a)*t,c=t*.3*(1-a*.5);s.globalAlpha=(1-a)*.85;const d=s.createRadialGradient(r,l,0,r,l,c*2);return d.addColorStop(0,"#ffffff"),d.addColorStop(.4,o.color),d.addColorStop(1,"rgba(0,0,0,0)"),s.fillStyle=d,s.beginPath(),s.arc(r,l,c*2,0,Math.PI*2),s.fill(),!0}),s.restore()}}function qe(n){return{...n,snakes:n.snakes.map(t=>({...t,cells:t.cells.map(([e,s])=>[e,s])})),food:n.food.map(([t,e])=>[t,e]),hazards:n.hazards.map(([t,e])=>[t,e])}}function Eo(n){if(!n||typeof n!="object")return[];const t=n.moves;if(!Array.isArray(t))return[];const e={up:"n",right:"e",down:"s",left:"w"};return t.map(s=>typeof s=="string"?e[s]:void 0)}function $o(n,t,e,s){const o=qe(n),i=o.snakes[t],[a,r]=Me[e];let l=i.cells[0][0]+a,c=i.cells[0][1]+r;s&&(l=J(l,n.side),c=J(c,n.side));const d=[[l,c],...i.cells.slice(0,-1)];if(l>=0&&l<n.side&&c>=0&&c<n.side&&n.food.some(([p,g])=>p===l&&g===c)&&d.length>0){const p=d[d.length-1];d.push([p[0],p[1]])}return o.snakes[t]={...i,cells:d,dir:e,score:d.length},o}function Mo(n,t,e,s,o){const i=[],a=t.length>n.length;for(let r=0;r<t.length;r++){const l=a?n[Math.max(0,r-1)]:n[Math.min(r,n.length-1)],c=t[r],d=o?Z(c[0],l[0],s):c[0],h=o?Z(c[1],l[1],s):c[1];i.push([re(l[0],d,e),re(l[1],h,e)])}return i}function qo(n,t){if(n.length===0)return[];const e=[[n[0][0],n[0][1]]];for(let s=1;s<n.length;s++){const o=e[s-1];e.push([Z(n[s][0],o[0],t),Z(n[s][1],o[1],t)])}return e}function Z(n,t,e){return n+Math.round((t-n)/e)*e}function J(n,t){return(n%t+t)%t}function Lo(n,t,e){const s=n.map(([c])=>c),o=n.map(([,c])=>c),i=(c,d)=>{const h=[],p=Math.floor((-d-1)/t),g=Math.ceil((t-c+1)/t);for(let f=p;f<=g;f++)d+f*t>=-1&&c+f*t<=t&&h.push(f*e);return h},a=i(Math.min(...s),Math.max(...s)),r=i(Math.min(...o),Math.max(...o)),l=[];for(const c of a)for(const d of r)l.push([c,d]);return l}function j(n,t,e,s=0){if(t.length===0)return;if(t.length===1){n.beginPath(),n.arc(t[0][0],t[0][1],n.lineWidth/2,0,Math.PI*2),n.fillStyle=n.strokeStyle,n.fill();return}if(s===0&&e>=1){n.beginPath(),n.moveTo(t[0][0],t[0][1]);for(let i=1;i<t.length;i++)n.lineTo(t[i][0],t[i][1]);n.stroke();return}const o=n.lineWidth;for(let i=0;i<t.length-1;i++){const a=i/(t.length-1),r=o*(1-(1-e)*a);let[l,c]=t[i],[d,h]=t[i+1];if(s!==0){const p=d-l,g=h-c,f=Math.hypot(p,g)||1,b=-g/f*s,u=p/f*s;l+=b,c+=u,d+=b,h+=u}n.lineWidth=r,n.beginPath(),n.moveTo(l,c),n.lineTo(d,h),n.stroke()}}function re(n,t,e){return n+(t-n)*e}function le(n){return 1-(1-n)*(1-n)}function Co(){return new So}const tt={10:"M258.962 29.14c-3.21.063-6.68 1.158-10.303 3.4-5.798 3.584-11.47 10.14-14.872 18.715-3.4 8.575-3.767 17.236-2.004 23.82 1.763 6.585 5.248 10.765 9.83 12.583 4.582 1.817 9.986 1.165 15.784-2.42 5.797-3.586 11.467-10.143 14.87-18.717 3.4-8.573 3.767-17.235 2.005-23.82-1.763-6.584-5.25-10.764-9.832-12.58-1.718-.683-3.55-1.018-5.478-.98zm83.428 36.012c-8.823 13.437-17.545 27.577-36.268 45.2l-1.615 1.52-2.137.596c-9.165 2.554-19 3.7-28.863 4.48-.54 5.822-1.76 11.47-3.492 16.946 14.814.187 28.827-.778 41.297-4.62 18.26-17.188 36.623-35.375 44.685-56.7l-13.607-7.422zm-133.135 31.58c-.172 0-.348.005-.527.02-1.248.117-2.846.825-5.022 2.126-11.898 12.29-14.007 33.196-.867 57.082 5.73 10.42 18.094 18.277 33.66 23.58 13.165 4.485 28.138 7.013 41.808 8.51l3.71-13.443c-6.24-1.808-14.008-3.65-22.142-6.082-11.813-3.53-24.576-8.437-34.355-18.432l-.343-.35-.303-.388c-5.047-6.43-5.557-13.842-5.6-22.496-.037-7.045.813-15.254 2.02-24.023-3.345-2.203-6.258-4.21-8.39-5.088-1.395-.575-2.445-1.008-3.65-1.018zm29.242 12.676c-.806 6.653-1.25 12.573-1.226 17.36.035 6.717 1.526 10.814 1.69 11.21 1.988 1.936 4.304 3.67 6.87 5.24 4.536-8.52 8.03-16.96 9.363-25.23l-16.696-8.58zm202.955 36.5c-38.698.407-97.748 25.527-127.31 46.75l21.93 26.664c23.08-25.157 50.67-42.282 78.29-49.248 28.02-7.068 56.45-3.25 78.33 13.597 1.784-1.8 2.504-3.56 2.694-5.432.305-3.01-.998-7.446-4.865-12.283-7.734-9.675-24.947-19.342-45.388-20.008-1.204-.04-2.43-.052-3.68-.04zm-259.16 8.734c-6.568 16.39-10.208 33.35-7.805 47.883 2.263 13.104 8.145 17.46 18.125 21.94 9.99 4.483 23.807 7.117 36.907 14.036l6.466 3.417-2.022 7.03c-5.67 19.72-14.65 38.776-28.312 56.41 8.66 10.85 24.016 19.95 40.84 29.016l6.138-10.893-17.434-29.078 4.14-4.914c10.787-12.804 16.836-38.882 20.882-55.754-7.692-7.7-25.79-16.08-42.803-28.55l-6.36-4.66 3.784-6.917c.52-.952 1.076-1.906 1.62-2.86-12.188-5.97-22.86-14.237-29.39-26.11-1.823-3.313-3.407-6.652-4.778-9.995zm260.135 29.922c-7.587-.093-15.517.908-23.664 2.963-26.07 6.576-53.767 24.346-75.986 51.377l-6.952 8.457-33.34-40.54c-1.77.288-3.426.55-5.21.842 9.908 16.11 16.95 31.17 25.693 40.888 5.715 6.352 11.743 10.584 20.38 12.742 8.025 2.006 18.66 2.104 33.263-1.126-2.695-7.855-2.26-16.004-.318-23.077 2.52-9.172 7.08-17.28 10.78-24.534l16.035 8.184c-2.167 4.244-4.322 8.392-6.112 12.324 5.102-.272 13.1-.745 20.61-1.246 8.984-.6 14.34-.982 17.38-1.197 1.703-3.57 3.562-6.718 5.905-9.497 3.513-4.17 8.686-7.383 14.256-8.108 1.392-.18 2.764-.257 4.13-.234 6.8.114 13.423 2.692 21.293 6.686 1.867-6.108 3.71-12.142 5.54-18.045-12.05-11.18-27.044-16.652-43.683-16.856zM277.92 210.86c-4.448.743-8.952 1.51-13.448 2.27 5.022 3.758 9.534 8.032 13.05 13.293l2.204 3.297-.948 3.852c-3.357 13.658-7.853 41.85-21.802 62.15l16.474 27.48-14.802 26.26c6.94-1.8 13.538-4.246 19.607-7.447l11-5.8 2.074 12.26c5.978 35.36-13.102 68.48-22.475 99.294 6.508 9.05 12.247 14.98 17.275 20.388 4.097 4.407 8.004 9.006 10.654 14.683h28.254c-1.863-9.857-5.227-15.497-17.834-26.75l-5.578-4.98 3.87-6.396c15.287-25.248 24.903-82.92 28.925-111.46l.92-6.526 6.503-1.092c10.253-1.72 16.833-5.857 22.162-11.826 5.33-5.97 9.233-14.076 12.258-23.29 1.334-4.067 2.448-8.31 3.478-12.62-16.163 3.494-29.45 3.68-40.754.855-12.36-3.088-21.957-9.893-29.4-18.164-12.948-14.39-20.65-32.733-31.665-49.73zm-108.337 19.524c-14.047 2.79-27.408 5.57-39.43 8.29-12.594 4.275-20.655 10.807-26.92 19.765-6.32 9.03-10.644 20.798-14.083 34.75-6.85 27.778-10.026 63.737-23.073 100.858l-17.465 65.434c3.948 3.74 7.722 6.273 11.717 9.855 3.488 3.13 6.69 7.757 8.58 13.504h23.447c.593-9.5-2.71-19.834-10.856-26.5l-5.18-4.235 26.142-62.953 5.35-.537c21.24-2.133 40.548-26.11 51.19-40.262l3.75-4.982 5.982 1.758c22.6 6.637 49.11 10.156 73.326 7.377-21.107-11.086-43.545-22.54-55.405-43.017l-3.175-5.483 4.09-4.84c12.58-14.873 20.877-30.868 26.6-47.8-8.712-3.39-18.953-6.002-28.93-10.48-5.508-2.472-10.967-5.897-15.657-10.5zm288.38.46c-.19-.005-.353.002-.493.02-1.118.146-1.406.185-2.817 1.858-1.352 1.604-3.214 4.944-5.216 9.785-1.614 5.982-.063 10.297 4.242 15.903 2.565 3.34 6.158 6.746 10.16 10.117 4.106-10.06 7.79-20.716 11.25-31.467-9.026-4.414-14.86-6.17-17.128-6.217zm-340.25 13.26c-.02 0-.32.076-.358.08.37.02.43-.086.36-.08zm313.353 4.83c-4.126.288-5.917.422-12.262.845-10.82.72-21.303 1.44-26.12 1.497.313 2.1 1.08 4.127 2.41 6.265l.532.854.33.95c.627 1.807 1.2 3.502 1.744 5.15 2.49 4.435 9.938 11.22 20.264 16.535 10.333 5.317 23.014 9.548 34.29 11.682 1.334-2.325 2.624-4.727 3.876-7.19-5.863-4.615-11.877-9.832-16.727-16.148-4.34-5.652-7.67-12.72-8.337-20.44zm-347.57 7.306c-5.687 2.507-11.285 5.163-16.114 7.74-6.134 3.273-10.633 6.54-12.37 8.03-4.556 15.79-6.52 30.088-11.78 44.884-4.72 13.276-12.487 26.58-26.66 40.11 11.624 10 23.234 16.21 37.47 15.316 8.202-29.22 11.38-58.08 17.63-83.44 2.888-11.71 6.472-22.743 11.825-32.64zm320.41 37.61c.224 5.742-.135 11.87-.944 19.608-.05 6.863 3.644 11.33 9.248 12.71 5.623 1.388 14.58-.295 24.822-12.622l.204-.244.22-.23c1.315-1.372 2.59-2.824 3.836-4.324-10.776-2.654-21.76-6.668-31.562-11.71-1.99-1.025-3.925-2.093-5.825-3.19zm-27.463 33.017c-6.36 6.718-14.816 11.936-25.082 14.857-.01.077-.026.17-.037.248l15.336 17.203-8.665 25.655-14.53 3.216c-1.706 8.07-3.678 16.264-5.95 24.278.423.08.85.17 1.27.246 10.442 1.892 19.172 1.915 26.915-1.684 17.515-18.86 28.118-31.565 31.95-53.44-1.87-7.903-10.127-20.008-19.666-29.138-.512-.49-1.026-.967-1.54-1.44zm-210.993 42.66c-5.333 6.792-12.013 14.51-20.16 21.164 16.307 31.444 34.568 62.892 57.48 92.173h26.753c-28.96-35.58-47.84-75.123-64.073-113.336z",9:"M27.084 18.248C-17.903 146.478 143.15 277.92 314.496 381.074c-4.645 13.767-5.585 27.628-3.394 40.635 4.44 26.355 20.974 48.997 42.86 62.425 21.884 13.428 49.776 17.57 75.645 5.765 25.87-11.804 48.69-38.923 62.737-84.654l-17.865-5.488c-13 42.318-32.806 64.094-52.63 73.14-19.825 9.047-40.69 5.998-58.116-4.693-17.425-10.69-30.75-29.095-34.205-49.6-3.455-20.507 2.232-43.318 24.677-65.218 20.743-20.24 32.068-41.615 30.434-61.24l-18.622 1.552c.74 8.89-4.35 22.76-16.684 37.486C222.057 230.8 73.838 128.622 27.084 18.248zm458.05 0C451.34 98.03 364.527 173.53 270.93 247.166c19.492 15.878 39.56 31.622 59.195 45.012 110.756-84.836 187.878-180.243 155.01-273.93zM127.58 292.146c-1.634 19.626 9.69 41 30.434 61.24 22.445 21.9 28.132 44.712 24.677 65.218-3.455 20.506-16.78 38.91-34.206 49.6-17.425 10.692-38.29 13.74-58.115 4.694-19.825-9.046-39.632-30.822-52.63-73.14l-17.865 5.488c14.046 45.73 36.867 72.85 62.736 84.654 25.87 11.805 53.763 7.663 75.648-5.765 21.885-13.428 38.42-36.07 42.86-62.426 2.19-13.005 1.25-26.863-3.393-40.628 13.986-8.42 27.905-17.022 41.648-25.803l-56.967-39.387c-6.55 5.103-13.063 10.2-19.52 15.293C150.55 316.46 145.46 302.59 146.2 293.7l-18.622-1.554zm18.1 73.614c-26.1 8.6-62.087 36.255-77.104 60.324 4.948 8.63 10.393 15.223 16.05 20.14 25.846-8.953 59.85-37.406 74.733-60.257-3.007-6.6-7.454-13.386-13.68-20.207zm220.863 0c-6.225 6.822-10.67 13.61-13.68 20.21 14.886 22.85 48.89 51.3 74.736 60.255 5.656-4.918 11.1-11.51 16.05-20.14-15.018-24.07-51.004-51.724-77.105-60.325z",8:"M460.283 403.386c0 9.601-14.032 27.556-26.827 32.24a32.948 32.948 0 0 1-10.44 1.37c-9.536 0-21.657-2.518-26.519-9.324-6.717-9.514-25.191-39.059-52.14-46.804-19.8-5.69-49.72-6.773-71.189-6.773-3.392 0-6.629 0-9.48.066q-7.436.1-15.048.1c-74.194 0-158.399-7.226-162.244-16.574 4.906 1.039 9.867 1.834 14.64 2.508a414.758 414.758 0 0 0 55.4 3.492h.121c20.165 0 52.859-1.315 85.597-8.155a53.035 53.035 0 1 0-18.695-61.565c-16.485 3.105-37.832 5.525-64.725 5.635h-2.21c-34.252 0-56.228-4.066-61.72-5.844a31.943 31.943 0 0 0-4.32-2.21c5.989-32.926 19.966-69.852 52.25-91.442 26.065-17.435 63.102-21.081 91.045-21.081 13.59 0 25.026.861 32.042 1.425 21.324 1.701 89.165 5.734 123.616 25.313 30.816 17.524 37.567 82.227 39.39 131.958.64 16.806 31.456 56.605 31.456 65.665zM97.18 381.089a75.41 75.41 0 0 1-14.242-4.364 28.948 28.948 0 0 1-7.458-4.696c-8.652 11.049-25.17 34.694-23.667 52.626.497 5.933 10.43 9.116 25.821 9.116 30.937 0 84.072-12.928 126.688-42.429l.254-.187c-24.01-.84-47.51-2.31-67.288-4.276-16.85-1.68-30.352-3.625-40.108-5.79zm170.01-218.251l4.564.353c7.845.597 18.783 1.437 31.147 2.807 5.403-18.938 15.469-18.982 15.734-22.772l.11-1.448a5.9 5.9 0 0 0-1.966-4.927c-18.65-15.922-30.286-52.074-28.43-93.762a5.68 5.68 0 0 0-5.104-6.077h-.387a5.712 5.712 0 0 0-5.524 5.281c-4.232 41.522-20.993 75.586-41.787 88.624a5.9 5.9 0 0 0-2.663 4.586l-.11 1.447c-.277 3.834 9.546 5.436 12.153 24.606 9.9.31 17.844.928 22.264 1.282zM69.194 332.462c3.79 2.873 11.48 7.072 34.34 10.287a396.66 396.66 0 0 0 52.947 3.315h.077a431.652 431.652 0 0 0 71.498-5.757 52.737 52.737 0 0 1-7.844-27.744 389.29 389.29 0 0 1-61.378 4.817h-2.276c-39.356 0-65.31-5.17-70.183-7.922a14.364 14.364 0 0 0-17.181 23.004zm204.009 15.469a35.357 35.357 0 1 0-35.357-35.357 35.357 35.357 0 0 0 35.368 35.302z",7:"M256.883 29.7L241.11 51.554l-23-14.06-6.202 26.224-26.63-4.193 4.308 26.604-26.21 6.317 14.165 22.93-21.794 15.86L177.61 147l-14.065 22.992 26.234 6.2-4.194 26.624 26.613-4.308 6.316 26.2 22.937-14.16 15.865 21.788 15.772-21.856 23 14.06 4.283-18.11 1.92-8.116 26.633 4.194-4.31-26.606 26.21-6.314-14.166-22.928 21.797-15.86-21.863-15.767 14.064-22.992-26.234-6.2 4.19-26.624-26.61 4.307-6.318-26.2-22.936 14.16-15.867-21.788zm-.252 51.68a49.657 49.64 0 0 1 49.657 49.64 49.657 49.64 0 0 1-49.656 49.638 49.657 49.64 0 0 1-49.655-49.638 49.657 49.64 0 0 1 49.656-49.64zm59.345 137.308l-8.082 34.164-29.96-18.315-9.747 13.504c11.734 82.04 18.1 163.835 54.654 247l16.553-66.185c10.51 13.815 27.52 26.056 49.656 33.092-31.075-77.557-42.77-158.987-54.714-240.37l-18.36-2.89zm-117.37.253l-19.76 3.2c-11.916 81.194-23.63 162.428-54.632 239.807 22.136-7.036 39.147-19.277 49.658-33.092l16.552 66.186c36.794-83.71 43.005-166.034 54.89-248.614l-8.595-11.8-29.88 18.442-8.232-34.127z",6:"M61.85 19.51c-15.08-.07-30.16 2.37-45.2 7.64C77.61 52.92 136.1 109.7 193.1 176.8l60.3-40.1C192.4 67.49 127.2 19.84 61.85 19.51zM442 32.08L109.9 252.7C90 265.9 70.45 268.9 53.86 267c-12.28-1.4-22.98-5.3-31.77-9.6-4.18 11.3-3.73 21-.16 27.5 4.67 8 14.54 13.6 35.43 10.7l22.8-3.2-14.01 18.1c-27.23 35.3-43.29 105 7.58 167.4 10.57 12.7 22.97 18 36.27 18.9 13.1 1 27-3 38.2-9.9 11.2-6.8 19.3-16.3 22.2-25.1 2.9-9 1.7-16.6-7.6-25.6-14.4-13.9-29.1-29.4-37-47.8 23.3-15.2 42.8-29.4 54.1-46.8 5.9-9.2 9.3-19.8 8.8-30.9-.6-11.1-4.8-22.3-12.4-34.2 95.2-68 199.2-130 296.4-197.68zM309.6 207.9l-59.5 39.2c26.7 34.1 53.2 69 79.6 102.4-14.7 12.4-28.6 17.5-37.5 16.7l-1.6 18.6c19.7 1.7 41-9.7 61.3-30.4 21.9-22.4 44.7-28.1 65.2-24.7 20.5 3.5 38.9 16.8 49.6 34.2 10.7 17.5 13.7 38.3 4.7 58.1-9 19.9-30.8 39.7-73.1 52.7l5.4 17.8c45.8-14 72.9-36.8 84.7-62.7 11.8-25.9 7.7-53.8-5.8-75.6-13.4-21.9-36-38.5-62.4-42.9-13-2.2-26.9-1.2-40.6 3.4-22.1-36.7-45.5-73-70-106.8zm-148.1 80.2c5.7 9.3 8.1 17 8.5 23.6.3 7-1.5 13.2-5.8 19.8-7.7 11.9-23.4 24.2-44.1 38.2-.3-2.2-.5-4.4-.6-6.8-1.1-23.6 11-48.7 42-74.8zm223 65c-6.6 3-13.4 7.4-20.2 13.6 8.6 26.1 36.2 62.1 60.3 77.1 8.6-4.9 15.2-10.3 20.1-16-8.9-25.8-37.4-59.9-60.2-74.7z",5:"M424.045 26.605l2.54 11.19 16.15-13.06zm-205.53 312.32a24.07 24.07 0 0 0 1.54-36l6.15-5.17c.72.73 1.41 1.5 2.07 2.31a32.09 32.09 0 0 1-45.62 44.74q2.13-3.69 4.14-7a24.12 24.12 0 0 0 31.72 1.12zm-18.1-19.47l10.32 7.66-15.53-2.3c.58-.81 1.14-1.58 1.69-2.29zm76.89-80.34l-31.33-38.65 15.26-12.34 31.33 38.65zm137.88-111.75l-31.33-38.65 15.27-12.37 31.33 38.64zm-41.83-26.18l25.91 32c-26.82 22.18-62.15 51.42-96.92 80.29l-26.75-33zm-252.76 239.74l10 1.16 32.83-65.86a28.13 28.13 0 0 0 41-38.11l4.52-3.67 26.53-21.52 27.14 33.48c-32.82 27.33-61.6 51.43-77.37 65-26.12 32.63-84.5 163.79-95.81 175.86-13.73-4.68-26.77-40.12-54-55.2 9.14-18.66 85.16-91.14 85.16-91.14zm33.05-113.81c1.5 2.11 5.69 5.81 8.38 5.81h.06c.35 0 1.29.17 2.48-1.65 7.15-11 18-16.41 25.26-12.77l-3.59 7.2c-1.77-.89-5.5.3-9.38 3.63l14.75 18.2a12.13 12.13 0 1 1-18.91.1l-6-7.44a10 10 0 0 1-4.4 1.12h-.23c-8.12 0-14.83-9.36-14.89-9.45zm259.84-158.46l46.5-37.73 11.06 13.68-46.49 37.69zm63.08 1.27l-37.46 30.31-5.73-7.07 37.42-30.34z",4:"M255.978 39.21C226.38 86.89 161.383 164.77 106 203.713V256.6c53.113-38.92 105.113-92.538 140.56-145.71L256 96.735l9.44 14.157c35.333 53 87.963 106.298 140.56 145.473V203.77C349.61 164.835 285.346 86.825 255.978 39.21zm0 108.406C226.38 195.293 161.383 273.174 106 312.116v52.89C159.113 326.09 211.113 272.47 246.56 219.3l9.44-14.16 9.44 14.16c35.333 53 87.963 106.298 140.56 145.473v-52.597c-56.39-38.937-120.654-116.944-150.022-164.557zm0 107.782C226.38 303.075 161.383 380.956 106 419.898v52.89c53.113-38.918 105.113-92.536 140.56-145.707l9.44-14.16 9.44 14.16c35.333 53 87.963 106.298 140.56 145.473v-52.597c-56.39-38.938-120.654-116.945-150.022-164.558z",3:"M185.6 29.02c-17.8.64-35.3 16.49-42.9 38.77-9.4 27.43-.7 55.21 19.4 62.11 20.1 6.8 44-9.9 53.3-37.3 9.4-27.42.7-55.21-19.4-62.08-3.3-1.12-6.8-1.63-10.4-1.5zm156 19.6c-10.3.1-22 .98-35.2 2.86 36.5 8.49 67.8 21.18 92.8 39.24L298.6 207.1c7.9 5.8 9.1 7.7 16.8 13.8l101.3-115.7c22.4 21.4 37.7 48.7 44.2 82.9 3.8-35.9-2.7-76-27.4-102.08 7.4-7.89 15.4-15.69 5.6-24.17-12.9-7.89-14.9-.32-23.1 9.25-17-12.63-37.4-22.75-74.4-22.48zM115.1 145.5C69.21 171.2 19.21 253.6 84.86 381c11.29 22-34.56 65.6-60.28 90.3l43.95 6.2c29.82-31.7 52.97-60.7 63.17-98.1 1.6-5.6-16.8-34.7-27.8-62.1 27.9 14.1 73.1 58.4 70.5 76.6-5.1 35.3-15.5 53.2-30.7 84.9l46.7 4.2c11.2-24.1 18.9-52.5 25.6-97.3 2-13.8-23.8-47-61.8-85.1-13.7-13.8-27.1-56.4 10.7-107.9 47.7 18.3 78.8 43.9 118.4 67.6l18.4-21.7c-38.5-30-79.5-65-129.8-88-17.9-8.2-39.9-14.6-56.8-5.1zm345.8 42.6c-16.6 2.1-92 37.8-125.1 56.3l69.3-6.1-86.7 100.2L421 264.2l-14.7 127.2 47.2-131.7 33.9 80.9c-7.2-84.5-11.2-109.1-26.5-152.5z",2:"M79.624 33.606L52.851 113.93l16.636 6.049a72.511 72.511 0 0 1 27.055-9.02V74.407l146 54.75v13.237h96v50h-69.91l108.203 39.345-19.818-84.07-18.657-68.404zm322.744 20.022l-17.652 3.531 12.99 64.947 18.72 1.813zm-287.826 46.754v74.012h110V141.63zm-18 28.75c-26.165 4.27-46 26.847-46 54.262 0 30.482 24.518 55 55 55 27.414 0 49.992-19.835 54.262-46H96.542zm276.645 8.683l28.04 118.96 60.231 24.093-11.295-56.474h-37.568l-22.24-84.916zm36.31 3.704l16.992 64.875h20.074l-3.931-19.655-8.598-42.99zm-166.955 18.875v14h78v-14zm-51.271 32l42.066 42.066 149.799 24.014-.772-3.278-3.09-3.414-163.316-59.388zm-15.303 10.152c-6.032 22.105-22.226 40.075-43.22 48.557l73.448 83.943 54.924 9.152 6.4-19.203-41.195-72.09zm109.508 125.508l-19.702 59.105 49.254-29.553zm-79.873 25.14L172.883 399l55.636 16.691 13.156-13.156 13.681-41.049zm-34.592 64.037l-41.907 27.938-11.074 33.225h202.512v-16.305z",S:"M218 19c-1 0-2.76.52-5.502 3.107-2.742 2.589-6.006 7.021-9.191 12.76-6.37 11.478-12.527 28.033-17.666 45.653-4.33 14.844-7.91 30.457-10.616 44.601 54.351 24.019 107.599 24.019 161.95 0-2.706-14.144-6.286-29.757-10.616-44.601-5.139-17.62-11.295-34.175-17.666-45.653-3.185-5.739-6.45-10.171-9.191-12.76C296.76 19.52 295 19 294 19c-6.5 0-9.092 1.375-10.822 2.85-1.73 1.474-3.02 3.81-4.358 7.34-1.338 3.53-2.397 8.024-5.55 12.783C270.116 46.73 263.367 51 256 51c-7.433 0-14.24-4.195-17.455-8.988-3.214-4.794-4.26-9.335-5.576-12.881-1.316-3.546-2.575-5.867-4.254-7.315C227.035 20.37 224.5 19 218 19zm-46.111 124.334c-1.41 9.278-2.296 17.16-2.57 22.602 6.61 5.087 17.736 10.007 31.742 13.302C217.18 183.031 236.6 185 256 185s38.82-1.969 54.94-5.762c14.005-3.295 25.13-8.215 31.742-13.302-.275-5.443-1.161-13.324-2.57-22.602-55.757 23.332-112.467 23.332-168.223 0zM151.945 155.1c-19.206 3.36-36.706 7.385-51.918 11.63-19.879 5.548-35.905 11.489-46.545 16.57-5.32 2.542-9.312 4.915-11.494 6.57-.37.28-.247.306-.445.546.333.677.82 1.456 1.73 2.479 1.973 2.216 5.564 4.992 10.627 7.744 10.127 5.504 25.944 10.958 45.725 15.506C139.187 225.24 194.703 231 256 231s116.813-5.76 156.375-14.855c19.78-4.548 35.598-10.002 45.725-15.506 5.063-2.752 8.653-5.528 10.627-7.744.91-1.023 1.397-1.802 1.73-2.479-.198-.24-.075-.266-.445-.547-2.182-1.654-6.174-4.027-11.494-6.568-10.64-5.082-26.666-11.023-46.545-16.57-15.212-4.246-32.712-8.272-51.918-11.631.608 5.787.945 10.866.945 14.9v3.729l-2.637 2.634c-10.121 10.122-25.422 16.191-43.302 20.399C297.18 200.969 276.6 203 256 203s-41.18-2.031-59.06-6.238c-17.881-4.208-33.182-10.277-43.303-20.399L151 173.73V170c0-4.034.337-9.113.945-14.9zm1.094 88.205C154.558 308.17 200.64 359 256 359c55.36 0 101.442-50.83 102.96-115.695a748.452 748.452 0 0 1-19.284 2.013c-1.33 5.252-6.884 25.248-15.676 30.682-13.61 8.412-34.006 7.756-48 0-7.986-4.426-14.865-19.196-18.064-27.012-.648.002-1.287.012-1.936.012-.65 0-1.288-.01-1.936-.012-3.2 7.816-10.078 22.586-18.064 27.012-13.994 7.756-34.39 8.412-48 0-8.792-5.434-14.346-25.43-15.676-30.682a748.452 748.452 0 0 1-19.285-2.013zM137.4 267.209c-47.432 13.23-77.243 32.253-113.546 61.082 42.575 4.442 67.486 21.318 101.265 48.719l16.928 13.732-21.686 2.211c-13.663 1.393-28.446 8.622-39.3 17.3-5.925 4.738-10.178 10.06-12.957 14.356 44.68 5.864 73.463 10.086 98.011 20.147 18.603 7.624 34.81 18.89 53.737 35.781l5.304-23.576c-1.838-9.734-4.134-19.884-6.879-30.3-5.12-7.23-9.698-14.866-13.136-22.007C201.612 397.326 199 391 199 384c0-3.283.936-6.396 2.428-9.133a480.414 480.414 0 0 0-6.942-16.863c-29.083-19.498-50.217-52.359-57.086-90.795zm237.2 0c-6.87 38.436-28.003 71.297-57.086 90.795a480.521 480.521 0 0 0-6.942 16.861c1.493 2.737 2.428 5.851 2.428 9.135 0 7-2.612 13.326-6.14 20.654-3.44 7.142-8.019 14.78-13.14 22.01-2.778 10.547-5.099 20.82-6.949 30.666l5.14 23.42c19.03-17.01 35.293-28.338 53.974-35.994 24.548-10.06 53.33-14.283 98.011-20.147-2.78-4.297-7.032-9.618-12.957-14.355-10.854-8.679-25.637-15.908-39.3-17.3l-21.686-2.212 16.928-13.732c33.779-27.4 58.69-44.277 101.265-48.719-36.303-28.829-66.114-47.851-113.546-61.082zM256 377c-8 0-19.592.098-28.234 1.826-4.321.864-7.8 2.222-9.393 3.324-1.592 1.103-1.373.85-1.373 1.85s1.388 6.674 4.36 12.846c2.971 6.172 7.247 13.32 11.964 19.924 4.717 6.604 9.925 12.699 14.465 16.806 4.075 3.687 7.842 5.121 8.211 5.377.37-.256 4.136-1.69 8.21-5.377 4.54-4.107 9.749-10.202 14.466-16.806 4.717-6.605 8.993-13.752 11.965-19.924C293.612 390.674 295 385 295 384s.22-.747-1.373-1.85c-1.593-1.102-5.072-2.46-9.393-3.324C275.592 377.098 264 377 256 377zm0 61.953c-.042.03-.051.047 0 .047s.042-.018 0-.047zm-11.648 14.701L235.047 495h41.56l-9.058-41.285C264.162 455.71 260.449 457 256 457c-4.492 0-8.235-1.316-11.648-3.346z",B:"M135.25 38.156c-16.082.46-32.345 7.235-46.47 17.407-17.216 12.4-31.534 30.2-37.31 50.687-5.78 20.488-1.95 44.032 16.155 63.406 14.573 15.595 19.996 29.328 20.563 40.5.566 11.173-3.554 20.304-10.376 27.406-13.643 14.206-37.278 17.995-50.5 6.094l-12.5 13.906c22.224 20.005 56.61 13.645 76.47-7.03 9.93-10.34 16.43-24.836 15.593-41.313-.836-16.478-8.83-34.407-25.594-52.345C67.18 141.782 65.16 126.6 69.47 111.312 73.78 96.025 85.484 80.97 99.72 70.72c14.233-10.253 30.704-15.365 43.218-13.44 9.566 1.474 17.565 6.055 23.062 17.44l15.938-9.19c-8.362-15.432-21.594-24.476-36.157-26.718-2.42-.372-4.866-.596-7.31-.656-1.07-.026-2.148-.03-3.22 0zM243.5 51.563l-120.125 69.374 24.906 43.157c15.03-18.11 33.446-33.898 55-46.344 20.615-11.903 42.444-19.803 64.595-23.938L243.5 51.563zm60.03 57.406c-1.026.01-2.065.034-3.092.06-29.894.803-60.05 8.877-87.813 24.907-88.84 51.298-119.255 164.55-68.03 253.282 51.222 88.73 164.505 119.013 253.343 67.717 88.837-51.295 119.223-164.55 68-253.28-34.666-60.05-97.713-93.346-162.407-92.688z",F:"M247 26v163.2c-15 .8-28.5 3.3-39.4 7.4-6.5 2.4-12.1 5.4-16.5 9.3-4.5 4-8.1 9.5-8.1 16.1 0 6.6 3.6 12.1 8.1 16.1 4.4 3.9 10 6.9 16.5 9.3 13 4.9 29.8 7.6 48.4 7.6 18.6 0 35.4-2.7 48.4-7.6 6.5-2.4 12.1-5.4 16.5-9.3 4.5-4 8.1-9.5 8.1-16.1 0-6.6-3.6-12.1-8.1-16.1-4.4-3.9-10-6.9-16.5-9.3-10.9-4.1-24.4-6.6-39.4-7.4V26zm38 .99v70.02L378.4 62zM247 207.3v29.4c-13.1-.7-24.8-3-33.1-6.2-5.1-1.9-8.9-4.1-10.9-5.9-2-1.8-2-2.5-2-2.6 0-.1 0-.8 2-2.6s5.8-4 10.9-5.9c8.3-3.2 20-5.5 33.1-6.2zm18 0c13.1.7 24.8 3 33.1 6.2 5.1 1.9 8.9 4.1 10.9 5.9 2 1.8 2 2.5 2 2.6 0 .1 0 .8-2 2.6s-5.8 4-10.9 5.9c-8.3 3.2-20 5.5-33.1 6.2zm-9 70.7L96 358l128-16-32 144h128l-32-144 128 16z",back:"M19.75 14.438c59.538 112.29 142.51 202.35 232.28 292.718l3.626 3.75.063-.062c21.827 21.93 44.04 43.923 66.405 66.25-18.856 14.813-38.974 28.2-59.938 40.312l28.532 28.53 68.717-68.717c42.337 27.636 76.286 63.646 104.094 105.81l28.064-28.06c-42.47-27.493-79.74-60.206-106.03-103.876l68.936-68.938-28.53-28.53c-11.115 21.853-24.413 42.015-39.47 60.593-43.852-43.8-86.462-85.842-130.125-125.47-.224-.203-.432-.422-.656-.625C183.624 122.75 108.515 63.91 19.75 14.437zm471.875 0c-83.038 46.28-154.122 100.78-221.97 161.156l22.814 21.562 56.81-56.812 13.22 13.187-56.438 56.44 24.594 23.186c61.802-66.92 117.6-136.92 160.97-218.72zm-329.53 125.906l200.56 200.53c-4.36 4.443-8.84 8.793-13.405 13.032L148.875 153.53l13.22-13.186zm-76.69 113.28l-28.5 28.532 68.907 68.906c-26.29 43.673-63.53 76.414-106 103.907l28.063 28.06c27.807-42.164 61.758-78.174 104.094-105.81l68.718 68.717 28.53-28.53c-20.962-12.113-41.08-25.5-59.937-40.313 17.865-17.83 35.61-35.433 53.157-52.97l-24.843-25.655-55.47 55.467c-4.565-4.238-9.014-8.62-13.374-13.062l55.844-55.844-24.53-25.374c-18.28 17.856-36.602 36.06-55.158 54.594-15.068-18.587-28.38-38.758-39.5-60.625z"},ce=260,Ao=520,To=340,O={10:"Marshal",9:"General",8:"Colonel",7:"Major",6:"Captain",5:"Lieutenant",4:"Sergeant",3:"Miner",2:"Scout",S:"Spy",B:"Bomb",F:"Flag"},Y=["10","9","8","7","6","5","4","3","2","S","B","F"],zo={S:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,F:10,B:11},Do={C:"S",D:"2",E:"3",F:"4",G:"5",H:"6",I:"7",J:"8",K:"9",L:"10",M:"F",B:"B"};function Po(){return new Bo}class Bo{ctx;root;squaresEl;piecesEl;trayEl;fallbackEl;flipped=!1;view=null;moves=new Map;placements=new Map;selected=null;myTurn=!1;drag=null;editor=null;editorSel=null;deployQueue=null;mount(t,e){this.ctx=e,this.flipped=e.humanSeat===1,Ro();const s=this.flipped?0:1,o=a=>`<span class="sg-side-tag sg-side-${a===0?"red":"blue"}">${a===0?"Red":"Blue"}</span>`;t.innerHTML=`
      <div class="sg-root">
        <div class="sg-bar sg-bar-top">
          ${o(s)}
          <span class="seat-slot" data-seat="${s}"></span>
          <div class="sg-tray sg-tray-top" title="Captured pieces"></div>
        </div>
        <div class="sg-stage">
          <div class="sg-board">
            <div class="sg-squares"></div>
            <div class="sg-pieces"></div>
          </div>
          <div class="sg-supply" hidden></div>
        </div>
        <div class="sg-bar sg-bar-bottom">
          ${o(1-s)}
          <span class="seat-slot" data-seat="${1-s}"></span>
          <div class="sg-tray sg-tray-bottom" title="Captured pieces"></div>
        </div>
        <pre class="sg-fallback" hidden></pre>
      </div>`,this.root=t.querySelector(".sg-root"),this.squaresEl=t.querySelector(".sg-squares"),this.piecesEl=t.querySelector(".sg-pieces"),this.trayEl=t.querySelector(".sg-supply"),this.fallbackEl=t.querySelector(".sg-fallback"),t.querySelector(".sg-board").insertAdjacentHTML("afterbegin",Go),this.buildSquares();const i=t.querySelector(".sg-board");i.addEventListener("pointerdown",a=>this.onPointerDown(a)),i.addEventListener("pointermove",a=>this.onPointerMove(a)),i.addEventListener("pointerup",a=>this.onPointerUp(a)),i.addEventListener("pointercancel",()=>this.cancelDrag())}buildSquares(){const t=document.createDocumentFragment();for(let e=0;e<10;e++)for(let s=0;s<10;s++){const o=this.cellAt(s,e),i=document.createElement("div");i.className="sg-sq",i.dataset.cell=String(o),_o.has(o)&&i.classList.add("sg-lake"),t.append(i)}this.squaresEl.replaceChildren(t)}cellAt(t,e){const s=this.flipped?9-t:t;return(this.flipped?e:9-e)*10+s}xyOf(t){const e=Math.floor(t/10),s=t%10;return{x:this.flipped?9-s:s,y:this.flipped?e:9-e}}squareEl(t){return this.squaresEl.querySelector(`[data-cell="${t}"]`)}pieceEl(t){return this.piecesEl.querySelector(`[data-cell="${t}"]`)}slotOfCell(t){const e=this.ctx.humanSeat===1?99-t:t;return e>=0&&e<40?e:-1}cellOfSlot(t){return this.ctx.humanSeat===1?99-t:t}flagAllowed(t){return t%10>=5}render(t){const e=Oo(t.viewData);if(!e){this.fallbackEl.hidden=!1,this.fallbackEl.textContent=t.view;return}this.fallbackEl.hidden=!0,this.view=e,this.syncPieces(e),this.syncTrays(e),this.syncDeployPanel(e),this.syncHighlights(e)}syncPieces(t){const e=document.createDocumentFragment();if(t.cells.forEach((s,o)=>{s===null||s==="~"||e.append(this.makePiece(o,s))}),this.editor){const s=this.ctx.humanSeat;this.editor.forEach((o,i)=>{if(o===null)return;const a=document.createElement("div");a.className=`sg-piece sg-edit sg-${s===0?"red":"blue"}`,a.title=O[o],a.innerHTML=H(o,s),this.place(a,this.cellOfSlot(i)),e.append(a)})}this.piecesEl.replaceChildren(e)}makePiece(t,e){const s=document.createElement("div");s.className=`sg-piece sg-${e.o===0?"red":"blue"}`,s.dataset.cell=String(t);const o=this.ctx.humanSeat!==0&&this.ctx.humanSeat!==1;return e.r===null?(s.classList.add("sg-hidden"),e.m&&s.classList.add("sg-has-moved"),s.title=e.m?"Hidden enemy (has moved)":"Hidden enemy"):e.v&&(o||e.o===this.ctx.humanSeat)?(s.classList.add("sg-known"),s.title=`${O[e.r]} (${o?"revealed":"revealed to the enemy"})`):o&&e.m?(s.classList.add("sg-has-moved"),s.title=`${O[e.r]} (has moved, rank still hidden)`):s.title=O[e.r]??e.r,s.innerHTML=H(e.r,e.o),this.place(s,t),s}place(t,e){const{x:s,y:o}=this.xyOf(e);t.style.transform=`translate(${s*100}%, ${o*100}%)`,t.dataset.cell=String(e)}syncTrays(t){const e=this.flipped?1:0,s=t.captured??[[],[]];this.fillTray(".sg-tray-bottom",s[e],e),this.fillTray(".sg-tray-top",s[1-e],1-e)}fillTray(t,e,s){const o=this.root.querySelector(t),i=[...e].sort((a,r)=>Y.indexOf(a)-Y.indexOf(r));o.replaceChildren(...i.map(a=>{const r=document.createElement("span");return r.className=`sg-tray-piece sg-${s===0?"red":"blue"}`,r.title=`${O[a]} (captured)`,r.innerHTML=H(a,s),r}))}editorCounts(){const t=new Map,e=this.view?.supply??[];for(const s of Y)t.set(s,e[zo[s]]??0);for(const s of this.editor??[])s!==null&&t.set(s,(t.get(s)??0)-1);return t}firstOpenSlot(t){return t.deployed?.[this.ctx.humanSeat]??0}syncDeployPanel(t){if(!(t.phase==="deploy"&&t.supply!==null&&t.toAct===this.ctx.humanSeat&&this.deployQueue===null)){this.editor=null,this.editorSel=null,this.trayEl.hidden=!0,this.root.classList.remove("sg-deploying");return}this.editor||(this.editor=new Array(40).fill(null),this.shuffleEditor()),this.trayEl.hidden=!1,this.root.classList.add("sg-deploying");const s=this.ctx.humanSeat===1?1:0,o=this.editorCounts(),i=[...o.values()].reduce((d,h)=>d+h,0),a=document.createElement("div");a.className="sg-supply-grid",a.append(...Y.map(d=>{const h=o.get(d)??0,p=document.createElement("button");return p.type="button",p.className=`sg-supply-btn sg-${s===0?"red":"blue"}`,this.editorSel===d&&p.classList.add("sg-supply-sel"),p.disabled=h===0&&this.editorSel!==d,p.title=`${O[d]} — ${h} to place`,p.innerHTML=`${H(d,s)}<span class="sg-supply-count">${h}</span>`,p.onclick=()=>{this.editorSel=this.editorSel===d?null:h>0?d:null,this.refreshEditor()},p}));const r=document.createElement("div");r.className="sg-supply-actions";const l=(d,h,p,g=!1)=>{const f=document.createElement("button");return f.type="button",f.className=h,f.textContent=d,f.onclick=p,f.disabled=g,f};r.append(l("Shuffle","sg-action",()=>{this.shuffleEditor(),this.refreshEditor()}),l("Clear","sg-action",()=>{this.editor=new Array(40).fill(null),this.editorSel=null,this.refreshEditor()}),l("Start battle","sg-action sg-action-go",()=>this.confirmDeployment(),i>0));const c=document.createElement("div");c.className="sg-supply-hint",c.textContent=i>0?`Arrange your army — ${i} to place. The flag stays on the right half.`:"Drag to rearrange, then start the battle.",this.trayEl.replaceChildren(a,r,c)}shuffleEditor(){if(!this.editor||!this.view)return;this.editorSel=null;const t=this.firstOpenSlot(this.view);for(let i=t;i<40;i++)this.editor[i]=null;const e=[];for(const[i,a]of this.editorCounts())for(let r=0;r<a;r++)e.push(i);const s=()=>this.editor.flatMap((i,a)=>i===null&&a>=t?[a]:[]),o=e.indexOf("F");if(o>=0){const i=s().filter(r=>this.flagAllowed(r)),a=i[Math.floor(Math.random()*i.length)];this.editor[a]="F",e.splice(o,1)}for(let i=e.length-1;i>0;i--){const a=Math.floor(Math.random()*(i+1));[e[i],e[a]]=[e[a],e[i]]}for(const i of s())this.editor[i]=e.pop()??null}confirmDeployment(){if(!this.editor||!this.view)return;const t=this.firstOpenSlot(this.view);this.deployQueue=this.editor.slice(t).map(e=>e??"").filter(e=>e!==""),this.editor=null,this.editorSel=null,this.trayEl.hidden=!0,this.root.classList.remove("sg-deploying"),this.drainDeployQueue()}drainDeployQueue(){if(!this.deployQueue||!this.myTurn)return;const t=this.deployQueue.shift();this.deployQueue.length===0&&(this.deployQueue=null);const e=t?this.placements.get(t):void 0;e&&this.submitOnce(e)}refreshEditor(){this.view&&(this.syncPieces(this.view),this.syncDeployPanel(this.view),this.syncHighlights(this.view))}syncHighlights(t){for(const e of this.squaresEl.children)e.classList.remove("sg-sq-last-from","sg-sq-last-to","sg-sq-next","sg-sq-selected","sg-sq-target","sg-sq-capture","sg-sq-movable","sg-sq-drop","sg-sq-home","sg-sq-flagzone");if(t.lastMove&&(this.squareEl(t.lastMove.from)?.classList.add("sg-sq-last-from"),this.squareEl(t.lastMove.to)?.classList.add("sg-sq-last-to")),this.editor&&this.view){const e=this.firstOpenSlot(this.view);for(let s=e;s<40;s++){const o=this.squareEl(this.cellOfSlot(s));o?.classList.add("sg-sq-home"),this.editorSel==="F"&&this.flagAllowed(s)&&o?.classList.add("sg-sq-flagzone")}}if(t.phase==="deploy"&&t.nextSquare!==null&&this.deployQueue!==null&&this.squareEl(t.nextSquare)?.classList.add("sg-sq-next"),this.myTurn&&this.view?.phase==="play"){for(const e of this.moves.keys())this.squareEl(e)?.classList.add("sg-sq-movable");this.selected!==null&&this.showSelection(this.selected)}}showSelection(t){this.squareEl(t)?.classList.add("sg-sq-selected");const e=this.moves.get(t);if(!(!e||!this.view))for(const s of e.keys()){const o=this.view.cells[s],i=o!==null&&o!=="~";this.squareEl(s)?.classList.add(i?"sg-sq-capture":"sg-sq-target")}}async animate(t,e){const s=Ho(t.data),o=this.ctx.animationScale();if(!s||o===0||this.view?.phase!=="play"){this.render(e),this.view?.phase==="deploy"&&o>0&&await y(40*o);return}const i=this.pieceEl(s.from);if(i&&(i.style.transitionDuration=`${ce*o}ms`,i.style.zIndex="4",this.place(i,s.to),await y(ce*o)),s.battle&&i){const a=this.pieceEl(s.to);this.reveal(i,s.battle.attacker,s.mover.o),a&&a!==i&&this.reveal(a,s.battle.defender,1-s.mover.o),await y(Ao*o);const r=[];s.battle.outcome!=="win"&&r.push(i),s.battle.outcome!=="loss"&&a&&a!==i&&r.push(a);for(const l of r)l.classList.add("sg-sinking");r.length&&await y(To*o)}this.render(e)}reveal(t,e,s){t.classList.add("sg-revealing"),t.innerHTML=H(e,s)}promptAction(t){this.moves.clear(),this.placements.clear();for(const e of t){const s=/^(\d+)->(\d+)$/.exec(e);if(s){const i=Number(s[1]),a=Number(s[2]);this.moves.has(i)||this.moves.set(i,new Map),this.moves.get(i).set(a,e);continue}const o=/^([A-M]) \(/.exec(e);if(o){const i=Do[o[1]];i&&this.placements.set(i,o[1])}}if(this.myTurn=!0,this.selected=null,this.deployQueue){setTimeout(()=>this.drainDeployQueue(),0);return}this.view&&(this.syncDeployPanel(this.view),this.syncHighlights(this.view))}submitOnce(t){this.myTurn&&(this.myTurn=!1,this.selected=null,this.moves.clear(),this.placements.clear(),this.trayEl.hidden=!0,this.root.classList.remove("sg-deploying"),this.view&&this.syncHighlights(this.view),this.ctx.submit(t))}cellFromEvent(t){const e=this.squaresEl.getBoundingClientRect(),s=Math.floor((t.clientX-e.left)/e.width*10),o=Math.floor((t.clientY-e.top)/e.height*10);return s<0||s>9||o<0||o>9?null:this.cellAt(s,o)}onPointerDown(t){if(t.button!==0)return;if(this.editor){this.onEditorPointerDown(t);return}if(!this.myTurn||this.view?.phase!=="play")return;const e=this.cellFromEvent(t);if(e===null)return;if(this.selected!==null){const i=this.moves.get(this.selected)?.get(e);if(i){this.submitOnce(i);return}}if(!this.moves.has(e)){this.selected=null,this.view&&this.syncHighlights(this.view);return}this.selected=e,this.view&&this.syncHighlights(this.view);const s=this.pieceEl(e);if(!s)return;const o=s.cloneNode(!0);o.classList.add("sg-ghost"),this.piecesEl.append(o),s.classList.add("sg-drag-src"),this.drag={piece:s,ghost:o,from:e,moved:!1,editorFrom:null},t.currentTarget.setPointerCapture(t.pointerId),this.moveGhost(t)}onEditorPointerDown(t){if(!this.editor||!this.view)return;const e=this.cellFromEvent(t),s=e===null?-1:this.slotOfCell(e);if(!(s>=this.firstOpenSlot(this.view)&&s<40))return;if(this.editorSel!==null){if(this.editorSel==="F"&&!this.flagAllowed(s))return;const l=this.editor[s];if(l==="F"&&this.editorSel!=="F")return;this.editor[s]=this.editorSel;const c=this.editorCounts();this.editorSel=l??((c.get(this.editorSel)??0)>0?this.editorSel:null),this.refreshEditor();return}if(this.editor[s]===null)return;const a=this.piecesEl.querySelector(`.sg-edit[data-cell="${e}"]`);if(!a)return;const r=a.cloneNode(!0);r.classList.add("sg-ghost"),this.piecesEl.append(r),a.classList.add("sg-drag-src"),this.drag={piece:a,ghost:r,from:e,moved:!1,editorFrom:s},t.currentTarget.setPointerCapture(t.pointerId),this.moveGhost(t)}moveGhost(t){if(!this.drag)return;const e=this.squaresEl.getBoundingClientRect(),s=e.width/10,o=t.clientX-e.left-s/2,i=t.clientY-e.top-s/2;this.drag.ghost.style.transform=`translate(${o}px, ${i}px)`}onPointerMove(t){if(!this.drag)return;this.drag.moved=!0,this.moveGhost(t);const e=this.cellFromEvent(t);for(const s of this.squaresEl.children)s.classList.remove("sg-sq-drop");e!==null&&(this.drag.editorFrom!==null?this.editorDropOk(this.drag.editorFrom,e)&&this.squareEl(e)?.classList.add("sg-sq-drop"):this.moves.get(this.drag.from)?.has(e)&&this.squareEl(e)?.classList.add("sg-sq-drop"))}editorDropOk(t,e){if(!this.editor||!this.view)return!1;const s=this.slotOfCell(e);if(s<this.firstOpenSlot(this.view)||s>=40||s===t)return!1;const o=this.editor[t],i=this.editor[s];return!(o==="F"&&!this.flagAllowed(s)||i==="F"&&!this.flagAllowed(t))}onPointerUp(t){if(!this.drag)return;const{from:e,moved:s,editorFrom:o}=this.drag,i=this.cellFromEvent(t);if(this.cancelDrag(),o!==null){if(!this.editor)return;if(!s||i===e){this.editorSel=this.editor[o],this.editor[o]=null,this.refreshEditor();return}if(i!==null&&this.editorDropOk(o,i)){const r=this.slotOfCell(i);[this.editor[o],this.editor[r]]=[this.editor[r],this.editor[o]],this.refreshEditor()}return}if(!s||i===null||i===e)return;const a=this.moves.get(e)?.get(i);a&&this.submitOnce(a)}cancelDrag(){if(this.drag){this.drag.ghost.remove(),this.drag.piece.classList.remove("sg-drag-src"),this.drag=null;for(const t of this.squaresEl.children)t.classList.remove("sg-sq-drop")}}unmount(){this.cancelDrag()}}function Oo(n){if(!n||typeof n!="object")return null;const t=n;return Array.isArray(t.cells)&&t.cells.length===100?t:null}function Ho(n){if(!n||typeof n!="object")return null;const t=n;return typeof t.from=="number"&&typeof t.to=="number"?t:null}const _o=new Set([42,43,46,47,52,53,56,57]),Go=`<svg class="sg-terrain" viewBox="0 0 600 600" preserveAspectRatio="none" aria-hidden="true">
  <defs>
    <filter id="sg-grass-fine" x="0" y="0" width="100%" height="100%">
      <feTurbulence type="fractalNoise" baseFrequency="0.55" numOctaves="2" seed="11" result="n" />
      <feColorMatrix in="n" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0.6 0.6 0 0 0" />
      <feComposite operator="in" in2="SourceGraphic" />
    </filter>
    <filter id="sg-grass-patch" x="0" y="0" width="100%" height="100%">
      <feTurbulence type="fractalNoise" baseFrequency="0.012 0.016" numOctaves="3" seed="4" result="n" />
      <feColorMatrix in="n" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0.9 0 0 0 0" />
      <feComposite operator="in" in2="SourceGraphic" />
    </filter>
    <filter id="sg-water-ripple" x="-10%" y="-10%" width="120%" height="120%">
      <feTurbulence type="fractalNoise" baseFrequency="0.02 0.09" numOctaves="2" seed="7" result="n" />
      <feDisplacementMap in="SourceGraphic" in2="n" scale="7" />
    </filter>
  </defs>
  <rect width="600" height="600" class="sg-t-field" />
  <rect width="600" height="600" class="sg-t-patch" filter="url(#sg-grass-patch)" />
  <rect width="600" height="600" class="sg-t-speckle" filter="url(#sg-grass-fine)" />
  <path class="sg-t-grid" d="${Array.from({length:9},(n,t)=>{const e=(t+1)*60;return`M${e} 0V600M0 ${e}H600`}).join("")}" />
  ${[123,363].map(n=>`
  <g>
    <rect x="${n-4}" y="238" width="122" height="124" rx="20" class="sg-t-bank" />
    <rect x="${n}" y="242" width="114" height="116" rx="16" class="sg-t-water" />
    <g filter="url(#sg-water-ripple)">
      <path class="sg-t-wave" d="M${n+12} 272 h90 M${n+12} 300 h90 M${n+12} 328 h90" />
    </g>
  </g>`).join("")}
  <rect x="1" y="1" width="598" height="598" class="sg-t-edge" />
</svg>`;function H(n,t){const e=t===0?"r":"b";return`<svg class="sg-badge" viewBox="0 0 100 100" aria-hidden="true">
    <defs>
      <linearGradient id="sg-rim-${e}" x1="0" y1="0" x2="0.6" y2="1">
        <stop offset="0" class="sg-rim-hi" /><stop offset="1" class="sg-rim-lo" />
      </linearGradient>
      <linearGradient id="sg-face-${e}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" class="sg-face-hi" /><stop offset="1" class="sg-face-lo" />
      </linearGradient>
      <pattern id="sg-rib-${e}" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="8" height="8" fill="url(#sg-face-${e})" />
        <line x1="0" y1="0" x2="0" y2="8" class="sg-rib-line" />
      </pattern>
    </defs>
    <rect x="5" y="3" width="90" height="94" rx="12" fill="url(#sg-rim-${e})" />
    <rect x="5" y="3" width="90" height="94" rx="12" class="sg-rim-edge" />
    ${n===null?Fo(e):No(n,e)}
  </svg>`}function No(n,t){const e=tt[n]??"",s=/^\d+$/.test(n)||n==="S"?n:"",o=s?`<circle cx="23" cy="20" r="13.5" class="sg-corner" />
       <text x="23" y="${n==="10"?25:27}" text-anchor="middle" class="sg-num${n==="10"?" sg-num-10":""}">${s}</text>`:"",i=s?.128:.148,a=50-256*i,r=(s?92:88)-512*i;return`<rect x="11" y="9" width="78" height="82" rx="8" fill="url(#sg-face-${t})" />
    <rect x="11" y="9" width="78" height="82" rx="8" class="sg-face-line" />
    <g transform="translate(${a} ${r}) scale(${i})">
      <path d="${e}" class="sg-figure-shadow" transform="translate(10 10)" />
      <path d="${e}" class="sg-figure" />
    </g>
    ${o}`}function Fo(n){return`<rect x="11" y="9" width="78" height="82" rx="8" fill="url(#sg-rib-${n})" />
    <rect x="11" y="9" width="78" height="82" rx="8" class="sg-face-line" />
    <rect x="17" y="15" width="66" height="70" rx="5" class="sg-back-border" />
    <g transform="translate(${50-256*.1} ${50-256*.1}) scale(0.1)">
      <path d="${tt.back}" class="sg-crest" />
    </g>`}const de="stratego-frontend-style";function Ro(){if(document.getElementById(de))return;const n=document.createElement("style");n.id=de,n.textContent=Io,document.head.append(n)}const Io=`
.sg-root {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: min(100%, var(--board-fit));
  margin: 0 auto;
  --sg-red: #b23a34;
  --sg-red-rim-hi: #d8655c;
  --sg-red-rim-lo: #6e1b16;
  --sg-red-face-hi: #c24a42;
  --sg-red-face-lo: #932e27;
  --sg-red-corner: #5e150f;
  --sg-blue: #33589e;
  --sg-blue-rim-hi: #5d7fc0;
  --sg-blue-rim-lo: #16294f;
  --sg-blue-face-hi: #47689f;
  --sg-blue-face-lo: #2b4778;
  --sg-blue-corner: #12274d;
  --sg-gold: #e9c97e;
  --sg-gold-deep: #caa552;
  --sg-t-field: #b3c68c;
  --sg-t-patch: #6f8b4f;
  --sg-t-speckle: #3f5a2b;
  --sg-t-grid: #55673c;
  --sg-t-bank: #cfc191;
  --sg-t-water: #8db8d8;
  --sg-t-wave: #eaf3f9;
  --sg-t-edge: #4c5a38;
}
.dark .sg-root {
  --sg-red-rim-hi: #b04840;
  --sg-red-rim-lo: #45100c;
  --sg-red-face-hi: #9c3831;
  --sg-red-face-lo: #6d211b;
  --sg-red-corner: #3c0d09;
  --sg-blue-rim-hi: #4a6aa8;
  --sg-blue-rim-lo: #0d1a35;
  --sg-blue-face-hi: #3a5588;
  --sg-blue-face-lo: #223a64;
  --sg-blue-corner: #0b1b3a;
  --sg-gold: #d9b96c;
  --sg-gold-deep: #a8843c;
  --sg-t-field: #35422a;
  --sg-t-patch: #1f2b16;
  --sg-t-speckle: #0b1107;
  --sg-t-grid: #141b0e;
  --sg-t-bank: #4d4834;
  --sg-t-water: #2c4d6e;
  --sg-t-wave: #6f97b8;
  --sg-t-edge: #10150b;
}

.sg-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 34px;
}
.sg-side-tag {
  font: 700 11px var(--mono, ui-monospace, monospace);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 3px 8px;
  border-radius: 999px;
  color: #fff;
}
.sg-side-red { background: var(--sg-red); }
.sg-side-blue { background: var(--sg-blue); }
.sg-tray {
  display: flex;
  flex-wrap: wrap;
  gap: 2px;
  margin-left: auto;
  max-width: 70%;
}
.sg-tray-piece {
  width: 22px;
  height: 22px;
  opacity: 0.9;
}
.sg-tray-piece svg { display: block; width: 100%; height: 100%; }

.sg-stage { display: flex; gap: 12px; align-items: flex-start; }
.sg-board {
  position: relative;
  flex: 1;
  aspect-ratio: 1;
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: var(--card-shadow);
  border: 1px solid var(--border);
  touch-action: none;
  user-select: none;
  -webkit-user-select: none;
}

/* --- terrain --- */
.sg-terrain { position: absolute; inset: 0; width: 100%; height: 100%; }
.sg-t-field { fill: var(--sg-t-field); }
.sg-t-patch { fill: var(--sg-t-patch); opacity: 0.34; }
.sg-t-speckle { fill: var(--sg-t-speckle); opacity: 0.2; }
.sg-t-grid { stroke: var(--sg-t-grid); stroke-width: 1.1; opacity: 0.45; fill: none; }
.sg-t-bank { fill: var(--sg-t-bank); }
.sg-t-water { fill: var(--sg-t-water); }
.sg-t-wave { stroke: var(--sg-t-wave); stroke-width: 2.2; opacity: 0.5; fill: none; stroke-linecap: round; }
.sg-t-edge { fill: none; stroke: var(--sg-t-edge); stroke-width: 2; opacity: 0.55; }

.sg-squares {
  position: absolute;
  inset: 0;
  display: grid;
  grid-template: repeat(10, 1fr) / repeat(10, 1fr);
}
.sg-sq { position: relative; }

/* --- square states --- */
.sg-sq-last-from, .sg-sq-last-to {
  background: color-mix(in srgb, var(--accent) 30%, transparent);
  box-shadow: inset 0 0 0 2px color-mix(in srgb, var(--accent) 55%, transparent);
}
.sg-sq-next {
  animation: sg-pulse 1.2s ease-in-out infinite;
  box-shadow: inset 0 0 0 3px var(--accent);
  background: color-mix(in srgb, var(--accent) 18%, transparent);
}
@keyframes sg-pulse {
  50% { box-shadow: inset 0 0 0 5px var(--accent); }
}
.sg-sq-selected { box-shadow: inset 0 0 0 3px var(--accent); }
.sg-sq-movable { cursor: grab; }
.sg-sq-target::after {
  content: '';
  position: absolute;
  inset: 50%;
  width: 30%;
  height: 30%;
  translate: -50% -50%;
  border-radius: 50%;
  background: color-mix(in srgb, var(--accent) 60%, transparent);
  box-shadow: 0 0 0 2px rgba(255,255,255,0.35);
}
.sg-sq-capture { box-shadow: inset 0 0 0 3px color-mix(in srgb, var(--bad, #d33) 80%, var(--accent)); }
.sg-sq-drop { box-shadow: inset 0 0 0 4px var(--accent); }
@media (prefers-reduced-motion: reduce) {
  .sg-sq-next { animation: none; }
}

/* --- pieces --- */
.sg-pieces { position: absolute; inset: 0; pointer-events: none; }
.sg-piece {
  position: absolute;
  width: 10%;
  height: 10%;
  /* Percent padding resolves against the pieces layer (the whole board), so
   * 0.55% of the board is 5.5% of this square. */
  padding: 0.55%;
  box-sizing: border-box;
  transition: transform 0.24s cubic-bezier(0.2, 0.8, 0.3, 1);
  will-change: transform;
}
.sg-piece svg {
  display: block;
  width: 100%;
  height: 100%;
  filter: drop-shadow(0 1.5px 2px rgba(0,0,0,0.45));
}
.sg-piece.sg-revealing svg { animation: sg-flip 0.5s ease; }
@keyframes sg-flip {
  0% { transform: rotateY(90deg); }
  100% { transform: rotateY(0deg); }
}
.sg-piece.sg-sinking { opacity: 0; scale: 0.6; transition: opacity 0.32s ease, scale 0.32s ease; }
.sg-piece.sg-drag-src { opacity: 0.35; }
.sg-ghost {
  transition: none;
  z-index: 6;
  opacity: 0.9;
  pointer-events: none;
}
.sg-has-moved::after {
  content: '';
  position: absolute;
  right: 10%;
  bottom: 8%;
  width: 13%;
  height: 13%;
  border-radius: 50%;
  background: var(--sg-gold);
  outline: 1.5px solid rgba(0,0,0,0.45);
}
.sg-known .sg-rim-edge { stroke: var(--sg-gold); stroke-width: 3.5; }

/* --- token materials --- */
.sg-red .sg-rim-hi { stop-color: var(--sg-red-rim-hi); }
.sg-red .sg-rim-lo { stop-color: var(--sg-red-rim-lo); }
.sg-red .sg-face-hi { stop-color: var(--sg-red-face-hi); }
.sg-red .sg-face-lo { stop-color: var(--sg-red-face-lo); }
.sg-red .sg-corner { fill: var(--sg-red-corner); }
.sg-red .sg-rib-line { stroke: var(--sg-red-rim-lo); stroke-width: 2.6; }
.sg-blue .sg-rim-hi { stop-color: var(--sg-blue-rim-hi); }
.sg-blue .sg-rim-lo { stop-color: var(--sg-blue-rim-lo); }
.sg-blue .sg-face-hi { stop-color: var(--sg-blue-face-hi); }
.sg-blue .sg-face-lo { stop-color: var(--sg-blue-face-lo); }
.sg-blue .sg-corner { fill: var(--sg-blue-corner); }
.sg-blue .sg-rib-line { stroke: var(--sg-blue-rim-lo); stroke-width: 2.6; }
.sg-rim-edge { fill: none; stroke: rgba(255,255,255,0.22); stroke-width: 1.6; }
.sg-face-line { fill: none; stroke: rgba(0,0,0,0.28); stroke-width: 1.2; }
.sg-figure { fill: var(--sg-gold); }
.sg-figure-shadow { fill: rgba(0,0,0,0.3); }
.sg-corner { stroke: var(--sg-gold-deep); stroke-width: 1.6; }
.sg-num {
  font: 800 21px var(--mono, ui-monospace, monospace);
  fill: var(--sg-gold);
}
.sg-num-10 { font-size: 16px; letter-spacing: -1px; }
.sg-back-border { fill: none; stroke: var(--sg-gold); stroke-width: 1.6; opacity: 0.65; }
.sg-crest { fill: var(--sg-gold); opacity: 0.7; }

/* --- editor squares --- */
.sg-sq-home { box-shadow: inset 0 0 0 1.5px color-mix(in srgb, var(--accent) 45%, transparent); }
.sg-sq-flagzone { background: color-mix(in srgb, var(--accent) 20%, transparent); }
.sg-piece.sg-edit { cursor: grab; pointer-events: none; }

/* --- deployment panel --- */
.sg-supply[hidden] { display: none; }
.sg-supply {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 10px;
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  width: 148px;
  flex-shrink: 0;
}
.sg-supply-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 6px;
}
.sg-supply-actions { display: flex; flex-direction: column; gap: 6px; }
.sg-action {
  padding: 7px 10px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg-inset);
  color: var(--text);
  font: 600 13px inherit;
  cursor: pointer;
}
.sg-action:hover:not(:disabled) { border-color: var(--accent); }
.sg-action:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.sg-action:disabled { opacity: 0.45; cursor: default; }
.sg-action-go {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
  font-weight: 700;
}
.sg-supply-hint { font-size: 11.5px; line-height: 1.35; color: var(--text-dim); }
.sg-supply-sel { border-color: var(--accent) !important; box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 45%, transparent); }
.sg-supply-btn {
  position: relative;
  aspect-ratio: 1;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg-inset);
  cursor: pointer;
  padding: 5px;
}
.sg-supply-btn:hover:not(:disabled) { border-color: var(--accent); }
.sg-supply-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.sg-supply-btn:disabled { opacity: 0.3; cursor: default; }
.sg-supply-btn svg { display: block; width: 100%; height: 100%; }
.sg-supply-count {
  position: absolute;
  right: 2px;
  top: 2px;
  font: 600 11px var(--mono, ui-monospace, monospace);
  color: var(--text-dim);
  background: var(--bg-raised);
  border-radius: 6px;
  padding: 0 4px;
}

.sg-fallback {
  font-family: var(--mono, ui-monospace, monospace);
  background: var(--bg-inset);
  padding: 12px;
  border-radius: var(--radius);
  overflow-x: auto;
}

@media (max-width: 560px) {
  .sg-stage { flex-direction: column; }
  .sg-supply { width: 100%; grid-template-columns: repeat(6, 1fr); }
}
`,he="twentyone-frontend-style",jo=`
.t21 { display: flex; flex-direction: column; gap: 14px; }
.t21-table { position: relative; display: flex; flex-direction: column; gap: 12px;
  border-radius: calc(var(--radius) + 4px); padding: 18px 20px;
  background: radial-gradient(120% 95% at 50% 0%, #17463a 0%, #0f3328 55%, #0a2419 100%);
  border: 1px solid var(--border);
  box-shadow: inset 0 0 70px rgba(0, 0, 0, .45), 0 14px 40px rgba(0, 0, 0, .45); }
.t21-seat { display: flex; flex-direction: column; gap: 10px; }
.t21-seat[data-pos="bottom"] { flex-direction: column-reverse; }
.t21-seat-bar { display: flex; align-items: center; gap: 12px; width: fit-content;
  padding: 6px 14px; border-radius: 999px; background: rgba(1, 4, 9, .4);
  color: #e6edf3;
  border: 1px solid transparent; transition: border-color .2s, box-shadow .2s; }
.t21-seat-bar.t21-active { border-color: #58a6ff;
  box-shadow: 0 0 14px rgba(88, 166, 255, .28); }
.t21-name { font-weight: 600; }
.t21-hearts { letter-spacing: 2px; font-size: 15px; line-height: 1; }
.t21-heart { color: #ff5d6c; text-shadow: 0 0 6px rgba(255, 93, 108, .45);
  display: inline-block; }
.t21-heart.t21-lost { color: rgba(230, 237, 243, .16); text-shadow: none; }
.t21-heart-break { animation: t21-heart-break .8s ease-in forwards; }
@keyframes t21-heart-break {
  40% { transform: scale(1.45); }
  to { transform: scale(.3); opacity: 0; }
}
.t21-badge { font-size: 10px; font-weight: 700; letter-spacing: .14em;
  text-transform: uppercase; color: #8fae6e; padding: 2px 8px;
  border: 1px solid #8fae6e; border-radius: 999px; }
.t21-cards { display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
  min-height: 88px; }
.t21-card { position: relative; width: clamp(46px, 9vw, 60px); aspect-ratio: 5 / 7;
  border-radius: 8px; background: linear-gradient(160deg, #fdfcf7, #e9e6d8);
  color: #222b3a; display: grid; place-items: center; font-weight: 700;
  font-size: clamp(18px, 3.4vw, 24px);
  box-shadow: 0 4px 10px rgba(0, 0, 0, .4), inset 0 0 0 1px rgba(0, 0, 0, .1); }
.t21-pip { position: absolute; top: 3px; left: 6px; font-size: 11px; font-weight: 600; }
.t21-pip-br { top: auto; left: auto; bottom: 3px; right: 6px; transform: rotate(180deg); }
.t21-back { background:
  repeating-linear-gradient(45deg, rgba(88, 166, 255, .16) 0 5px, transparent 5px 10px),
  repeating-linear-gradient(-45deg, rgba(188, 140, 255, .12) 0 5px, transparent 5px 10px),
  linear-gradient(#233048, #1b2740);
  box-shadow: 0 4px 10px rgba(0, 0, 0, .4), inset 0 0 0 2px rgba(88, 166, 255, .25);
  color: rgba(88, 166, 255, .75); font-size: clamp(15px, 2.6vw, 19px); }
.t21-hole { outline: 2px dashed rgba(88, 166, 255, .5); outline-offset: 2px; }
.t21-hole::after { content: 'hole'; position: absolute; bottom: 2px; left: 50%;
  transform: translateX(-50%); font-size: 8px; font-weight: 600;
  letter-spacing: .14em; text-transform: uppercase; color: #58a6ff; }
.t21-spent { opacity: .55; filter: saturate(.7); }
.t21-deal { animation: t21-deal .32s cubic-bezier(.2, .8, .3, 1.15) backwards; }
@keyframes t21-deal {
  from { transform: translateY(-20px) rotate(4deg) scale(.72); opacity: 0; }
}
.t21-flip { animation: t21-flip .5s ease both; }
@keyframes t21-flip { from { transform: rotateY(90deg); } }
.t21-placeholder { color: rgba(230, 237, 243, .6); font-size: 13px; font-style: italic; }
.t21-total { width: fit-content; min-height: 27px; padding: 4px 12px;
  border-radius: 999px; background: rgba(1, 4, 9, .45); border: 1px solid #2d352c;
  font-size: 13px; color: rgba(230, 237, 243, .65); }
.t21-total b { color: #e6edf3; font-size: 15px; }
.t21-total.t21-bust b { color: #d96a5a; }
.t21-total.t21-sweet b { color: #ffd566; text-shadow: 0 0 8px rgba(255, 213, 102, .5); }
.t21-mid { display: flex; align-items: center; justify-content: space-between;
  gap: 14px; padding: 2px 2px; color: #e6edf3; }
.t21-round b { font-size: 15px; display: block; }
.t21-stake { font-size: 12px; color: rgba(230, 237, 243, .6); }
.t21-deck { display: flex; align-items: center; gap: 10px; }
.t21-deck-pile { position: relative; width: 34px; height: 46px; }
.t21-deck-pile i { position: absolute; inset: 0; border-radius: 6px;
  background: linear-gradient(#233048, #1b2740);
  box-shadow: inset 0 0 0 1.5px rgba(88, 166, 255, .3), 0 2px 5px rgba(0, 0, 0, .4); }
.t21-deck-pile i:nth-child(1) { transform: translate(-3px, 2px) rotate(-4deg); }
.t21-deck-pile i:nth-child(3) { transform: translate(3px, -2px) rotate(3deg); }
.t21-deck-pulse { animation: t21-deck-pulse .26s ease; }
@keyframes t21-deck-pulse { 50% { transform: scale(1.12); } }
.t21-deck-count { font-size: 12px; color: rgba(230, 237, 243, .6); white-space: nowrap; }
.t21-banner { position: absolute; inset: 0; display: grid; place-items: center;
  pointer-events: none; z-index: 3; }
.t21-banner[hidden] { display: none; }
.t21-banner-chip { max-width: 82%; text-align: center; padding: 12px 26px;
  border-radius: 14px; background: rgba(1, 4, 9, .85); backdrop-filter: blur(4px);
  border: 1px solid #2d352c; color: #e6edf3; font-weight: 700; font-size: 17px;
  animation: t21-pop .35s cubic-bezier(.2, .9, .3, 1.3) backwards; }
.t21-banner-good .t21-banner-chip { border-color: #8fae6e; color: #8fae6e; }
.t21-banner-bad .t21-banner-chip { border-color: #d96a5a; color: #d96a5a; }
@keyframes t21-pop { from { transform: scale(.7); opacity: 0; } }
.t21-actions { display: flex; gap: 12px; justify-content: center; min-height: 64px; }
.t21-btn { flex: 1 1 0; max-width: 230px; padding: 12px 18px; border-radius: 14px;
  border: 1px solid var(--border); background: var(--bg-raised); color: var(--text);
  font-weight: 700; font-size: 17px; letter-spacing: .07em; text-transform: uppercase;
  transition: transform .08s, filter .15s, box-shadow .15s; }
.t21-btn:hover:not(:disabled) { filter: brightness(1.15); }
.t21-btn:active:not(:disabled) { transform: translateY(2px) scale(.985); }
.t21-btn:disabled { opacity: .45; cursor: default; }
.t21-btn-draw { background: linear-gradient(135deg, var(--accent), var(--accent-2));
  border: none; color: #fff; box-shadow: 0 6px 18px rgba(88, 166, 255, .3); }
.dark .t21-btn-draw { color: #04111f; }
.t21-btn span { display: block; font-size: 11px; font-weight: 500; letter-spacing: 0;
  text-transform: none; opacity: .8; }
.t21-stand-flash { animation: t21-stand-flash .4s ease; }
@keyframes t21-stand-flash { 30% { box-shadow: 0 0 0 2px var(--accent); } }
@media (max-width: 520px) {
  .t21-table { padding: 12px; }
  .t21-cards { gap: 6px; min-height: 72px; }
}
`;function Yo(){if(document.getElementById(he))return;const n=document.createElement("style");n.id=he,n.textContent=jo,document.head.append(n)}const pe=`
  <div class="t21-seat-bar">
    <span class="t21-name"></span>
    <span class="t21-hearts"></span>
    <span class="t21-badge" hidden>stood</span>
  </div>
  <div class="t21-cards"></div>
  <div class="t21-total"></div>`;function ue(n){return n.reduce((t,e)=>t+e,0)}class Wo{ctx;seatEls=[];roundEl;stakeEl;deckEl;deckPile;deckCountEl;bannerEl;bannerChip;actionsEl;prevCounts=[0,0];lastRound=0;prevHearts=null;roundEndSeen=!1;mount(t,e){this.ctx=e,Yo(),t.innerHTML=`
      <div class="t21">
        <div class="t21-table">
          <section class="t21-seat" data-pos="top">${pe}</section>
          <div class="t21-mid">
            <div class="t21-round"><b></b><span class="t21-stake"></span></div>
            <div class="t21-deck">
              <span class="t21-deck-count"></span>
              <div class="t21-deck-pile"><i></i><i></i><i></i></div>
            </div>
          </div>
          <section class="t21-seat" data-pos="bottom">${pe}</section>
          <div class="t21-banner" hidden><div class="t21-banner-chip"></div></div>
        </div>
        <div class="t21-actions"></div>
      </div>`;const s=i=>{const a=t.querySelector(`[data-pos="${i}"]`);return{bar:a.querySelector(".t21-seat-bar"),name:a.querySelector(".t21-name"),hearts:a.querySelector(".t21-hearts"),badge:a.querySelector(".t21-badge"),cards:a.querySelector(".t21-cards"),total:a.querySelector(".t21-total")}},o=e.humanSeat>=0?e.humanSeat:0;this.seatEls=[],this.seatEls[o]=s("bottom"),this.seatEls[1-o]=s("top"),this.roundEl=t.querySelector(".t21-round b"),this.stakeEl=t.querySelector(".t21-stake"),this.deckEl=t.querySelector(".t21-deck"),this.deckPile=t.querySelector(".t21-deck-pile"),this.deckCountEl=t.querySelector(".t21-deck-count"),this.bannerEl=t.querySelector(".t21-banner"),this.bannerChip=t.querySelector(".t21-banner-chip"),this.actionsEl=t.querySelector(".t21-actions"),e.humanSeat<0&&(this.actionsEl.style.display="none")}seatName(t){return t===this.ctx.humanSeat?"You":this.ctx.humanSeat>=0?"Bot":`Player ${t}`}cardEl(t,e){const s=document.createElement("div");return t===null?(s.className="t21-card t21-back t21-hole",s.textContent="?"):(s.className=e?"t21-card t21-hole":"t21-card",s.innerHTML=`<span class="t21-pip">${t}</span><b>${t}</b><span class="t21-pip t21-pip-br">${t}</span>`),s}setTotal(t,e,s){if(s===null){t.innerHTML="&nbsp;",t.classList.remove("t21-bust","t21-sweet");return}t.innerHTML=`${e} <b>${s}</b>`,t.classList.toggle("t21-bust",s>21),t.classList.toggle("t21-sweet",s===21)}renderHearts(t,e,s){const o=[];for(let i=0;i<s;i++){const a=document.createElement("span");a.className=i<e?"t21-heart":"t21-heart t21-lost",a.textContent="♥",o.push(a)}t.replaceChildren(...o)}renderSeat(t,e,s){const o=this.seatEls[t],i=e.players[t],a=this.ctx.animationScale();if(o.name.textContent=this.seatName(t),this.renderHearts(o.hearts,e.hearts[t],e.maxHearts),o.badge.hidden=!(e.roundActive&&i.stood),o.bar.classList.toggle("t21-active",e.roundActive&&!s.isOver&&e.toAct===t),e.roundActive){let r=0;const l=this.prevCounts[t],c=(h,p)=>{const g=this.cardEl(h,p);return r>=l&&a>0&&(g.classList.add("t21-deal"),g.style.animationDuration=`${320*a}ms`,g.style.animationDelay=`${(r-l)*80*a}ms`),r++,g},d=[c(i.up[0]??null,!1),c(i.down,!0)];for(const h of i.up.slice(1))d.push(c(h,!1));o.cards.replaceChildren(...d),i.total!==null?this.setTotal(o.total,"total",i.total):this.setTotal(o.total,"showing",ue(i.up)),this.prevCounts[t]=i.up.length+1}else if(e.lastReveal){const r=e.lastReveal.up[t],l=e.lastReveal.downs[t],c=[this.cardEl(r[0]??null,!1),this.cardEl(l,!0)];for(const d of r.slice(1))c.push(this.cardEl(d,!1));for(const d of c)d.classList.add("t21-spent");o.cards.replaceChildren(...c),this.setTotal(o.total,"total",ue(r)+l),this.prevCounts[t]=0}else{const r=document.createElement("div");r.className="t21-placeholder",r.textContent="waiting for the deal…",o.cards.replaceChildren(r),this.setTotal(o.total,"",null),this.prevCounts[t]=0}}showBanner(t,e){this.bannerChip.textContent=t,this.bannerEl.className=`t21-banner${e?` t21-banner-${e}`:""}`,this.bannerEl.hidden=!1}endText(t,e,s){return t===null?`Round ${s}: push — no damage`:`${t===this.ctx.humanSeat?"You win":`${this.seatName(t)} wins`} round ${s} · −${e} ♥`}endClass(t){return this.ctx.humanSeat<0||t===null?"":t===this.ctx.humanSeat?"good":"bad"}render(t){const e=t.viewData;if(!e)return;const s=e.round!==this.lastRound;if(s&&(this.prevCounts=[0,0]),t.isOver){const o=this.ctx.humanSeat>=0?e.hearts[this.ctx.humanSeat]>0?"good":"bad":"";this.showBanner(t.result??"Game over",o)}else if(e.roundActive)this.bannerEl.hidden=!0,this.roundEndSeen=!1;else if(s&&this.lastRound>0&&!this.roundEndSeen&&this.prevHearts){const o=e.hearts[0]<this.prevHearts[0]?0:e.hearts[1]<this.prevHearts[1]?1:null,i=o===null?null:1-o,a=o===null?0:this.prevHearts[o]-e.hearts[o];this.showBanner(this.endText(i,a,this.lastRound),this.endClass(i)),this.roundEndSeen=!0}this.roundEl.textContent=`Round ${e.round}`,this.stakeEl.textContent=`${e.round} ♥ at stake`,this.deckCountEl.textContent=e.roundActive?`${e.deckCount} in deck`:"",this.deckEl.style.visibility=e.roundActive?"visible":"hidden",this.renderSeat(0,e,t),this.renderSeat(1,e,t),t.toAct!==t.humanSeat&&this.actionsEl.replaceChildren(),this.lastRound=e.round,this.prevHearts=[e.hearts[0],e.hearts[1]]}async showdown(t,e){for(const s of[0,1]){const o=this.seatEls[s],i=o.cards.querySelector(".t21-back");if(i){const a=this.cardEl(t.downs[s],!0);a.classList.add("t21-flip"),a.style.animationDuration=`${500*e}ms`,i.replaceWith(a)}this.setTotal(o.total,"total",t.totals[s])}if(await y(700*e),this.showBanner(this.endText(t.winner,t.damage,this.lastRound),this.endClass(t.winner)),t.winner!==null){const s=1-t.winner,o=[...this.seatEls[s].hearts.querySelectorAll(".t21-heart:not(.t21-lost)")];for(const i of o.slice(Math.max(0,o.length-t.damage)))i.style.animationDuration=`${800*e}ms`,i.classList.add("t21-heart-break")}await y(900*e)}async animate(t,e){const s=this.ctx.animationScale(),o=t.data??null;if(o?.kind==="draw")s>0&&(this.deckPile.classList.add("t21-deck-pulse"),await y(260*s),this.deckPile.classList.remove("t21-deck-pulse")),this.render(e),await y(160*s);else if(o?.kind==="stand"){if(this.render(e),s>0){const i=this.seatEls[o.seat].bar;i.classList.add("t21-stand-flash"),await y(340*s),i.classList.remove("t21-stand-flash")}}else o?.kind==="roundEnd"?(this.roundEndSeen=!0,s>0?await this.showdown(o,s):this.showBanner(this.endText(o.winner,o.damage,this.lastRound),this.endClass(o.winner)),this.render(e),await y(250*s)):(this.render(e),await y(200*s))}promptAction(t){const e={draw:"take a card",stand:"hold your total"},s=t.map((o,i)=>{const a=document.createElement("button");a.type="button",a.className=o==="draw"?"t21-btn t21-btn-draw":"t21-btn";const r=e[o];return a.innerHTML=r?`${o}<span>${r}</span>`:o,a.onclick=()=>{for(const l of s)l.disabled=!0;this.ctx.submit(String(i))},a});this.actionsEl.replaceChildren(...s)}unmount(){}}function Uo(){return new Wo}const Le={chess:vs,connect4:Ls,go:Hs,"liars-dice":js,othello:Xs,pente:ro,poker:fo,snake:Co,stratego:Po,twentyone:Uo};function Vo(n){const t=Le[n];return t?t():new Cs}function Ko(n){return n in Le}const Qo={alphabeta:"Alpha-Beta","alphabeta-rich":"Alpha-Beta (rich)",azero:"AlphaZero (CPU)","azero-gpu":"AlphaZero",mcts:"MCTS","mcts-eval":"MCTS (eval)","mcts-spec":"MCTS (spec)",rollout:"Rollout",history:"Neural",ataraxios:"Ataraxios",belief:"Belief",bns:"Best-node search",random:"Random"};function Ct(n){return Qo[n]??n}const Xo={"stratego/ataraxios":"Follows Ataraxos AI's implementation. A 27M-parameter transformer pair — an 8-layer move net plus a 4-layer setup net that arranges its own army. Trained from scratch by self-play in MLX on a Apple M5 Max MacBook: 6.5 days, 7,600 iterations, ~1.5 billion moves.","chess/azero-gpu":"AlphaZero conv-resnet trained by self-play with MCTS in one overnight run on a MacBook.","go/azero-gpu":"AlphaZero self-play net with board-size-agnostic global-pool heads, similar to KataGo, trained for about two days on a MacBook. Play-time search is MCTS.","pente/azero-gpu":"AlphaZero self-play net.","snake/bns":"A simultaneous best-node search with bitboard territory, collision-aware quiescence, transpositions, and a phase-aware Battlesnake evaluation.","liars-dice/history":"PPO over a history-attention encoder with a belief head, trained in a multi-round self-play league with an exploiter pool across about ten days. The shipped net is the league's round-21 head-to-head champion.","liars-dice/rollout":"A Monte-Carlo rollout bot: samples the hidden dice, plays out candidate bids, and picks the best average.","poker/equity":"A Monte-Carlo equity bot: samples hole cards and runouts to estimate win probability and bets accordingly.","othello/alphabeta":"Classic alpha-beta search over a hand-tuned positional evaluation.","connect4/alphabeta":"Classic alpha-beta search with a hand-tuned evaluation.","pente/alphabeta":"Alpha-beta search with a VCF hybrid.","twentyone/__solver__":"The game solved offline into exact lookup tables, playing perfectly within its heart budget."};function Zo(n,t){return Xo[`${n}/${t}`]}const B={"chess/alphabeta":{key:"depth",levels:[["Easy","2"],["Medium","4"],["Hard","6"]]},"chess/alphabeta-rich":{key:"depth",levels:[["Easy","2"],["Medium","4"],["Hard","6"]]},"chess/azero":{key:"sims",levels:[["Trivial","1"],["Easy","64"],["Medium","256"],["Hard","800"]]},"chess/azero-gpu":{key:"sims",levels:[["Trivial","1"],["Easy","1200"],["Medium","4800"],["Hard","12000"]]},"othello/alphabeta":{key:"depth",levels:[["Easy","3"],["Medium","5"],["Hard","7"]]},"othello/mcts":{key:"sims",levels:[["Easy","500"],["Medium","2000"],["Hard","6000"]]},"connect4/alphabeta":{key:"depth",levels:[["Easy","5"],["Medium","7"],["Hard","9"]]},"connect4/mcts":{key:"sims",levels:[["Easy","500"],["Medium","2000"],["Hard","6000"]]},"pente/alphabeta":{key:"depth",levels:[["Easy","2"],["Medium","4"],["Hard","5"]]},"pente/mcts":{key:"sims",levels:[["Easy","1000"],["Medium","4000"],["Hard","10000"]]},"pente/azero-gpu":{key:"sims",levels:[["Trivial","1"],["Easy","16"],["Medium","64"],["Hard","256"]]},"go/mcts":{key:"sims",levels:[["Easy","400"],["Medium","1500"],["Hard","4000"]]},"go/mcts-eval":{key:"sims",levels:[["Easy","400"],["Medium","1500"],["Hard","4000"]]},"go/mcts-spec":{key:"sims",levels:[["Easy","400"],["Medium","1500"],["Hard","4000"]]},"go/azero-gpu":{key:"sims",levels:[["Trivial","1"],["Easy","400"],["Medium","1200"],["Hard","2400"]]},"liars-dice/rollout":{key:"rollouts",levels:[["Easy","100"],["Medium","400"],["Hard","1000"]]},"poker/equity":{key:"samples",levels:[["Easy","300"],["Medium","1200"],["Hard","3000"]]},"poker/rollout":{key:"rollouts",levels:[["Easy","60"],["Medium","150"],["Hard","400"]]},"snake/bns":{key:"millis",levels:[["Easy","25"],["Medium","120"],["Hard","440"]]}};function W(n,t){return B[`${n}/${t}`]}const Jo={players:["2","3","4","5","6"],dice:["3","4","5","6"],hearts:["3","6"],size:["9","13","15","19"]},ti={pente:{size:["19"]},snake:{players:["2","3","4"],mode:["standard","royale","constrictor","wrapped","wrapped-constrictor"],food:["standard","one"],model:["mcs","brs+","full"]},stratego:{setup:["random","manual"]}};function fe(n,t){return ti[n]?.[t]??Jo[t]}const ei=new Set(["azero","azero-gpu","ataraxios","heuristic"]);function si(n){const t=n.optsSchema.find(e=>e.key==="bot");return t?t.value.split("|").filter(e=>!ei.has(e)):[]}function G(n){const t=n.indexOf(":"),e=t<0?n:n.slice(0,t);if(!e)throw new Error(`bot spec has no bot name: '${n}'`);const s={};if(t>=0){const o=n.slice(t+1);for(const i of o.split(",")){const a=i.indexOf("=");if(a<=0)throw new Error(`bot option must be key=value, got '${i}' in '${n}'`);s[i.slice(0,a)]=i.slice(a+1)}}return{bot:e,opts:s}}function U(n,t={}){const e=Object.entries(t).filter(([,s])=>s!=="");return e.length?`${n}:${e.map(([s,o])=>`${s}=${o}`).join(",")}`:n}function ge(n,t){const e=W(n,t);return e?(e.levels[1]??e.levels[0])[1]:""}function N(n){const t=[];for(const e of n.split(","))e.includes("=")&&!e.includes(":")&&t.length?t[t.length-1]+=`,${e}`:t.push(e);return t}const oi={go:{size:"9"},"liars-dice":{players:"2",dice:"5"}};class ii{constructor(t,e,s,o,i){this.root=t,this.compare=e,this.games=s,this.statsHost=o,this.onBack=i}hosts=[];running=!1;gen=0;entrants=[];render(){const t=this.compare.map(s=>`<option value="${s.id}">${s.id}</option>`).join("");this.root.innerHTML=`
      <div class="tourney">
        <button type="button" class="link back">&larr; arcade</button>
        <h2>Tournament lab</h2>
        <p class="muted">Round-robin between bots, paired seat-swapped games on a pool of engine
           workers, Bradley-Terry Elo fitted live. Same statistics as the lab's CLI.</p>
        <div class="tourney-form">
          <label class="opt-row"><span>game</span>
            <select class="t-game">${t}</select></label>
          <div class="t-entrants"></div>
          <button type="button" class="link t-add">+ add bot</button>
          <label class="opt-row"><span>games / pairing</span>
            <select class="t-games">
              <option value="4">4</option>
              <option value="8" selected>8</option>
              <option value="16">16</option>
              <option value="32">32</option>
            </select></label>
          <button type="button" class="primary t-run">Run tournament</button>
        </div>
        <div class="t-progress"></div>
        <div class="t-standings"></div>
        <div class="t-matrix"></div>
      </div>`,this.root.querySelector(".back").onclick=()=>{this.destroy(),this.onBack()};const e=this.root.querySelector(".t-game");e.onchange=()=>{this.entrants=this.defaultEntrants(e.value),this.renderEntrants()},this.root.querySelector(".t-add").onclick=()=>{const o=this.botsFor(e.value)[0]??"";this.entrants.push({bot:o,level:this.mediumLevel(e.value,o)}),this.renderEntrants()},this.entrants=this.defaultEntrants(e.value),this.renderEntrants(),this.root.querySelector(".t-run").onclick=()=>void this.run()}botsFor(t){const e=this.games.find(s=>s.id===t);return e?si(e):[]}mediumLevel(t,e){const s=W(t,e);return s?(s.levels[1]??s.levels[0])[1]:""}defaultEntrants(t){return this.botsFor(t).map(e=>({bot:e,level:this.mediumLevel(t,e)}))}renderEntrants(){const t=this.root.querySelector(".t-game").value,e=this.botsFor(t),s=this.root.querySelector(".t-entrants");s.innerHTML=this.entrants.map((o,i)=>{const a=e.map(c=>`<option value="${c}"${c===o.bot?" selected":""}>${ct(Ct(c))}</option>`).join(""),r=W(t,o.bot),l=r?`<select class="t-level" data-i="${i}">${r.levels.map(([c,d])=>`<option value="${d}"${d===o.level?" selected":""}>${c}</option>`).join("")}</select>`:'<span class="t-nolevel">—</span>';return`<div class="t-entrant">
          <select class="t-bot" data-i="${i}">${a}</select>
          ${l}
          <button type="button" class="link t-remove" data-i="${i}" title="remove">×</button>
        </div>`}).join("");for(const o of s.querySelectorAll(".t-bot"))o.onchange=()=>{const i=Number(o.dataset.i);this.entrants[i]={bot:o.value,level:this.mediumLevel(t,o.value)},this.renderEntrants()};for(const o of s.querySelectorAll(".t-level"))o.onchange=()=>{this.entrants[Number(o.dataset.i)].level=o.value};for(const o of s.querySelectorAll(".t-remove"))o.onclick=()=>{this.entrants.splice(Number(o.dataset.i),1),this.renderEntrants()}}async run(){if(this.running){this.gen++,this.stopPool(),this.running=!1,this.root.querySelector(".t-run").textContent="Run tournament";return}const t=++this.gen,e=this.root.querySelector(".t-game").value,s=this.entrants.map(x=>{const v=W(e,x.bot),A=v&&x.level?{[v.key]:x.level}:{};return U(x.bot,A)}),o=Math.max(2,Number(this.root.querySelector(".t-games").value)||8),i=(Math.floor(Math.random()*2147483647)|1)>>>0,a=oi[e]??{};if(s.length<2){this.progress("Add at least two bots.");return}this.running=!0,this.root.querySelector(".t-run").textContent="Stop";const r=s.length,l=Math.max(1,Math.floor(o/2)),c=Array.from({length:r},()=>Array.from({length:r},()=>({w:0,d:0,l:0}))),d=[];for(let x=0;x<r;x++)for(let v=x+1;v<r;v++)for(let A=0;A<l;A++)d.push({i:x,j:v,k:A});let h=0;const p=d.length;this.renderTables(s,c,null),this.progress(`0 / ${p} pairs`);const g=navigator.hardwareConcurrency||4,f=Math.max(1,Math.min(4,g-2,p));this.hosts=Array.from({length:f},()=>new vt);let b=0,u=Promise.resolve();const m=async x=>{for(;this.gen===t&&b<d.length;){const v=d[b++],A=(i^v.i*r+v.j<<16)>>>0;try{const $=await x.pairs(e,a,s[v.i],s[v.j],A,v.k,v.k+1);if(this.gen!==t)return;const z=c[v.i][v.j];z.w+=$.w,z.d+=$.d,z.l+=$.l;const st=c[v.j][v.i];st.w+=$.l,st.d+=$.d,st.l+=$.w,h++,this.progress(`${h} / ${p} pairs`),u=u.then(async()=>{if(this.gen!==t)return;const ze=c.map(Pe=>Pe.map(ot=>[ot.w,ot.d,ot.l])),De=await this.statsHost.fitElo(ze);this.gen===t&&this.renderTables(s,c,De)})}catch($){if(this.gen!==t)return;this.progress(`error: ${$ instanceof Error?$.message:$}`),this.gen++,this.running=!1;const z=this.root.querySelector(".t-run");z&&(z.textContent="Run tournament");return}}};if(await Promise.all(this.hosts.map(m)),await u.catch(()=>{}),this.gen===t){this.progress(`done — ${h*2} games across ${p} pairs on ${f} workers`),this.running=!1;const x=this.root.querySelector(".t-run");x&&(x.textContent="Run tournament")}this.stopPool()}renderTables(t,e,s){const o=t.length,i=t.map((c,d)=>e[d].reduce((h,p)=>({w:h.w+p.w,d:h.d+p.d,l:h.l+p.l}),{w:0,d:0,l:0})),a=t.map((c,d)=>d);s&&a.sort((c,d)=>s[d]-s[c]);const r=a.map((c,d)=>{const h=i[c],p=s?`${s[c]>=0?"+":""}${s[c].toFixed(0)}`:"—";return`<tr><td>${d+1}</td><td class="t-spec">${ct(t[c])}</td>
                <td class="t-elo">${p}</td><td>${h.w}-${h.d}-${h.l}</td></tr>`}).join("");this.root.querySelector(".t-standings").innerHTML=`
      <table class="t-table">
        <thead><tr><th>#</th><th>bot</th><th>elo</th><th>W-D-L</th></tr></thead>
        <tbody>${r}</tbody>
      </table>`;let l='<table class="t-table t-grid"><thead><tr><th></th>';for(let c=0;c<o;c++)l+=`<th>${c+1}</th>`;l+="</tr></thead><tbody>";for(let c=0;c<o;c++){l+=`<tr><th>${c+1}. ${ct(ai(t[c]))}</th>`;for(let d=0;d<o;d++){const h=e[c][d];l+=c===d?'<td class="t-self">·</td>':`<td>${h.w+h.d+h.l?`${h.w}-${h.d}-${h.l}`:""}</td>`}l+="</tr>"}l+="</tbody></table>",this.root.querySelector(".t-matrix").innerHTML=l}progress(t){const e=this.root.querySelector(".t-progress");e&&(e.textContent=t)}stopPool(){for(const t of this.hosts)t.terminate();this.hosts=[]}destroy(){this.gen++,this.stopPool()}}function ai(n){return n.length>24?`${n.slice(0,22)}…`:n}function ct(n){return n.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}const Ce="19",be={chess:{bot:"azero-gpu",sims:"4800"},"liars-dice":{players:"5",dice:"5",faces:"6",rollouts:"400",bot:"history"},twentyone:{hearts:"3"},othello:{depth:"5"},connect4:{depth:"7"},pente:{size:Ce,bot:"azero-gpu",sims:"64"},go:{size:"19",bot:"azero-gpu",sims:"1200"},snake:{players:"2",food:"one",bot:"bns",millis:"25"},stratego:{bot:"ataraxios",setup:"manual"}},ni=new Set([]),me=new Set(["snake"]),Ae={"data/azero/chess.bin":"artifacts/azero-chess.bin","data/twentyone/solver-h3.bin":"artifacts/t21-solver-h3.bin","data/twentyone/solver-h6.bin":"artifacts/t21-solver-h6.bin","runs/ld_history/best.bin":"artifacts/ld-history-champion.bin","runs/stratego/ataraxios.bin":"artifacts/ataraxios.bin"};function ri(n,t){const e=t.bots?N(t.bots).map(o=>G(o).bot):[],s=[];if(n==="chess"){const o=t.bot==="azero"||e.includes("azero"),i=t.net??(o?"data/azero/chess.bin":null);i&&s.push(i)}return n==="twentyone"&&s.push(`data/twentyone/solver-h${t.hearts??"6"}.bin`),n==="liars-dice"&&(t.bot==="history"||e.includes("history"))&&s.push("runs/ld_history/best.bin"),n==="stratego"&&(t.bot==="ataraxios"||e.includes("ataraxios"))&&s.push("runs/stratego/ataraxios.bin"),s.filter(o=>o in Ae)}function li(n,t){return n.filter(e=>e.key!=="seed"&&e.key!=="seat"&&e.key!=="bot"&&!e.nativeOnly).map(e=>({key:e.key,value:t[e.key]??e.value.split("|")[0].replace(/\.{3}$/,""),note:e.note,bots:e.bots}))}function xt(n,t){const e=n.optsSchema.find(s=>s.key==="bot");return e?t.bot??(n.solo?"":e.value.split("|")[0]):""}const ci={chess:["White","Black"],othello:["Black","White"],go:["Black","White"],pente:["Black","White"],connect4:["Red","Yellow"],twentyone:["Player 1","Player 2"]};function di(n,t){return ci[n]?.[t]??`Seat ${t+1}`}const Te={value:"__solver__",label:"CFR solver",sendsBot:!1},hi={chess:["azero-gpu"],"liars-dice":["history","rollout"],poker:["equity"],othello:["alphabeta"],connect4:["alphabeta"],go:["azero-gpu"],pente:["azero-gpu","alphabeta"],snake:["bns"],stratego:["ataraxios"]};function dt(n){const t=n.optsSchema.find(s=>s.key==="bot");if(!t)return[Te];const e=new Set(t.value.split("|"));return(hi[n.id]??[]).filter(s=>e.has(s)).map(s=>({value:s,label:Ct(s),sendsBot:!0}))}function xe(n,t){return n.optsSchema.find(s=>s.key==="bot")?xt(n,t):Te.value}function ht(n,t){if(n.solo)return 1;const e=n.optsSchema.find(s=>s.key==="players");return e&&Number(t.players??e.value.split("|")[0])||2}function ye(){return(Math.floor(Math.random()*2147483647)|1)>>>0}function pi(n){const t=n.value.split("|")[0];return!t||t.endsWith("...")?void 0:t}function ui(n,t){return[...new Set([...Object.keys(n),...Object.keys(t)])].every(s=>n[s]===t[s])}function w(n){return n.replace(/[&<>"']/g,t=>`&#${t.charCodeAt(0)};`)}function ve(n,t){const e=n.map(([s,o])=>`<option value="${w(o)}"${o===t?" selected":""}>${w(s)}</option>`);return n.some(([,s])=>s===t)||e.unshift(`<option value="${w(t)}" selected>Custom (${w(t)})</option>`),e.join("")}const fi='<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3.2"/><path d="M19.4 13.5a7.7 7.7 0 0 0 0-3l1.7-1.3-1.8-3.1-2 .8a7.6 7.6 0 0 0-2.6-1.5L14.3 2h-3.6l-.4 2.1a7.6 7.6 0 0 0-2.6 1.5l-2-.8L3.9 8l1.7 1.3a7.7 7.7 0 0 0 0 3L3.9 13.5l1.8 3.1 2-.8a7.6 7.6 0 0 0 2.6 1.5l.4 2.1h3.6l.4-2.1a7.6 7.6 0 0 0 2.6-1.5l2 .8 1.8-3.1z"/></svg>',gi='<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 3h6M10 3v6.2L5.3 17a2 2 0 0 0 1.8 3h9.8a2 2 0 0 0 1.8-3L14 9.2V3"/><path d="M7.2 14h9.6"/></svg>',bi='<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true"><path d="M12 .5a11.5 11.5 0 0 0-3.64 22.42c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.36-3.88-1.36-.53-1.34-1.3-1.7-1.3-1.7-1.05-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.78 2.73 1.26 3.4.96.1-.75.4-1.27.73-1.56-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.18 1.18a11 11 0 0 1 5.79 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.42-2.69 5.39-5.25 5.68.41.35.78 1.05.78 2.12v3.14c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5z"/></svg>';function we(n){switch(n){case"chess":return'<div class="mini mini-chess"><span class="mini-pc" style="left:30%;top:30%">♞</span><span class="mini-pc mini-pc-w" style="left:70%;top:70%">♙</span></div>';case"liars-dice":return`<div class="mini mini-dice">
        <span class="mini-die"><i style="left:25%;top:25%"></i><i style="left:65%;top:65%"></i></span>
        <span class="mini-die mini-die-2"><i style="left:45%;top:45%"></i><i style="left:18%;top:18%"></i><i style="left:72%;top:72%"></i></span>
        <span class="mini-cup"></span></div>`;case"twentyone":return'<div class="mini mini-t21"><span class="mini-card">7♠</span><span class="mini-card mini-card-2">9♦</span><span class="mini-heart">♥♥♥</span></div>';case"poker":return'<div class="mini mini-poker"><span class="mini-pcard mini-pcard-r">A♥</span><span class="mini-pcard">K♠</span><span class="mini-chip mini-chip-1"></span><span class="mini-chip mini-chip-2"></span><span class="mini-chip mini-chip-3"></span></div>';case"othello":return`<div class="mini mini-othello">
        <svg class="mini-ot-svg" viewBox="0 0 320 320" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <defs>
            <radialGradient id="ot-w" cx="0.35" cy="0.3" r="0.8"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#cfc9b8"/></radialGradient>
            <radialGradient id="ot-b" cx="0.35" cy="0.3" r="0.8"><stop offset="0" stop-color="#5a5a5a"/><stop offset="1" stop-color="#0a0a0a"/></radialGradient>
          </defs>
          <rect width="320" height="320" fill="#2f6b46"/>
          ${[40,80,120,160,200,240,280].map(t=>`<line x1="${t}" y1="0" x2="${t}" y2="320" stroke="rgba(0,0,0,0.32)" stroke-width="2"/><line x1="0" y1="${t}" x2="320" y2="${t}" stroke="rgba(0,0,0,0.32)" stroke-width="2"/>`).join("")}
          ${[[80,80],[80,240],[240,80],[240,240]].map(([t,e])=>`<circle cx="${t}" cy="${e}" r="3.5" fill="rgba(0,0,0,0.42)"/>`).join("")}
          ${[["w",3,1],["b",4,1],["b",2,2],["w",3,2],["b",4,2],["w",5,2],["w",2,3],["b",3,3],["b",4,3],["b",5,3],["w",6,3],["b",1,4],["w",2,4],["w",3,4],["b",4,4],["w",5,4],["b",3,5],["w",4,5],["b",5,5],["b",4,6]].map(([t,e,s])=>`<circle cx="${20+40*Number(e)}" cy="${20+40*Number(s)}" r="16" fill="url(#ot-${t})" stroke="rgba(0,0,0,0.35)" stroke-width="0.8"/>`).join("")}
        </svg>
      </div>`;case"connect4":return`<div class="mini mini-c4">
        <svg class="mini-c4-svg" viewBox="0 0 274 110" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <defs>
            <radialGradient id="c4-r" cx="0.36" cy="0.3" r="0.85"><stop offset="0" stop-color="#ff8a7a"/><stop offset="0.55" stop-color="#e23b2e"/><stop offset="1" stop-color="#a01a12"/></radialGradient>
            <radialGradient id="c4-y" cx="0.36" cy="0.3" r="0.85"><stop offset="0" stop-color="#ffe89a"/><stop offset="0.55" stop-color="#f2c037"/><stop offset="1" stop-color="#b8860b"/></radialGradient>
          </defs>
          <rect x="2" y="2" width="270" height="250" rx="10" fill="#2256b6"/>
          <rect x="2" y="2" width="270" height="26" rx="10" fill="#2c63c9"/>
          ${[23,61,99,137,175,213,251].flatMap((t,e)=>[20,58,96].map((s,o)=>{const a={"0-2":"c4-r","1-1":"c4-y","1-2":"c4-r","2-0":"c4-r","2-1":"c4-y","2-2":"c4-r","3-0":"c4-y","3-1":"c4-r","3-2":"c4-y","4-1":"c4-r","4-2":"c4-y","5-2":"c4-r","6-1":"c4-y","6-2":"c4-r"}[`${e}-${o}`];return`<circle cx="${t}" cy="${s}" r="15.5" fill="${a?`url(#${a})`:"#16335f"}"/>`})).join("")}
        </svg>
      </div>`;case"go":return'<div class="mini mini-go"><span class="mini-stone mini-stone-b" style="left:44px;top:44px"></span><span class="mini-stone mini-stone-w" style="left:66px;top:66px"></span><span class="mini-stone mini-stone-b" style="left:44px;top:88px"></span><span class="mini-stone mini-stone-w" style="left:88px;top:44px"></span></div>';case"pente":return`<div class="mini mini-pente">${[["b",66,66],["b",88,66],["b",110,66],["b",132,66],["b",154,66],["b",88,88],["b",154,44],["w",44,66],["w",176,66],["w",66,44],["w",110,44],["w",132,88],["w",66,88]].map(([t,e,s])=>`<span class="mini-pstone mini-pstone-${t}" style="left:${e}px;top:${s}px"></span>`).join("")}</div>`;case"snake":return`<div class="mini mini-snake">
        <svg class="mini-snake-svg" viewBox="0 0 220 110" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <defs>
            <linearGradient id="snk-body" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0" stop-color="#3fb950"/><stop offset="1" stop-color="#1f7a34"/>
            </linearGradient>
            <radialGradient id="snk-head" cx="0.36" cy="0.3" r="0.85">
              <stop offset="0" stop-color="#86efa0"/><stop offset="0.6" stop-color="#46c45c"/><stop offset="1" stop-color="#1f7a34"/>
            </radialGradient>
            <radialGradient id="snk-food" cx="0.36" cy="0.3" r="0.85">
              <stop offset="0" stop-color="#ffc7be"/><stop offset="0.5" stop-color="#f85149"/><stop offset="1" stop-color="#b21f17"/>
            </radialGradient>
          </defs>
          <circle cx="40" cy="28" r="9" fill="url(#snk-food)"/>
          <path class="snk-rim" d="M22 77 H88 V33 H132 V77 H176"/>
          <path class="snk-tube" d="M22 77 H88 V33 H132 V77 H176"/>
          <path class="snk-gloss" d="M22 77 H88 V33 H132 V77 H176"/>
          <circle cx="176" cy="77" r="14.5" fill="url(#snk-head)" stroke="#0c3a1c" stroke-width="1.5"/>
          <circle cx="171" cy="70" r="4.2" fill="#fff"/><circle cx="172.2" cy="70" r="2.1" fill="#0a1f12"/>
          <circle cx="182" cy="71" r="4.2" fill="#fff"/><circle cx="183.2" cy="71" r="2.1" fill="#0a1f12"/>
          <circle cx="186" cy="74.5" r="0.9" fill="#0a1f12"/><circle cx="186" cy="80" r="0.9" fill="#0a1f12"/>
          <path d="M189 77 H201 M201 77 L207 73 M201 77 L207 81" fill="none" stroke="#e5484d" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>`;case"coil":return`<div class="mini mini-slither">
        <svg class="mini-slither-svg" viewBox="0 0 220 110" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <defs>
            <linearGradient id="cl-body" x1="0" y1="0" x2="1" y2="0.3">
              <stop offset="0" stop-color="#33d6ff"/><stop offset="0.5" stop-color="#1f8bff"/><stop offset="1" stop-color="#7b5bff"/>
            </linearGradient>
            <radialGradient id="cl-head" cx="0.36" cy="0.3" r="0.85">
              <stop offset="0" stop-color="#bdf0ff"/><stop offset="0.55" stop-color="#33b6ff"/><stop offset="1" stop-color="#1463d8"/>
            </radialGradient>
            <radialGradient id="cl-p1" cx="0.4" cy="0.35" r="0.8"><stop offset="0" stop-color="#fff2a8"/><stop offset="1" stop-color="#f5b400"/></radialGradient>
            <radialGradient id="cl-p2" cx="0.4" cy="0.35" r="0.8"><stop offset="0" stop-color="#c8ffd0"/><stop offset="1" stop-color="#34d058"/></radialGradient>
          </defs>
          <circle cx="40" cy="30" r="5" fill="url(#cl-p1)"/>
          <circle cx="150" cy="86" r="5" fill="url(#cl-p2)"/>
          <circle cx="92" cy="20" r="4" fill="url(#cl-p1)"/>
          <path class="cl-rim" d="M18 78 C 64 92, 70 40, 116 46 S 176 84, 202 50"/>
          <path class="cl-tube" d="M18 78 C 64 92, 70 40, 116 46 S 176 84, 202 50"/>
          <path class="cl-bands" d="M18 78 C 64 92, 70 40, 116 46 S 176 84, 202 50"/>
          <path class="cl-gloss" d="M18 71 C 64 85, 70 33, 116 39 S 176 77, 202 44"/>
          <circle cx="202" cy="50" r="17" fill="url(#cl-head)" stroke="#0d3a7a" stroke-width="1.5"/>
          <circle cx="206" cy="43" r="5" fill="#fff" stroke="#0a1230" stroke-width="0.8"/><circle cx="208.4" cy="43" r="2.6" fill="#0a1230"/><circle cx="206.6" cy="41.4" r="1" fill="#fff"/>
          <circle cx="206" cy="56" r="5" fill="#fff" stroke="#0a1230" stroke-width="0.8"/><circle cx="208.4" cy="56" r="2.6" fill="#0a1230"/><circle cx="206.6" cy="54.4" r="1" fill="#fff"/>
        </svg>
      </div>`;case"stratego":return`<div class="mini mini-stratego">
        <svg viewBox="0 0 220 110" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <defs>
            <linearGradient id="sgm-red" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stop-color="#c24a42"/><stop offset="1" stop-color="#932e27"/>
            </linearGradient>
            <linearGradient id="sgm-blue" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stop-color="#47689f"/><stop offset="1" stop-color="#2b4778"/>
            </linearGradient>
            <pattern id="sgm-rib" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="7" height="7" fill="#33589e"/>
              <line x1="0" y1="0" x2="0" y2="7" stroke="#16294f" stroke-width="2.2"/>
            </pattern>
            <filter id="sgm-grain" x="0" y="0" width="100%" height="100%">
              <feTurbulence type="fractalNoise" baseFrequency="0.5" numOctaves="2" seed="3" result="n"/>
              <feColorMatrix in="n" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0.7 0.7 0 0 0"/>
              <feComposite operator="in" in2="SourceGraphic"/>
            </filter>
          </defs>
          <rect width="220" height="110" fill="#b3c68c"/>
          <rect width="220" height="110" fill="#3f5a2b" opacity="0.16" filter="url(#sgm-grain)"/>
          <g stroke="#55673c" stroke-width="1" opacity="0.4">
            ${Array.from({length:9},(t,e)=>`<line x1="${(e+1)*22}" y1="0" x2="${(e+1)*22}" y2="110"/>`).join("")}
            <line x1="0" y1="33" x2="220" y2="33"/><line x1="0" y1="77" x2="220" y2="77"/>
          </g>
          <rect x="40" y="30" width="52" height="52" rx="12" fill="#cfc191"/>
          <rect x="43" y="33" width="46" height="46" rx="9" fill="#8db8d8"/>
          <path d="M50 48 h32 M54 60 h24 M50 70 h32" fill="none" stroke="#eaf3f9" stroke-width="2" opacity="0.55" stroke-linecap="round"/>
          <g transform="rotate(-4 132 62)">
            <rect x="106" y="20" width="52" height="84" rx="10" fill="#6e1b16"/>
            <rect x="110" y="25" width="44" height="74" rx="7" fill="url(#sgm-red)"/>
            <g transform="translate(111 40) scale(0.122)"><path d="${tt[10]}" fill="#e9c97e"/></g>
            <circle cx="121" cy="34" r="8.5" fill="#5e150f" stroke="#caa552" stroke-width="1.2"/>
            <text x="121" y="38" text-anchor="middle" font-family="ui-monospace,monospace" font-weight="800" font-size="11" letter-spacing="-1" fill="#e9c97e">10</text>
          </g>
          <g transform="rotate(5 184 66)">
            <rect x="160" y="24" width="52" height="84" rx="10" fill="#16294f"/>
            <rect x="164" y="29" width="44" height="74" rx="7" fill="url(#sgm-rib)"/>
            <rect x="168" y="33" width="36" height="66" rx="5" fill="none" stroke="#e9c97e" stroke-width="1.4" opacity="0.65"/>
            <g transform="translate(163 43) scale(0.09)"><path d="${tt.back}" fill="#e9c97e" opacity="0.75"/></g>
          </g>
        </svg>
      </div>`;case"dood":return'<div class="mini mini-doom"><span class="mini-doom-word">DOOD</span></div>';default:return'<div class="mini"></div>'}}class mi{constructor(t){this.root=t,window.addEventListener("hashchange",()=>this.route())}host=new vt;manifest;frontend=null;clientBot=null;tourney=null;slither=null;gen=0;speedScale=1;submitResolve=null;logEl=null;statusEl=null;sideEl=null;readoutEl=null;debugOn=localStorage.getItem("arcadeDebug")==="1";debugSubs=new Set;async start(){this.root.innerHTML='<div class="boot">Waking the engine…</div>',this.manifest=await this.host.manifest(),this.manifest.games=this.manifest.games.filter(t=>!ni.has(t.id)),this.route()}route(){const[t,e]=location.hash.replace(/^#/,"").split("?"),s=t.split("/").filter(Boolean),o=new URLSearchParams(e??"");if(s[0]==="lab"){this.renderTournament();return}if(s[0]==="dood"){this.renderDoom();return}if(s[0]==="coil"){this.renderSlither();return}if(s[0]==="g"&&s[1]){const i=this.manifest.games.find(a=>a.id===s[1]);if(i){const a=o.get("mode")==="watch"?"watch":"play";this.startMatch(i,a);return}history.replaceState(null,"","#/")}this.renderHome()}navTo(t){const e=`#${t}`;location.hash===e?this.route():location.hash=e}syncMatchUrl(t,e){const s=e==="watch"?"?mode=watch":"";history.replaceState(null,"",`#/g/${t.id}${s}`)}setGamesLink(t){const e=document.querySelector("[data-games-link]");e&&(e.hidden=!t)}renderHome(){this.teardown(),this.setGamesLink(!1);const t=this.manifest.games.map(i=>`
        <div class="card" data-game="${i.id}" role="button" tabindex="0">
          ${we(i.id)}
          <div class="card-text">
            <span class="card-name">${w(i.name||i.id)}</span>
          </div>
          <button type="button" class="card-watch" title="Watch bots play">watch</button>
        </div>`).join(""),e=`
        <div class="card card-doom" data-special="dood" role="button" tabindex="0">
          ${we("dood")}
          <div class="card-text">
            <span class="card-name">DOOD</span>
          </div>
        </div>`;this.root.innerHTML=`
      <div class="home">
        <header class="home-head">
          <h1>Games Room</h1>
        </header>
        <div class="card-grid">${t}${e}</div>
        <footer class="home-footer">
          <button type="button" class="icon-btn tourney-link" title="Tournament lab" aria-label="Tournament lab">${gi}</button>
          <a class="icon-btn" href="https://github.com/henri123lemoine/games" title="GitHub" aria-label="GitHub">${bi}</a>
        </footer>
      </div>`;for(const i of this.root.querySelectorAll(".card")){const a=this.manifest.games.find(l=>l.id===i.dataset.game);if(!a)continue;const r=()=>this.navTo(`/g/${a.id}`);i.onclick=r,i.onkeydown=l=>{(l.key==="Enter"||l.key===" ")&&(l.preventDefault(),r())},i.querySelector(".card-watch").onclick=l=>{l.stopPropagation(),this.navTo(`/g/${a.id}?mode=watch`)}}const s=this.root.querySelector('.card[data-special="dood"]');if(s){const i=()=>this.navTo("/dood");s.onclick=i,s.onkeydown=a=>{(a.key==="Enter"||a.key===" ")&&(a.preventDefault(),i())}}const o=this.root.querySelector('.card[data-special="coil"]');if(o){const i=()=>this.navTo("/coil");o.onclick=i,o.onkeydown=a=>{(a.key==="Enter"||a.key===" ")&&(a.preventDefault(),i())}}this.root.querySelector(".tourney-link").onclick=()=>this.navTo("/lab")}renderDoom(){this.teardown();const t="./doom-ai/index.html";this.root.innerHTML=`
      <div class="match doom-screen">
        <header class="match-bar">
          <span class="match-title">DOOM · THE FOUNDRY FFA</span>
          <span class="spacer"></span>
          <span class="muted doom-note">1–3 tactical opponents · click Fight, then use mouse + WASD</span>
        </header>
        <div class="doom-frame-wrap">
          <iframe class="doom-frame" src="${t}" title="DOOM"
            allow="autoplay; fullscreen"></iframe>
        </div>
      </div>`,this.setGamesLink(!0)}async renderSlither(){this.teardown(),this.root.innerHTML=`
      <div class="match slither-screen">
        <header class="match-bar">
          <span class="match-title">Coil</span>
          <span class="spacer"></span>
          <span class="muted">vs. the trained encircle bot · runs in your browser</span>
        </header>
        <div class="slither-mount"></div>
      </div>`,this.setGamesLink(!0);const t=this.root.querySelector(".slither-mount"),{SlitherScreen:e}=await je(async()=>{const{SlitherScreen:o}=await import("./index-mgQTu_g6.js");return{SlitherScreen:o}},__vite__mapDeps([0,1]),import.meta.url),s=this.gen;this.slither=new e,await this.slither.mount(t),s!==this.gen&&(this.slither.destroy(),this.slither=null)}renderTournament(){this.teardown(),this.setGamesLink(!0),this.tourney=new ii(this.root,this.manifest.compare,this.manifest.games,this.host,()=>this.navTo("/")),this.tourney.render()}buildOpts(t,e,s){const o={...be[t.id],...s};if(e==="watch"?t.solo?o.bot||=t.watchBot:o.seat="watch":t.solo?delete o.bot:o.seat==="watch"&&(o.seat="0"),o.bots){delete o.bot;for(const i of t.optsSchema)i.bots.length>0&&delete o[i.key];if(D()){const i=o.seat==="watch"?-1:Number(o.seat??"0");o.bots=N(o.bots).map((a,r)=>{const l=G(a);return r!==i&&l.bot==="azero-gpu"&&(new Set(F.map(([,d])=>d)).has(l.opts.sims??"")||(l.opts.sims=String(_))),U(l.bot,l.opts)}).join(",")}}else{const i=xt(t,o);for(const r of t.optsSchema)r.bots.length>0&&!r.bots.includes(i)&&delete o[r.key];const a=B[`${t.id}/${i}`];i==="azero-gpu"&&D()&&a&&(new Set(F.map(([,l])=>l)).has(o[a.key]??"")||(o[a.key]=String(_)))}return t.id==="pente"&&this.seatStates(t,o).some(i=>i==="azero-gpu")&&(o.size=Ce),o.seed||=String(ye()),o}async startMatch(t,e,s={}){const o=++this.gen;this.teardownMatch();try{const i=this.buildOpts(t,e,s);this.syncMatchUrl(t,e),this.renderMatchSkeleton(t,e,i);const a=this.clientBotConfigs(t,i);a.length&&D()&&this.showCpuNote(),await this.loadArtifacts(t,i);const r=await this.host.create(t.id,i);if(o!==this.gen)return;const l=this.root.querySelector(".board");this.frontend=Vo(t.id);const c={gameId:t.id,opts:i,humanSeat:r.humanSeat,numSeats:r.numSeats,submit:h=>this.submit(h),animationScale:()=>this.animationScale(),debug:()=>this.debugOn,onDebugChange:h=>this.onDebugChange(h),setDebugReadout:h=>this.setDebugReadout(h),debugLog:h=>this.debugLog(h)};this.frontend.mount(l,c),this.frontend.render(r),this.fillSeatSlots(t,i);const d=await ls(t.id,a);if(o!==this.gen){d?.cancel();return}this.clientBot=d,this.clientBot?.cpuFallback&&this.showCpuNote(this.clientBot.cpuFallback),t.id==="go"&&a.length&&!this.clientBot?.cpuFallback&&this.checkGoConformance(),this.setStatus(r.humanSeat<0?"Bots playing…":"Thinking…"),this.runLoop(o)}catch(i){o===this.gen&&this.setStatus(`Could not start: ${pt(i)}`,"error")}}renderMatchSkeleton(t,e,s){this.speedScale=1;const o=t.id==="snake"&&e==="play"?`<label class="speed-label">pace
            <select class="speed">
              <option value="1.25">relaxed</option>
              <option value="1" selected>normal</option>
              <option value="0.7">fast</option>
            </select>
          </label>`:e==="watch"?`<label class="speed-label">speed
            <select class="speed">
              <option value="2">slow</option>
              <option value="1" selected>normal</option>
              <option value="0.4">fast</option>
              <option value="0">instant</option>
            </select>
          </label>`:"";this.root.innerHTML=`
      <div class="match">
        <header class="match-bar">
          <span class="match-title">${w(t.name||t.id)}</span>
          <span class="spacer"></span>
          ${o}
          <button type="button" class="link again">rematch</button>
          <button type="button" class="icon-btn gear" title="Match settings" aria-label="Match settings">${fi}</button>
        </header>
        ${this.quickControlsHtml(t,s)}
        <div class="cpu-note" hidden></div>
        <div class="cpu-note gpu-mismatch-note" hidden></div>
        <div class="match-body${t.solo||me.has(t.id)?" match-body--solo":""}">
          <section class="board"></section>
          ${this.sideHtml(t)}
        </div>
        <div class="drawer" hidden>
          <div class="drawer-panel">
            <h3>Match settings</h3>
            <div class="drawer-fields"></div>
            <div class="drawer-actions">
              <button type="button" class="primary drawer-apply">Restart with these</button>
              <button type="button" class="link drawer-close">cancel</button>
            </div>
          </div>
        </div>
      </div>`,this.logEl=this.root.querySelector(".log"),this.statusEl=this.root.querySelector(".status"),this.sideEl=this.root.querySelector(".side"),this.readoutEl=this.root.querySelector(".debug-readout");const i=this.root.querySelector(".debug-check");i&&(i.onchange=()=>this.setDebug(i.checked)),this.setGamesLink(!0),this.root.querySelector(".again").onclick=()=>void this.startMatch(t,e,{...s,seed:String(ye())});const a=this.root.querySelector(".speed");a&&(a.onchange=l=>{this.speedScale=Number(l.target.value)});const r=this.root.querySelector(".free-input");r&&(r.onsubmit=l=>{l.preventDefault();const c=r.querySelector("input");c.value.trim()&&(this.submit(c.value.trim()),c.value="")}),this.wireDrawer(t,s),this.wireQuickControls(t,s)}quickControlsHtml(t,e){if(t.solo||e.bots)return"";const s=[],o=(i,a,r,l,c)=>`<label class="qc"><span class="qc-name">${w(a)}</span><select class="qc-select" data-key="${w(i)}">${ve(r,l)}</select></label>`;for(const i of t.optsSchema){if(i.bots.length||i.key==="seat"||i.key==="seed"||i.nativeOnly)continue;const a=fe(t.id,i.key);if(!a||a.length<=1)continue;const r=e[i.key]??i.value.split("|")[0];s.push(o(i.key,i.key,a.map(l=>[l,l]),r))}return s.length?`<div class="match-controls">${s.join("")}</div>`:""}wireQuickControls(t,e){for(const s of this.root.querySelectorAll(".match-controls .qc-select"))s.onchange=()=>{const o={};e.seat!==void 0&&(o.seat=e.seat),e.bot!==void 0&&(o.bot=e.bot);for(const a of this.root.querySelectorAll(".match-controls .qc-select")){const r=a.dataset.key;r&&a.value.trim()!==""&&(o[r]=a.value.trim())}const i=e.seat==="watch"?"watch":"play";this.startMatch(t,i,o)}}showCpuNote(t){const e=this.root.querySelector(".cpu-note");e&&(e.textContent=t??"CPU FALLBACK ACTIVE: No compatible WebGPU device was detected. AlphaZero is running on the CPU, so only the Trivial and Light levels are offered. Open it in a WebGPU browser (recent Chrome/Edge) for the full difficulty ladder.",e.hidden=!1)}showGpuMismatchNote(t){const e=this.root.querySelector(".gpu-mismatch-note");e&&(e.textContent=t,e.hidden=!1)}async checkGoConformance(){const t=this.gen;try{const[e,s]=await Promise.all([ke(),Mt()]),o=xi(s);if(yi(o)?.pass){console.info("[go-selfcheck] cached pass; skipping re-run");return}const a=await Fe(e,s,{limit:10});if(t!==this.gen)return;vi(o,a),console.info(`[go-selfcheck] pass=${a.pass} maxDp=${a.maxDp.toExponential(2)} maxDv=${a.maxDv.toExponential(2)} over ${a.count} fixtures`),a.pass||this.reportGoMismatch(a)}catch(e){console.info(`[go-selfcheck] skipped: ${pt(e)}`)}}reportGoMismatch(t){const e=t.worst?` at ply ${t.worst.plies}, ${t.worst.size}×${t.worst.size}`:"";this.showGpuMismatchNote(`Inference check: this browser's GPU computes the AlphaZero network differently from the reference (max policy Δ ${t.maxDp.toExponential(2)}, value Δ ${t.maxDv.toExponential(2)}${e}). Move quality may be degraded on this device.`)}sideHtml(t){if(t.solo||me.has(t.id))return"";const e=Ko(t.id)?"":`<form class="free-input">
          <input placeholder="or type a move…" autocomplete="off" />
          <button type="submit">send</button>
        </form>`;return`<aside class="side${this.debugOn?" debug-on":""}">
        <div class="status">Starting…</div>
        <div class="log-head">
          <span class="log-title">Log</span>
          <label class="debug-toggle">
            <input type="checkbox" class="debug-check"${this.debugOn?" checked":""} />
            <span class="debug-pill"></span>
            <span class="debug-word">debug</span>
          </label>
        </div>
        <div class="debug-readout" aria-live="polite"></div>
        <div class="log" aria-live="polite"></div>
        ${e}
      </aside>`}fillSeatSlots(t,e){let s=[...this.root.querySelectorAll(".board .seat-slot[data-seat]")];if(s.length===0){const a=ht(t,e),r=document.createElement("div");r.className="roster",r.innerHTML=Array.from({length:a},(l,c)=>`<label class="seat"><span class="seat-name">${w(this.seatName(t,c))}</span><span class="seat-slot" data-seat="${c}"></span></label>`).join(""),this.root.querySelector(".match-bar").insertAdjacentElement("afterend",r),s=[...r.querySelectorAll(".seat-slot[data-seat]")]}const o=this.seatStates(t,e),i=dt(t);for(const a of s){const r=Number(a.dataset.seat),l=document.createElement("select");l.className="seat-select";const c=[{value:"__you__",label:"You"}].concat(i.map(g=>({value:g.value,label:g.label})));for(const g of c){const f=document.createElement("option");f.value=g.value,f.textContent=g.label,f.selected=g.value===o[r],l.append(f)}l.onchange=()=>this.applySeatChange(t,e,r,l.value);const d=this.seatLevelSelect(t,e,r,o[r]),h=this.seatInfoButton(t,o[r]),p=[l,d,h].filter(g=>g!==null);a.replaceChildren(p.length>1?this.fragment(p):l)}}seatInfoButton(t,e){const s=Zo(t.id,e);if(!s)return null;const o=document.createElement("span");o.className="seat-info";const i=document.createElement("button");i.type="button",i.className="seat-info-btn",i.title="About this opponent",i.setAttribute("aria-label","About this opponent"),i.textContent="i";const a=document.createElement("div");a.className="bot-info-pop",a.hidden=!0,a.innerHTML=`<strong>${w(Ct(e))}</strong><span>${w(s)}</span>`;const r=l=>{o.contains(l.target)||(a.hidden=!0,document.removeEventListener("pointerdown",r))};return i.onclick=()=>{if(a.hidden=!a.hidden,!a.hidden){const l=i.getBoundingClientRect(),c=Math.min(320,window.innerWidth-24);a.style.width=`${c}px`,a.style.left=`${Math.max(12,Math.min(l.left,window.innerWidth-c-12))}px`,a.style.top=`${l.bottom+8}px`,document.addEventListener("pointerdown",r);const d=window.innerHeight-l.bottom-20;a.style.maxHeight=`${Math.max(120,d)}px`}},o.append(i,a),o}fragment(t){const e=document.createDocumentFragment();for(const s of t)e.append(s);return e}seatLevelSelect(t,e,s,o){if(o==="__you__")return null;const i=B[`${t.id}/${o}`];if(!i)return null;const r=o==="azero-gpu"&&D()?F:i.levels,l=this.seatLevel(t,e,s,o),c=document.createElement("select");c.className="seat-level",c.setAttribute("aria-label","Difficulty");for(const[d,h]of r){const p=document.createElement("option");p.value=h,p.textContent=d,p.selected=h===l,c.append(p)}if(!r.some(([,d])=>d===l)){const d=document.createElement("option");d.value=l,d.textContent=`Custom (${l})`,d.selected=!0,c.prepend(d)}return c.onchange=()=>this.applyLevelChange(t,e,s,c.value),c}seatLevel(t,e,s,o){const i=B[`${t.id}/${o}`];if(!i)return"";if(e.bots){const r=N(e.bots)[s];if(r){const l=G(r).opts[i.key];if(l!==void 0)return l}}const a=o==="azero-gpu"&&D();return e[i.key]??(a?String(_):ge(t.id,o))}applyLevelChange(t,e,s,o){const i=this.seatStates(t,e),a=i.indexOf("__you__"),r=i.find(h=>h!=="__you__")??dt(t)[0]?.value??"",l=i.map((h,p)=>{const g=h==="__you__"?r:h,f=p===s?o:this.seatLevel(t,e,p,g),b=this.seatBotOptions(t,e,p,g),u=B[`${t.id}/${g}`];return u&&f&&(b[u.key]=f),U(g,b)}),d={...this.gameLevelCarry(t,e),bots:l.join(",")};a>=0?this.startMatch(t,"play",{...d,seat:String(a)}):this.startMatch(t,"watch",d)}seatName(t,e){return t.solo?"Player":di(t.id,e)}seatStates(t,e){const s=ht(t,e);if(t.solo)return[e.bot??"__you__"];let o;if(e.bots){const a=N(e.bots);o=Array.from({length:s},(r,l)=>a[l]?G(a[l]).bot:xe(t,e))}else{const a=xe(t,e);o=Array.from({length:s},()=>a)}const i=e.seat==="watch"?-1:Number(e.seat??"0");return i>=0&&i<s&&(o[i]="__you__"),o}seatBotOptions(t,e,s,o){if(e.bots){const a=N(e.bots)[s];if(a){const r=G(a);if(r.bot===o)return{...r.opts}}return{}}const i={};for(const a of t.optsSchema)a.bots.includes(o)&&e[a.key]!==void 0&&(i[a.key]=e[a.key]);return i}clientBotConfigs(t,e){const s=this.seatStates(t,e),o=[];for(const[i,a]of s.entries()){if(a==="__you__"||!Se(t.id,a))continue;const r=this.seatBotOptions(t,e,i,a),l=new Set(t.optsSchema.filter(h=>!h.nativeOnly&&h.bots.includes(a)).map(h=>h.key)),c=Object.keys(r).filter(h=>!l.has(h));if(c.length)throw new Error(`unused option(s) for client bot '${a}' at seat ${i}: ${c.join(", ")}`);const d={...e,...r,bot:a};delete d.bots,delete d.seat;for(const h of t.optsSchema){if(h.nativeOnly||!h.bots.includes(a)||d[h.key]!==void 0)continue;const p=be[t.id]?.[h.key]??pi(h);p!==void 0&&(d[h.key]=p)}o.push({seat:i,bot:a,opts:d})}return o}gameLevelCarry(t,e){const s={};for(const o of t.optsSchema)o.bots.length===0&&o.key!=="seat"&&o.key!=="seed"&&!o.nativeOnly&&e[o.key]!==void 0&&(s[o.key]=e[o.key]);return s}applySeatChange(t,e,s,o){if(t.solo){o==="__you__"?this.startMatch(t,"play",{}):this.startMatch(t,"watch",{bot:o});return}const i=ht(t,e),a=dt(t),r=u=>a.find(m=>m.value===u)?.sendsBot??!1,l=this.seatStates(t,e),c=[...l];if(c[s]=o,o==="__you__"){const u=a[0]?.value??"__solver__";for(let m=0;m<i;m++)m!==s&&c[m]==="__you__"&&(c[m]=u)}const d=c.indexOf("__you__"),p=c.filter(u=>u!=="__you__")[0]??a[0]?.value??"",g=c.map((u,m)=>{const x=u==="__you__"?p:u,v=u!=="__you__"&&x!==l[m],A=v?{}:this.seatBotOptions(t,e,m,x),$=B[`${t.id}/${x}`];if($){const z=v?ge(t.id,x):this.seatLevel(t,e,m,x);z&&(A[$.key]=z)}return{bot:x,botOpts:A,human:u==="__you__"}}),f=g.filter(u=>!u.human);if(f.length>0&&f.every(u=>u.bot===f[0].bot&&ui(u.botOpts,f[0].botOpts))){const u=this.gameLevelCarry(t,e),m=r(f[0].bot)?{bot:f[0].bot,...f[0].botOpts}:{};d>=0?this.startMatch(t,"play",{...u,...m,seat:String(d)}):this.startMatch(t,"watch",{...u,...m})}else{const u=g.map(x=>U(x.bot,x.botOpts)),m=this.gameLevelCarry(t,e);d>=0?this.startMatch(t,"play",{...m,bots:u.join(","),seat:String(d)}):this.startMatch(t,"watch",{...m,bots:u.join(",")})}}wireDrawer(t,e){const s=this.root.querySelector(".drawer"),o=s.querySelector(".drawer-fields"),i=c=>c?`<small class="opt-note">${w(c)}</small>`:"",a=(c,d,h="")=>`<label class="opt-row"><span>${w(c)}</span>${d}${i(h)}</label>`,r=(c,d,h,p)=>a(d,`<select name="d-${w(c)}">${ve(h,p)}</select>`),l=()=>{const c=xt(t,e),d=e.bots?void 0:B[`${t.id}/${c}`],h=li(t.optsSchema,e).filter(u=>(u.bots.length===0||!e.bots&&u.bots.includes(c))&&!(d&&u.key===d.key)),p=c==="azero-gpu"&&D(),g=d?r("difficulty-target","difficulty",p?F:d.levels,e[d.key]??(p?String(_):d.levels[1][1])):"",f=h.map(u=>{const m=fe(t.id,u.key);return m?r(u.key,u.key,m.map(x=>[x,x]),u.value):a(u.key,`<input name="d-${w(u.key)}" value="${w(u.value)}" autocomplete="off" />`,u.note)}),b=g+f.join("");o.innerHTML=b||'<p class="muted">No settings for this game.</p>',o.dataset.diffKey=d?d.key:"",s.hidden=!1};this.root.querySelector(".gear").onclick=l,s.querySelector(".drawer-close").onclick=()=>{s.hidden=!0},s.onclick=c=>{c.target===s&&(s.hidden=!0)},s.querySelector(".drawer-apply").onclick=()=>{const c={};e.seat!==void 0&&(c.seat=e.seat),e.bot!==void 0&&(c.bot=e.bot),e.bots!==void 0&&(c.bots=e.bots);const d=o.dataset.diffKey??"",h=o.querySelectorAll("input, select");for(const g of h){let f=g.name.replace(/^d-/,"");if(f==="difficulty-target"){if(!d)continue;f=d}g.value.trim()!==""&&(c[f]=g.value.trim())}const p=t.solo?e.bot?"watch":"play":e.seat==="watch"?"watch":"play";this.startMatch(t,p,c)}}async runLoop(t){const e=s=>{t===this.gen&&this.setStatus(pt(s),"error")};for(;t===this.gen;){let s;try{s=await this.host.step()}catch(r){e(r);return}if(t!==this.gen)return;if(s){try{this.log(s),await this.clientBot?.onMove(s);const r=await this.host.state();if(t!==this.gen)return;const l=r.isOver?Promise.resolve():this.host.prepare();await this.frontend.animate(s,r),await l}catch(r){e(r);return}continue}const o=await this.host.state();if(t!==this.gen)return;if(this.frontend.render(o),o.isOver){const r=await this.clientBot?.finalResult?.()||"";if(t!==this.gen)return;const l=r||o.result||"Game over";this.setStatus(l,"result"),this.logText(`— ${l}`);return}if(this.clientBot&&o.toAct>=0&&o.toAct!==o.humanSeat){this.setStatus("Thinking…");try{const r=performance.now(),l=await this.clientBot.chooseMove(o),c=performance.now()-r;if(t!==this.gen)return;const d=await this.host.apply(l);if(t!==this.gen)return;this.log(d,c),await this.clientBot.onMove(d);const h=await this.host.state();if(t!==this.gen)return;await this.frontend.animate(d,h)}catch(r){e(r);return}continue}this.setStatus("Your turn");const i=this.host.prepare();this.frontend.promptAction(o.labels);const a=await new Promise(r=>this.submitResolve=r);if(t!==this.gen)return;o.numSeats>1&&this.setStatus("Thinking…");try{await i;const r=await this.host.apply(a);if(t!==this.gen)return;this.log(r),await this.clientBot?.onMove(r);const l=await this.host.state();if(t!==this.gen)return;const c=l.isOver?Promise.resolve():this.host.prepare();await this.frontend.animate(r,l),await c}catch(r){e(r)}}}async loadArtifacts(t,e){for(const s of ri(t.id,e)){const o=et(Ae[s]),i=await fetch(o);if(!i.ok)throw new Error(`artifact ${o} missing (HTTP ${i.status})`);await this.host.artifact(s,await i.arrayBuffer())}}submit(t){const e=this.submitResolve;e&&(this.submitResolve=null,e(t))}animationScale(){return window.matchMedia("(prefers-reduced-motion: reduce)").matches?0:this.speedScale}log(t,e){this.logText(t.text),t.detail&&this.logText(t.detail,"detail");const s=[`seat ${t.seat}`,`label ${t.label}`];e!==void 0&&s.push(`think ${Math.round(e)}ms`),this.logText(s.join(" · "),"meta")}debugLog(t){this.logText(t,"meta")}logText(t,e){if(!this.logEl)return;const s=document.createElement("div");s.className=e?`log-line log-${e} log-debug`:"log-line",s.textContent=t,this.logEl.append(s),this.logEl.scrollTop=this.logEl.scrollHeight}setDebug(t){if(t!==this.debugOn){this.debugOn=t,localStorage.setItem("arcadeDebug",t?"1":"0"),this.sideEl?.classList.toggle("debug-on",t);for(const e of this.debugSubs)e(t)}}onDebugChange(t){return this.debugSubs.add(t),t(this.debugOn),()=>this.debugSubs.delete(t)}setDebugReadout(t){this.readoutEl&&this.readoutEl.replaceChildren(...t.map(e=>{const s=document.createElement("div");return s.className="readout-row",s.textContent=e,s}))}setStatus(t,e="info"){this.statusEl&&(this.statusEl.textContent=t,this.statusEl.className=`status status-${e}`)}teardownMatch(){this.clientBot?.cancel(),this.clientBot=null,this.frontend?.unmount(),this.frontend=null,this.submitResolve=null,this.debugSubs.clear()}teardown(){this.gen++,this.tourney?.destroy(),this.tourney=null,this.slither?.destroy(),this.slither=null,this.teardownMatch(),this.logEl=null,this.statusEl=null,this.sideEl=null,this.readoutEl=null}}function pt(n){return n instanceof Error?n.message:String(n)}function xi(n){const t=new Uint8Array(n),e=t.length,s=e?[t[0],t[e>>2|0],t[e>>1|0],t[3*e>>2|0],t[e-1]].join("."):"0";return`azeroGoSelfcheck:${e}:${s}`}function yi(n){try{const t=localStorage.getItem(n);return t?JSON.parse(t):null}catch{return null}}function vi(n,t){try{localStorage.setItem(n,JSON.stringify(t))}catch{}}new mi(document.getElementById("app")).start().catch(n=>{document.getElementById("app").innerHTML=`<div class="boot">Failed to start the engine: ${n instanceof Error?n.message:n}</div>`});
