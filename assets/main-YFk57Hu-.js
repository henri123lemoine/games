import{A as kt,s as Et,a as St}from"./azgpu-qjTz9flN.js";const $t=600,qt=8;let R=null;function Mt(){return R??=(async()=>{const n="./azero/azero-chess.azweb",t=await fetch(n);if(!t.ok)throw new Error(`weights ${n} missing (HTTP ${t.status})`);const e=await kt.init(await t.arrayBuffer());return e.lost.then(()=>{R=null}),e})(),R.catch(()=>{R=null}),R}class Lt{constructor(t,e){this.host=t,this.gpu=e}cancelled=!1;onMove(t){return this.host.azPush(t.label)}async chooseMove(t){let e=new Float32Array(0),s=new Float32Array(0);for(;;){if(this.cancelled)throw new Error("cancelled");const o=await this.host.azAdvance(e,s);if(o.n===0)break;if(this.cancelled)throw new Error("cancelled");const{logits:a,values:i}=await this.gpu.forward(o.features,o.n),r=[];for(let c=0;c<o.n;c++){const l=o.support.subarray(o.offsets[c],o.offsets[c+1]);r.push(...Et(a,l,c*St))}e=Float32Array.from(r),s=i.slice(0,o.n)}return(await this.host.azBest()).uci}cancel(){this.cancelled=!0}}async function Ct(n,t){const e=await Mt(),s=Number(t.sims)>0?Number(t.sims):$t,o=Number(t.seed)>>>0||1;return await n.azNew(s,qt,o),new Lt(n,e)}const Tt=new Map([["chess/azero-gpu",Ct]]);function At(n,t){return t&&Tt.get(`${n}/${t}`)||null}class vt{worker;nextId=1;pending=new Map;constructor(){this.worker=new Worker(new URL(""+new URL("worker-DF19Nucr.js",import.meta.url).href,import.meta.url),{type:"module"}),this.worker.onmessage=t=>{const e=this.pending.get(t.data.id);e&&(this.pending.delete(t.data.id),t.data.ok?e.resolve(t.data.data):e.reject(new Error(t.data.error)))},this.worker.onerror=t=>this.rejectAll(`engine worker error: ${t.message||"unknown"}`),this.worker.onmessageerror=()=>this.rejectAll("engine worker message error")}rejectAll(t){for(const e of this.pending.values())e.reject(new Error(t));this.pending.clear()}call(t){const e=this.nextId++;return new Promise((s,o)=>{this.pending.set(e,{resolve:s,reject:o}),this.worker.postMessage({...t,id:e})})}manifest(){return this.call({op:"manifest"})}create(t,e){return this.call({op:"create",game:t,opts:e})}step(){return this.call({op:"step"})}state(){return this.call({op:"state"})}apply(t){return this.call({op:"apply",input:t})}artifact(t,e){return this.call({op:"artifact",key:t,bytes:e})}pairs(t,e,s,o,a,i,r){return this.call({op:"pairs",game:t,opts:e,a:s,b:o,seed:a,lo:i,hi:r})}field(t,e,s,o,a,i,r){return this.call({op:"field",game:t,opts:e,a:s,b:o,seed:a,lo:i,hi:r})}elo(t,e,s){return this.call({op:"elo",w:t,d:e,l:s})}fitElo(t){return this.call({op:"fitElo",records:t})}azNew(t,e,s){return this.call({op:"azNew",sims:t,leaves:e,seed:s})}azPush(t){return this.call({op:"azPush",uci:t})}azAdvance(t,e){return this.call({op:"azAdvance",priors:t,values:e})}azBest(){return this.call({op:"azBest"})}terminate(){this.worker.terminate(),this.rejectAll("engine terminated")}}function b(n){return new Promise(t=>setTimeout(t,n))}const zt=["up","down","left","right"],Bt={ArrowUp:"up",ArrowDown:"down",ArrowLeft:"left",ArrowRight:"right",w:"up",s:"down",a:"left",d:"right",W:"up",S:"down",A:"left",D:"right"},Dt={up:"↑",down:"↓",left:"←",right:"→"},Y=110,D=140;function V(n){if(!n||typeof n!="object")return null;const t=n;return!Array.isArray(t.cells)||t.cells.length!==16||typeof t.score!="number"||typeof t.over!="boolean"?null:t}function U(n){return typeof n=="string"&&zt.includes(n)}function Pt(n,t){const e=[];for(let s=0;s<4;s++)n==="left"?e.push(t*4+s):n==="right"?e.push(t*4+(3-s)):n==="up"?e.push(s*4+t):e.push((3-s)*4+t);return e}function F(n,t){const e=new Array(16).fill(0),s=[];for(let o=0;o<4;o++){const a=Pt(t,o);let i=0,r=0;for(const c of a){const l=n[c];l!==0&&(r===l?(e[a[i-1]]=l*2,s.push({from:c,to:a[i-1],value:l,merged:!0}),r=0):(e[a[i]]=l,s.push({from:c,to:a[i],value:l,merged:!1}),i++,r=l))}}return{moves:s,after:e}}function X(n,t){return n.every((e,s)=>e===t[s])}function Rt(n){const t=n<=2048?`g2048-v${n}`:"g2048-vmax",e=String(n).length,s=e<=2?"g2048-d2":e===3?"g2048-d3":e===4?"g2048-d4":"g2048-d5";return`g2048-tile-inner ${t} ${s}`}const It=`
.g2048 {
  margin: auto;
  width: min(100%, 430px);
  display: flex;
  flex-direction: column;
  gap: 14px;
  user-select: none;
}
.g2048-top {
  display: flex;
  align-items: stretch;
  gap: 10px;
}
.g2048-logo {
  margin-right: auto;
  align-self: center;
  font-size: 1.8rem;
  font-weight: 850;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #edc22e, #f65e3b);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.g2048-scorebox {
  position: relative;
  min-width: 84px;
  padding: 6px 14px;
  text-align: center;
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) - 2px);
}
.g2048-scorebox small {
  display: block;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.62rem;
  color: var(--text-dim);
}
.g2048-scorebox b {
  font-size: 1.15rem;
  font-variant-numeric: tabular-nums;
}
.g2048-gain {
  position: absolute;
  left: 0;
  right: 0;
  top: 18px;
  text-align: center;
  font-weight: 800;
  color: var(--good);
  pointer-events: none;
  animation: g2048-rise 600ms ease-out forwards;
}
@keyframes g2048-rise {
  from { opacity: 1; transform: translateY(0); }
  to { opacity: 0; transform: translateY(-26px); }
}
.g2048-board {
  position: relative;
  aspect-ratio: 1;
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  container-type: size;
}
.g2048-cells, .g2048-tiles {
  position: absolute;
  inset: 6px;
}
.g2048-cell, .g2048-tile {
  position: absolute;
  width: 25%;
  height: 25%;
}
.g2048-slide {
  transition: transform ${Y}ms ease-in-out;
  will-change: transform;
}
.g2048-cell-inner, .g2048-tile-inner {
  position: absolute;
  inset: 5px;
  border-radius: 6px;
}
.g2048-cell-inner {
  background: rgba(120, 120, 120, 0.12);
}
.g2048-tile-inner {
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  font-variant-numeric: tabular-nums;
  line-height: 1;
}
.g2048-d2 { font-size: 38px; font-size: 10cqw; }
.g2048-d3 { font-size: 32px; font-size: 8.2cqw; }
.g2048-d4 { font-size: 26px; font-size: 6.6cqw; }
.g2048-d5 { font-size: 21px; font-size: 5.4cqw; }
.g2048-v2 { background: #eee4da; color: #776e65; }
.g2048-v4 { background: #ede0c8; color: #776e65; }
.g2048-v8 { background: #f2b179; color: #f9f6f2; }
.g2048-v16 { background: #f59563; color: #f9f6f2; }
.g2048-v32 { background: #f67c5f; color: #f9f6f2; }
.g2048-v64 { background: #f65e3b; color: #f9f6f2; }
.g2048-v128 { background: #edcf72; color: #f9f6f2; box-shadow: 0 0 12px rgba(237, 207, 114, 0.28); }
.g2048-v256 { background: #edcc61; color: #f9f6f2; box-shadow: 0 0 14px rgba(237, 204, 97, 0.34); }
.g2048-v512 { background: #edc850; color: #f9f6f2; box-shadow: 0 0 16px rgba(237, 200, 80, 0.4); }
.g2048-v1024 { background: #edc53f; color: #f9f6f2; box-shadow: 0 0 18px rgba(237, 197, 63, 0.46); }
.g2048-v2048 { background: #edc22e; color: #f9f6f2; box-shadow: 0 0 22px rgba(237, 194, 46, 0.55); }
.g2048-vmax {
  background: #21262e;
  color: #58a6ff;
  border: 1px solid #58a6ff;
  box-shadow: 0 0 18px rgba(88, 166, 255, 0.4);
}
.g2048-pop { animation: g2048-pop ${D}ms ease-out backwards; }
@keyframes g2048-pop {
  from { transform: scale(0); }
  to { transform: scale(1); }
}
.g2048-merge { animation: g2048-merge ${D}ms ease-in-out; }
@keyframes g2048-merge {
  0% { transform: scale(1); }
  50% { transform: scale(1.22); }
  100% { transform: scale(1); }
}
.g2048-shake { animation: g2048-shake 180ms ease-in-out; }
@keyframes g2048-shake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-6px); }
  75% { transform: translateX(6px); }
}
.g2048-overlay {
  position: absolute;
  inset: 0;
  z-index: 5;
  display: none;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  border-radius: var(--radius);
  background: rgba(255, 255, 255, 0.72);
  backdrop-filter: blur(2px);
}
.dark .g2048-overlay { background: rgba(13, 17, 23, 0.74); }
.g2048-overlay.g2048-show { display: flex; }
.g2048-overlay-title { font-size: 1.5rem; font-weight: 800; }
.g2048-overlay-sub { color: var(--text-dim); }
.g2048-pad {
  display: grid;
  grid-template-columns: repeat(3, 56px);
  grid-template-rows: repeat(2, 42px);
  gap: 6px;
  justify-content: center;
}
.g2048-pad.g2048-hidden { display: none; }
.g2048-btn {
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) - 2px);
  color: var(--text);
  font-size: 1.05rem;
  transition: border-color 0.12s, color 0.12s;
}
.g2048-btn-up { grid-column: 2; grid-row: 1; }
.g2048-btn-left { grid-column: 1; grid-row: 2; }
.g2048-btn-down { grid-column: 2; grid-row: 2; }
.g2048-btn-right { grid-column: 3; grid-row: 2; }
.g2048-btn:not(:disabled):hover { border-color: var(--accent); color: var(--accent); }
.g2048-btn:disabled { opacity: 0.32; cursor: default; }
@media (max-width: 480px) {
  .g2048-pad { grid-template-columns: repeat(3, 48px); grid-template-rows: repeat(2, 38px); }
}
`;function Ht(){if(document.getElementById("g2048-frontend-style"))return;const n=document.createElement("style");n.id="g2048-frontend-style",n.textContent=It,document.head.append(n)}class jt{ctx;cells=new Array(16).fill(0);score=0;pending=null;boardEl;tilesEl;scoreEl;scoreBoxEl;bestEl;overlayEl;overlaySubEl;padBtns=new Map;mount(t,e){this.ctx=e,Ht(),t.innerHTML=`
      <div class="g2048">
        <div class="g2048-top">
          <span class="g2048-logo">2048</span>
          <div class="g2048-scorebox g2048-scorebox-score">
            <small>score</small><b class="g2048-score">0</b>
          </div>
          <div class="g2048-scorebox">
            <small>best tile</small><b class="g2048-best">0</b>
          </div>
        </div>
        <div class="g2048-board">
          <div class="g2048-cells"></div>
          <div class="g2048-tiles"></div>
          <div class="g2048-overlay">
            <span class="g2048-overlay-title">Game over</span>
            <span class="g2048-overlay-sub"></span>
          </div>
        </div>
        <div class="g2048-pad"></div>
      </div>`,this.boardEl=t.querySelector(".g2048-board"),this.tilesEl=t.querySelector(".g2048-tiles"),this.scoreEl=t.querySelector(".g2048-score"),this.scoreBoxEl=t.querySelector(".g2048-scorebox-score"),this.bestEl=t.querySelector(".g2048-best"),this.overlayEl=t.querySelector(".g2048-overlay"),this.overlaySubEl=t.querySelector(".g2048-overlay-sub");const s=t.querySelector(".g2048-cells");for(let a=0;a<16;a++){const i=document.createElement("div");i.className="g2048-cell",this.setPos(i,a),i.innerHTML='<div class="g2048-cell-inner"></div>',s.append(i)}const o=t.querySelector(".g2048-pad");if(e.humanSeat<0)o.classList.add("g2048-hidden");else for(const a of["up","left","down","right"]){const i=document.createElement("button");i.type="button",i.className=`g2048-btn g2048-btn-${a}`,i.textContent=Dt[a],i.title=a,i.disabled=!0,i.onclick=()=>this.trySubmit(a),o.append(i),this.padBtns.set(a,i)}window.addEventListener("keydown",this.onKey)}render(t){const e=V(t.viewData);if(!e)return;t.toAct!==t.humanSeat&&this.setPending(null);const s=[];let o=!0;for(let i=0;i<16;i++)this.cells[i]===0&&e.cells[i]!==0?s.push(i):this.cells[i]!==e.cells[i]&&(o=!1);const a=o&&s.length>0&&s.length<=2&&this.ctx.animationScale()>0?new Set(s):void 0;this.applyView(e,a)}async animate(t,e){const s=V(e.viewData);if(!s)return;const o=this.ctx.animationScale(),a=this.eventDir(t),i=s.score-this.score;if(!a||o===0){this.applyView(s);return}i>0&&this.showGain(i);let r=this.cells,c=F(r,a);if(!X(c.after,s.cells)){const d=this.findSpawn(r,a,s.cells);if(!d){const p=r.every(g=>g===0)?new Set(s.cells.flatMap((g,m)=>g!==0?[m]:[])):void 0;this.applyView(s,p);return}r=d.cells,c=F(r,a),this.cells=r,this.rebuildTiles(r,new Set([d.cell])),await b(D*o)}this.tilesEl.replaceChildren();const l=[];for(const d of c.moves){const p=this.makeTile(d.value,d.from);p.classList.add("g2048-slide"),p.style.transitionDuration=`${Y*o}ms`,this.tilesEl.append(p),l.push({el:p,move:d})}this.tilesEl.offsetWidth;for(const{el:d,move:p}of l)this.setPos(d,p.to);await b(Y*o+25);const h=new Set(c.moves.filter(d=>d.merged).map(d=>d.to));this.applyView(s,void 0,h),h.size>0&&await b(D*o)}promptAction(t){this.setPending(t)}unmount(){window.removeEventListener("keydown",this.onKey)}onKey=t=>{if(this.ctx.humanSeat<0||t.metaKey||t.ctrlKey||t.altKey)return;const e=t.target;if(e&&(e.tagName==="INPUT"||e.tagName==="TEXTAREA"||e.isContentEditable))return;const s=Bt[t.key];s&&(t.preventDefault(),this.trySubmit(s))};trySubmit(t){if(!this.pending)return;const e=this.pending.indexOf(t);if(e<0){this.shake();return}this.setPending(null),this.ctx.submit(String(e))}setPending(t){this.pending=t;for(const[e,s]of this.padBtns)s.disabled=!t||!t.includes(e)}eventDir(t){const e=t.data;if(e&&typeof e=="object"&&"dir"in e){const s=e.dir;if(U(s))return s}return U(t.label)?t.label:null}findSpawn(t,e,s){for(let o=0;o<16;o++)if(t[o]===0)for(const a of[2,4]){const i=t.slice();if(i[o]=a,X(F(i,e).after,s))return{cells:i,cell:o}}return null}applyView(t,e,s){this.cells=t.cells.slice(),this.score=t.score,this.rebuildTiles(this.cells,e,s),this.scoreEl.textContent=String(t.score),this.bestEl.textContent=String(Math.max(0,...t.cells)),this.overlayEl.classList.toggle("g2048-show",t.over),t.over&&(this.overlaySubEl.textContent=`score ${t.score}`)}rebuildTiles(t,e,s){const o=this.ctx.animationScale();this.tilesEl.replaceChildren();for(let a=0;a<16;a++){if(t[a]===0)continue;const i=this.makeTile(t[a],a),r=i.firstElementChild;o>0&&e?.has(a)?(r.classList.add("g2048-pop"),r.style.animationDuration=`${D*o}ms`):o>0&&s?.has(a)&&(r.classList.add("g2048-merge"),r.style.animationDuration=`${D*o}ms`),this.tilesEl.append(i)}}makeTile(t,e){const s=document.createElement("div");s.className="g2048-tile",this.setPos(s,e);const o=document.createElement("div");return o.className=Rt(t),o.textContent=String(t),s.append(o),s}setPos(t,e){const s=Math.floor(e/4),o=e%4;t.style.transform=`translate(${o*100}%, ${s*100}%)`}showGain(t){const e=document.createElement("span");e.className="g2048-gain",e.textContent=`+${t}`,e.addEventListener("animationend",()=>e.remove()),this.scoreBoxEl.append(e),setTimeout(()=>e.remove(),900)}shake(){this.boardEl.classList.remove("g2048-shake"),this.boardEl.offsetWidth,this.boardEl.classList.add("g2048-shake")}}function Nt(){return new jt}const Ft=["q","r","b","n","p"],Ot={q:1,r:2,b:2,n:2,p:8},Gt={q:9,r:5,b:3,n:3,p:1},Yt=["q","r","b","n"],W=/^[a-h][1-8][a-h][1-8][qrbn]?$/,_t=240,Wt=120,Vt=160,Ut={p:`<circle class="pcb" cx="22.5" cy="15.5" r="4.5"/>
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
<rect class="pcb" x="11" y="35.5" width="23" height="4.5" rx="2"/>`};function O(n,t){const e=Ut[n]??"";return`<svg class="chess-pc ${t?"chess-pc-w":"chess-pc-b"}" viewBox="0 0 45 45" aria-hidden="true">${e}</svg>`}function T(n){return(n.charCodeAt(1)-49)*8+(n.charCodeAt(0)-97)}function I(n,t){return n.charAt((7-Math.floor(t/8))*8+t%8)}function K(n){if(typeof n!="object"||n===null)return null;const t=n;return typeof t.board!="string"||t.board.length!==64?null:{board:t.board,turn:t.turn==="b"?"b":"w",check:t.check===!0}}function Xt(n){if(typeof n!="object"||n===null)return null;const t=n;if(typeof t.from!="string"||typeof t.to!="string"||!W.test(t.from+t.to))return null;const e=s=>typeof s=="string"&&s.length===2?s:null;return{from:t.from,to:t.to,capturedSquare:e(t.capturedSquare),castleRookFrom:e(t.castleRookFrom),castleRookTo:e(t.castleRookTo)}}function Kt(n){return W.test(n)?{from:n.slice(0,2),to:n.slice(2,4),capturedSquare:null,castleRookFrom:null,castleRookTo:null}:null}class Qt{ctx;host;rootEl;boardEl;piecesEl;promoEl;bars;squareEls=[];pieceEls=new Map;flipped=!1;view=null;lastMove=null;gameOver=!1;moves=new Map;selected=null;inputArmed=!1;drag=null;skipSlide=!1;promoFromDrag=!1;mount(t,e){this.ctx=e,this.host=t,this.flipped=e.humanSeat===1,Zt();const s=`
      <span class="chess-turn-dot"></span>
      <span class="chess-bar-name"></span>
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
      </div>`,this.rootEl=t.querySelector(".chess-root"),this.boardEl=t.querySelector(".chess-board"),this.piecesEl=t.querySelector(".chess-pieces"),this.promoEl=t.querySelector(".chess-promo");const o=t.querySelector(".chess-bar-top"),a=t.querySelector(".chess-bar-bottom"),i=d=>({root:d,tray:d.querySelector(".chess-tray"),score:d.querySelector(".chess-score")});this.bars=this.flipped?{w:i(o),b:i(a)}:{w:i(a),b:i(o)};const r=d=>{const p=d==="w"?"White":"Black";return(d==="w"?0:1)===e.humanSeat?`You · ${p}`:`Bot · ${p}`};for(const d of["w","b"])this.bars[d].root.querySelector(".chess-bar-name").textContent=r(d);const c=t.querySelector(".chess-ranks"),l=t.querySelector(".chess-files");for(let d=0;d<8;d++){const p=this.flipped?d+1:8-d,g=this.flipped?7-d:d;c.insertAdjacentHTML("beforeend",`<span>${p}</span>`),l.insertAdjacentHTML("beforeend",`<span>${"abcdefgh"[g]}</span>`)}const h=t.querySelector(".chess-squares");this.squareEls=new Array(64);for(let d=0;d<8;d++)for(let p=0;p<8;p++){const g=this.flipped?7-p:p,m=this.flipped?d:7-d,y=m*8+g,u=document.createElement("div");u.className=`chess-sq ${(g+m)%2===1?"chess-sq-light":"chess-sq-dark"}`,u.dataset.sq=String(y),this.squareEls[y]=u,h.append(u)}this.boardEl.addEventListener("pointerdown",d=>this.onPointerDown(d)),this.boardEl.addEventListener("pointermove",d=>this.onPointerMove(d)),this.boardEl.addEventListener("pointerup",d=>this.onPointerUp(d)),this.boardEl.addEventListener("pointercancel",()=>this.cancelDrag(!0)),this.boardEl.addEventListener("contextmenu",d=>{this.drag&&d.preventDefault()}),this.promoEl.addEventListener("click",d=>{d.target===this.promoEl&&(this.closePromo(),this.select(null))})}render(t){const e=K(t.viewData);e&&(this.view=e,this.gameOver=t.isOver,this.syncAll())}async animate(t,e){const s=this.skipSlide;this.skipSlide=!1,this.disarm();const o=K(e.viewData);if(!o)return;const a=Xt(t.data)??Kt(t.label);a&&(this.lastMove={from:T(a.from),to:T(a.to)}),this.gameOver=e.isOver;const i=this.ctx.animationScale();a&&i>0&&!s&&await this.slide(a,i),this.view=o,this.syncAll(),i>0&&!s&&await b(Wt*i)}promptAction(t){if(!(this.ctx.humanSeat<0)){this.moves.clear();for(const e of t){if(!W.test(e))continue;const s=T(e.slice(0,2)),o=T(e.slice(2,4));let a=this.moves.get(s);a||(a=new Map,this.moves.set(s,a));const i=a.get(o);i?i.push(e):a.set(o,[e])}this.inputArmed=!0,this.rootEl.classList.add("chess-armed");for(const e of this.moves.keys())this.squareEls[e].classList.add("chess-sq-movable")}}unmount(){this.host.replaceChildren()}squareAt(t,e){const s=this.boardEl.getBoundingClientRect();if(t<s.left||t>=s.right||e<s.top||e>=s.bottom)return null;const o=Math.floor((t-s.left)/s.width*8),a=Math.floor((e-s.top)/s.height*8),i=this.flipped?7-o:o;return(this.flipped?a:7-a)*8+i}onPointerDown(t){if(!this.inputArmed||!this.promoEl.hidden||this.drag||t.pointerType==="mouse"&&t.button!==0)return;const e=this.squareAt(t.clientX,t.clientY);if(e===null)return;if(this.selected!==null){const i=this.moves.get(this.selected)?.get(e);if(i){i.length>1?this.openPromo(i):this.submitMove(i[0]);return}}if(!this.moves.has(e)){this.select(null);return}const s=this.pieceEls.get(e);if(!s)return;t.preventDefault();const o=this.selected===e;this.select(e);const a=s.cloneNode(!0);a.classList.add("chess-piece-ghost"),this.piecesEl.append(a),s.classList.add("chess-piece-drag"),this.boardEl.classList.add("chess-dragging"),this.drag={pointerId:t.pointerId,from:e,el:s,ghost:a,wasSelected:o,hover:null},this.boardEl.setPointerCapture(t.pointerId),this.moveDragTo(t.clientX,t.clientY)}onPointerMove(t){!this.drag||t.pointerId!==this.drag.pointerId||this.moveDragTo(t.clientX,t.clientY)}moveDragTo(t,e){if(!this.drag)return;const s=this.boardEl.getBoundingClientRect(),o=s.width/8,a=t-s.left-o/2,i=e-s.top-o/2;this.drag.el.style.transform=`translate(${a}px, ${i}px) scale(1.15)`;const r=this.squareAt(t,e);r!==this.drag.hover&&(this.drag.hover!==null&&this.squareEls[this.drag.hover].classList.remove("chess-sq-drop"),this.drag.hover=null,r!==null&&r!==this.drag.from&&this.moves.get(this.drag.from)?.has(r)&&(this.squareEls[r].classList.add("chess-sq-drop"),this.drag.hover=r))}onPointerUp(t){if(!this.drag||t.pointerId!==this.drag.pointerId)return;const e=this.drag;this.drag=null,this.endDragVisuals(e);const s=this.squareAt(t.clientX,t.clientY),o=s!==null&&s!==e.from?this.moves.get(e.from)?.get(s):void 0;if(s!==null&&o){this.settle(e.el,s),this.removeVictim(e.from,s),o.length>1?(this.promoFromDrag=!0,this.openPromo(o)):this.submitMove(o[0],!0);return}this.snapBack(e.el,e.from),s===e.from&&e.wasSelected&&this.select(null)}cancelDrag(t){if(!this.drag)return;const e=this.drag;this.drag=null,this.endDragVisuals(e),t?this.snapBack(e.el,e.from):(e.el.classList.remove("chess-piece-drag"),this.place(e.el,e.from))}endDragVisuals(t){t.hover!==null&&this.squareEls[t.hover].classList.remove("chess-sq-drop"),t.ghost.remove(),this.boardEl.classList.remove("chess-dragging")}settle(t,e){t.classList.remove("chess-piece-drag"),t.style.zIndex="5",this.place(t,e)}snapBack(t,e){t.classList.remove("chess-piece-drag");const s=Vt*this.ctx.animationScale();if(s<=0){this.place(t,e);return}t.style.zIndex="5",t.style.transitionDuration=`${s}ms`,t.offsetWidth,this.place(t,e),window.setTimeout(()=>{t.style.transitionDuration="",t.style.zIndex=""},s+30)}removeVictim(t,e){const s=this.pieceEls.get(e);if(s){s.remove(),this.pieceEls.delete(e);return}if(!this.view||I(this.view.board,t).toLowerCase()!=="p"||t%8===e%8)return;const a=Math.floor(t/8)*8+e%8,i=this.pieceEls.get(a);i&&(i.remove(),this.pieceEls.delete(a))}select(t){if(this.clearSelection(),t===null)return;const e=this.moves.get(t);if(e){this.selected=t,this.squareEls[t].classList.add("chess-sq-selected");for(const s of e.keys())this.squareEls[s].classList.add(this.pieceEls.has(s)?"chess-sq-capture":"chess-sq-target")}}submitMove(t,e=!1){this.skipSlide=e,this.promoFromDrag=!1,this.disarm(),this.ctx.submit(t)}openPromo(t){const e=T(t[0].slice(0,2)),s=this.view?I(this.view.board,e):"P",o=s===s.toUpperCase(),a=this.promoFromDrag,i=document.createElement("div");i.className="chess-promo-panel";for(const r of Yt){const c=t.find(h=>h.charAt(4)===r);if(!c)continue;const l=document.createElement("button");l.type="button",l.className="chess-promo-btn",l.innerHTML=O(r,o),l.onclick=()=>this.submitMove(c,a),i.append(l)}this.promoEl.replaceChildren(i),this.promoEl.hidden=!1}closePromo(){const t=this.promoFromDrag;this.promoFromDrag=!1,this.promoEl.hidden=!0,this.promoEl.replaceChildren(),t&&this.view&&this.syncPieces(this.view)}disarm(){this.cancelDrag(!1),this.inputArmed=!1,this.moves.clear(),this.clearSelection(),this.closePromo(),this.rootEl.classList.remove("chess-armed");for(const t of this.squareEls)t.classList.remove("chess-sq-movable")}clearSelection(){this.selected=null;for(const t of this.squareEls)t.classList.remove("chess-sq-selected","chess-sq-target","chess-sq-capture","chess-sq-drop")}syncAll(){this.view&&(this.clearSelection(),this.syncPieces(this.view),this.syncHighlights(this.view),this.syncBars(this.view))}syncPieces(t){this.pieceEls.clear();const e=document.createDocumentFragment();for(let s=0;s<64;s++){const o=I(t.board,s);if(o===".")continue;const a=document.createElement("div");a.className="chess-piece",a.innerHTML=O(o.toLowerCase(),o===o.toUpperCase()),this.place(a,s),this.pieceEls.set(s,a),e.append(a)}this.piecesEl.replaceChildren(e)}syncHighlights(t){for(const e of this.squareEls)e.classList.remove("chess-sq-last","chess-sq-check","chess-sq-mate");if(this.lastMove&&(this.squareEls[this.lastMove.from].classList.add("chess-sq-last"),this.squareEls[this.lastMove.to].classList.add("chess-sq-last")),t.check){const e=t.turn==="w"?"K":"k";for(let s=0;s<64;s++)I(t.board,s)===e&&(this.squareEls[s].classList.add("chess-sq-check"),this.gameOver&&this.squareEls[s].classList.add("chess-sq-mate"))}}syncBars(t){const e={};for(const i of t.board)i!=="."&&(e[i]=(e[i]??0)+1);const s=i=>{const r=[];let c=0;for(const l of Ft){const h=e[i==="w"?l.toUpperCase():l]??0,d=Math.max(0,(Ot[l]??0)-h);c+=d*(Gt[l]??0);for(let p=0;p<d;p++)r.push(l)}return{pieces:r,pts:c}},o=s("w"),a=s("b");for(const i of["w","b"]){const r=i==="w"?a:o,c=i==="w"?a.pts-o.pts:o.pts-a.pts,l=this.bars[i];l.tray.replaceChildren(...r.pieces.map(h=>{const d=document.createElement("span");return d.className="chess-tray-piece",d.innerHTML=O(h,i==="b"),d})),l.score.textContent=c>0?`+${c}`:"",l.root.classList.toggle("chess-bar-active",!this.gameOver&&t.turn===i)}}place(t,e){const s=this.flipped?7-e%8:e%8,o=this.flipped?Math.floor(e/8):7-Math.floor(e/8);t.style.transform=`translate(${s*100}%, ${o*100}%)`}async slide(t,e){const s=_t*e,o=T(t.from),a=T(t.to),i=this.pieceEls.get(o);if(!i)return;const r=t.capturedSquare!==null?T(t.capturedSquare):this.pieceEls.has(a)?a:null;if(r!==null&&r!==o){const l=this.pieceEls.get(r);l&&(l.style.transition=`opacity ${s}ms ease`,l.style.opacity="0")}const c=(l,h)=>{l.style.zIndex="3",l.style.transitionDuration=`${s}ms`,l.offsetWidth,this.place(l,h)};if(c(i,a),t.castleRookFrom!==null&&t.castleRookTo!==null){const l=this.pieceEls.get(T(t.castleRookFrom));l&&c(l,T(t.castleRookTo))}await b(s+30)}}function Jt(){return new Qt}const Q="chess-frontend-style";function Zt(){if(document.getElementById(Q))return;const n=document.createElement("style");n.id=Q,n.textContent=te,document.head.append(n)}const te=`
.chess-root {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin: auto;
  width: 100%;
  max-width: 560px;
  max-width: min(560px, calc(100dvh - 250px));
  min-width: 260px;
}

.chess-bar {
  display: flex;
  align-items: center;
  gap: 10px;
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
`,k=7,$=6,G=["Red","Yellow"];function J(n){if(!n||typeof n!="object")return null;const t=n;return typeof t.cells=="string"&&t.cells.length===k*$?t:null}function ee(n){if(!n||typeof n!="object")return null;const t=n;return typeof t.col=="number"&&typeof t.row=="number"&&typeof t.player=="number"?t:null}function Z(n,t){return($-1-t)*k+n}function se(n,t){if(!n||!t)return null;for(let e=0;e<k*$;e++)if(n.cells[e]==="."&&t.cells[e]!==".")return{col:e%k,row:$-1-Math.floor(e/k),player:t.cells[e]==="x"?0:1};return null}const tt="connect4-frontend-style",oe=`
.c4-root {
  align-self: center;
  width: min(100%, 580px);
  display: flex;
  flex-direction: column;
  gap: 14px;
  user-select: none;
}
.c4-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.c4-chip {
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
  aspect-ratio: ${k} / ${$};
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
    0 0 / calc(100% / ${k}) calc(100% / ${$});
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
    0 0 / calc(100% / ${k}) calc(100% / ${$});
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
  width: calc(100% / ${k});
  height: calc(100% / ${$});
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
`;function ie(){if(document.getElementById(tt))return;const n=document.createElement("style");n.id=tt,n.textContent=oe,document.head.append(n)}class ae{ctx;rootEl;discsEl;hitsEl;msgEl;fallbackEl;chips=[];discs=new Map;view=null;colToAction=null;ghost=null;anims=new Set;mount(t,e){this.ctx=e,ie(),t.innerHTML=`
      <div class="c4-root">
        <div class="c4-bar">
          <div class="c4-chip c4-chip-0"><span class="c4-swatch"></span><span></span></div>
          <div class="c4-msg"></div>
          <div class="c4-chip c4-chip-1"><span class="c4-swatch"></span><span></span></div>
        </div>
        <div class="c4-board">
          <div class="c4-layer c4-backdrop"></div>
          <div class="c4-layer c4-discs"></div>
          <div class="c4-layer c4-frame"></div>
          <div class="c4-layer c4-hits"></div>
        </div>
        <pre class="c4-fallback"></pre>
      </div>`,this.rootEl=t.querySelector(".c4-root"),this.discsEl=t.querySelector(".c4-discs"),this.hitsEl=t.querySelector(".c4-hits"),this.msgEl=t.querySelector(".c4-msg"),this.fallbackEl=t.querySelector(".c4-fallback"),this.chips=[t.querySelector(".c4-chip-0"),t.querySelector(".c4-chip-1")];for(let s=0;s<2;s++){const o=e.humanSeat===s?"You":"Bot";this.chips[s].lastElementChild.textContent=`${G[s]} · ${o}`}for(let s=0;s<k;s++){const o=document.createElement("div");o.className="c4-hit",o.addEventListener("pointerenter",()=>this.showGhost(s)),o.addEventListener("pointerleave",()=>this.hideGhost()),o.addEventListener("click",()=>this.clickColumn(s)),this.hitsEl.append(o)}}render(t){this.disableInput();const e=J(t.viewData);if(this.view=e,!e){this.rootEl.classList.add("c4-text-only"),this.fallbackEl.textContent=t.view;return}this.rootEl.classList.remove("c4-text-only"),this.rebuildDiscs(e),this.decorateWin(e,!0);for(let s=0;s<2;s++)this.chips[s].classList.toggle("c4-active",!t.isOver&&e.turn===s);this.msgEl.textContent=t.isOver?e.winner!==null?`${G[e.winner]} connects four!`:"Draw — board full":`${G[e.turn]} to move`}async animate(t,e){const s=this.view,o=J(e.viewData),a=ee(t.data)??se(s,o);this.render(e);const i=this.ctx.animationScale();if(!o||!a||i<=0)return;const r=this.discs.get(Z(a.col,a.row));if(!r)return;const c=o.winLine!==null;c&&this.decorateWin(o,!1);const l=$-a.row,h=(150+100*Math.sqrt(l))*i;await this.run(r.animate([{transform:`translateY(${-l*100-30}%)`,offset:0,easing:"cubic-bezier(0.5, 0, 0.9, 0.6)"},{transform:"translateY(0%)",offset:.62,easing:"cubic-bezier(0.1, 0.5, 0.5, 1)"},{transform:"translateY(-17%)",offset:.8,easing:"cubic-bezier(0.5, 0, 0.9, 0.6)"},{transform:"translateY(0%)",offset:1}],{duration:h/.62})),c&&(this.decorateWin(o,!0),await b(650*i))}promptAction(t){const e=new Map;t.forEach((s,o)=>{const a=/(\d+)/.exec(s);a&&e.set(Number(a[1])-1,o)}),this.colToAction=e,this.hitsEl.classList.add("c4-live")}unmount(){for(const t of this.anims)t.cancel();this.anims.clear()}rebuildDiscs(t){this.discsEl.replaceChildren(),this.discs.clear(),this.ghost=null;for(let e=0;e<k*$;e++){const s=t.cells[e];if(s===".")continue;const o=this.makeDisc(s==="x"?0:1,e%k,Math.floor(e/k));this.discs.set(e,o),this.discsEl.append(o)}}makeDisc(t,e,s){const o=document.createElement("div");return o.className=`c4-disc c4-p${t}`,o.style.left=`${e*100/k}%`,o.style.top=`${s*100/$}%`,o}decorateWin(t,e){if(!t.winLine)return;const s=new Set(t.winLine);for(const[o,a]of this.discs)a.classList.toggle("c4-win",e&&s.has(o)),a.classList.toggle("c4-dim",e&&!s.has(o))}showGhost(t){this.hideGhost();const e=this.view;if(!(!e||!this.colToAction?.has(t)||this.ctx.humanSeat<0))for(let s=0;s<$;s++){const o=Z(t,s);if(e.cells[o]==="."){this.ghost=this.makeDisc(this.ctx.humanSeat,t,Math.floor(o/k)),this.ghost.classList.add("c4-ghost"),this.discsEl.append(this.ghost);return}}}hideGhost(){this.ghost?.remove(),this.ghost=null}clickColumn(t){const e=this.colToAction?.get(t);e!==void 0&&(this.disableInput(),this.ctx.submit(String(e)))}disableInput(){this.colToAction=null,this.hideGhost(),this.hitsEl.classList.remove("c4-live")}async run(t){this.anims.add(t);try{await t.finished}catch{}finally{this.anims.delete(t)}}}function ne(){return new ae}class re{ctx;viewEl;actionsEl;mount(t,e){this.ctx=e,t.innerHTML=`
      <div class="generic">
        <pre class="generic-view"></pre>
        <div class="generic-actions"></div>
      </div>`,this.viewEl=t.querySelector(".generic-view"),this.actionsEl=t.querySelector(".generic-actions")}render(t){this.viewEl.textContent=t.view,t.toAct!==t.humanSeat&&this.actionsEl.replaceChildren()}async animate(t,e){this.render(e),await b(250*this.ctx.animationScale())}promptAction(t){const e=t.map((s,o)=>{const a=document.createElement("button");return a.className="action-btn",a.textContent=s,a.onclick=()=>this.ctx.submit(String(o)),a});this.actionsEl.replaceChildren(...e)}unmount(){}}function et(n){if(!n||typeof n!="object")return null;const t=n;return typeof t.size=="number"&&typeof t.cells=="string"&&t.cells.length===t.size*t.size&&Array.isArray(t.captures)?t:null}const st="go-frontend-style",L=1;function yt(n){return String.fromCharCode(97+n+(n>=8?1:0))}function le(n,t){return`${yt(n%t)}${Math.floor(n/t)+1}`}function ce(n,t){const e=n.charCodeAt(0)-97;if(e<0||e>25||n[0]==="i")return null;const s=e>8?e-1:e,o=parseInt(n.slice(1),10);return!Number.isFinite(o)||s>=t||o<1||o>t?null:(o-1)*t+s}function de(n){const t=[],e=L+n-1;for(let s=0;s<n;s++){const o=L+s;t.push(`M ${o} ${L} L ${o} ${e}`,`M ${L} ${o} L ${e} ${o}`)}return t.join(" ")}function he(n){const t=[],e=n>=13?3:2;if(n>=7)for(const s of[e,n-1-e])for(const o of[e,n-1-e])t.push(s*n+o);if(n%2===1&&n>=5){const s=(n-1)/2;t.push(s*n+s),n>=15&&(t.push(e*n+s,(n-1-e)*n+s),t.push(s*n+e,s*n+(n-1-e)))}return t}const pe=`
.go { display: flex; flex-direction: column; gap: 14px; }
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
.go-turn-dot { width: 11px; height: 11px; border-radius: 50%; flex: none;
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, .25), 0 1px 2px rgba(0, 0, 0, .4); }
.go-board-wrap { position: relative; width: 100%; max-width: min(74vh, 640px); margin: 0 auto; }
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
`;function ue(){if(document.getElementById(st))return;const n=document.createElement("style");n.id=st,n.textContent=pe,document.head.append(n)}class fe{ctx;svg;stonesG;fxG;ghostEl;markerEl;toastEl;passBtn;turnChip;plaques=[];capEls=[];size=0;view=null;lastMove=null;interactive=!1;labelIndex=new Map;legalPoints=new Set;stoneEls=new Map;mount(t,e){this.ctx=e,ue(),t.innerHTML=`
      <div class="go">
        <div class="go-hud">
          <div class="go-player" data-seat="0">
            <span class="go-stone-icon go-stone-icon-b"></span>
            <span class="go-pinfo"><span class="go-pname">Black</span><span class="go-psub"></span></span>
            <span class="go-pcaps"><b>0</b>captures</span>
          </div>
          <div class="go-turn-chip"><span class="go-turn-dot"></span><span class="go-turn-text"></span></div>
          <div class="go-player" data-seat="1">
            <span class="go-stone-icon go-stone-icon-w"></span>
            <span class="go-pinfo"><span class="go-pname">White</span><span class="go-psub"></span></span>
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
      </div>`,this.svg=t.querySelector(".go-svg"),this.toastEl=t.querySelector(".go-toast"),this.passBtn=t.querySelector(".go-pass"),this.turnChip=t.querySelector(".go-turn-chip"),this.plaques=[...t.querySelectorAll(".go-player")],this.capEls=this.plaques.map(s=>s.querySelector(".go-pcaps b"));for(const[s,o]of this.plaques.entries()){const a=o.querySelector(".go-psub"),i=[s===e.humanSeat?"you":"bot"];s===1&&i.push(`+${this.view?.komi??7.5} komi`),a.textContent=i.join(" · ")}e.humanSeat<0&&(this.passBtn.style.display="none"),this.passBtn.onclick=()=>{const s=this.labelIndex.get("pass");!this.interactive||s===void 0||(this.setInteractive(!1),this.ctx.submit(String(s)))}}xy(t){return{x:L+t%this.size,y:L+(this.size-1-Math.floor(t/this.size))}}buildBoard(t){this.size=t;const e=t-1+2*L;this.svg.setAttribute("viewBox",`0 0 ${e} ${e}`);const s=he(t).map(c=>{const{x:l,y:h}=this.xy(c);return`<circle cx="${l}" cy="${h}" r="${t>13?.08:.1}" fill="rgba(40,24,8,.78)"/>`}).join(""),o=[];for(let c=0;c<t;c++)o.push(`<text x="${L+c}" y="${L+t-1+.72}">${yt(c)}</text>`,`<text x="${L-.66}" y="${L+(t-1-c)+.11}">${c+1}</text>`);const a=[];for(let c=0;c<t*t;c++){const{x:l,y:h}=this.xy(c);a.push(`<rect class="go-hit" data-p="${c}" x="${l-.5}" y="${h-.5}" width="1" height="1"/>`)}this.svg.innerHTML=`
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
      <path d="${de(t)}" stroke="rgba(46,28,8,.8)" stroke-width="0.032" fill="none" stroke-linecap="square"/>
      ${s}
      <g fill="rgba(46,28,8,.55)" font-size="0.32" text-anchor="middle" font-family="inherit">${o.join("")}</g>
      <g class="go-stones" filter="url(#go-shadow)"></g>
      <g class="go-fx"></g>
      <circle class="go-marker" r="0.17" fill="none" stroke-width="0.07" opacity="0"/>
      <circle class="go-ghost" r="0.45" opacity="0"/>
      <g class="go-hits"></g>`,this.stonesG=this.svg.querySelector(".go-stones"),this.fxG=this.svg.querySelector(".go-fx"),this.markerEl=this.svg.querySelector(".go-marker"),this.ghostEl=this.svg.querySelector(".go-ghost");const i=this.svg.querySelector(".go-hits");i.innerHTML=a.join("");const r=c=>{const l=c.target.getAttribute?.("data-p");return l==null?null:Number(l)};i.addEventListener("click",c=>{const l=r(c);l!==null&&this.tryPlay(l)}),i.addEventListener("pointerover",c=>this.showGhost(r(c))),i.addEventListener("pointerout",()=>this.showGhost(null))}tryPlay(t){if(!this.interactive||!this.legalPoints.has(t))return;const e=this.labelIndex.get(le(t,this.size));e!==void 0&&(this.setInteractive(!1),this.ctx.submit(String(e)))}showGhost(t){if(t===null||!this.interactive||!this.legalPoints.has(t)||this.view?.cells[t]!=="."){this.ghostEl.setAttribute("opacity","0");return}const{x:e,y:s}=this.xy(t);this.ghostEl.setAttribute("cx",String(e)),this.ghostEl.setAttribute("cy",String(s)),this.ghostEl.setAttribute("fill",this.ctx.humanSeat===1?"rgba(250,248,238,.62)":"rgba(12,12,16,.55)"),this.ghostEl.setAttribute("opacity","1")}setInteractive(t){this.interactive=t,this.passBtn.disabled=!t||!this.labelIndex.has("pass"),t||this.ghostEl.setAttribute("opacity","0"),this.svg.querySelectorAll(".go-hit").forEach(e=>e.classList.toggle("go-hit-on",t&&this.legalPoints.has(Number(e.getAttribute("data-p")))))}drawStones(t){this.stoneEls.clear(),this.stonesG.replaceChildren();for(let e=0;e<t.cells.length;e++){const s=t.cells[e];s!=="b"&&s!=="w"||this.stonesG.append(this.makeStone(e,s==="b"?0:1))}if(this.lastMove!==null&&t.cells[this.lastMove]!=="."){const{x:e,y:s}=this.xy(this.lastMove);this.markerEl.setAttribute("cx",String(e)),this.markerEl.setAttribute("cy",String(s)),this.markerEl.setAttribute("stroke",t.cells[this.lastMove]==="b"?"#f2f0e4":"#1c1c20"),this.markerEl.setAttribute("opacity","1")}else this.markerEl.setAttribute("opacity","0")}makeStone(t,e){const{x:s,y:o}=this.xy(t),a=document.createElementNS("http://www.w3.org/2000/svg","circle");return a.setAttribute("cx",String(s)),a.setAttribute("cy",String(o)),a.setAttribute("r","0.47"),a.setAttribute("fill",e===0?"url(#go-stone-b)":"url(#go-stone-w)"),this.stoneEls.set(t,a),a}render(t){const e=et(t.viewData);if(!e)return;e.size!==this.size&&this.buildBoard(e.size),this.view=e,this.drawStones(e),this.capEls[0].textContent=String(e.captures[0]),this.capEls[1].textContent=String(e.captures[1]);const s=this.turnChip.querySelector(".go-turn-dot"),o=this.turnChip.querySelector(".go-turn-text");t.isOver?(o.textContent="Game over",s.style.background="var(--text-dim)",this.plaques.forEach(a=>a.classList.remove("go-active"))):(o.textContent=e.turn===0?"Black to move":"White to move",s.style.background=e.turn===0?"radial-gradient(circle at 35% 30%, #7c8088, #0a0a0d)":"radial-gradient(circle at 35% 30%, #ffffff, #c4c0ae)",this.plaques.forEach((a,i)=>a.classList.toggle("go-active",i===e.turn))),t.toAct!==t.humanSeat&&this.setInteractive(!1)}async animate(t,e){const s=t.data??null,o=this.ctx.animationScale(),a=et(e.viewData);if(a&&a.size!==this.size&&this.buildBoard(a.size),s&&typeof s.point=="number"){if(this.lastMove=s.point,this.render(e),o>0){const i=this.stoneEls.get(s.point);i&&(i.style.animationDuration=`${280*o}ms`,i.classList.add("go-drop"));const r=s.captured??[];for(const c of r){const l=this.makeStone(c,s.seat^1);this.stoneEls.delete(c),l.style.animationDuration=`${340*o}ms`,l.style.animationDelay=`${120*o}ms`,l.classList.add("go-die"),this.fxG.append(l)}await b((r.length>0?500:300)*o),this.fxG.replaceChildren()}}else s&&s.move==="pass"?(this.lastMove=null,this.render(e),o>0&&(this.toastEl.textContent=`${s.seat===0?"Black":"White"} passes`,this.toastEl.classList.add("go-toast-show"),await b(650*o),this.toastEl.classList.remove("go-toast-show"))):(this.render(e),await b(200*o))}promptAction(t){this.labelIndex=new Map(t.map((e,s)=>[e,s])),this.legalPoints=new Set(t.map(e=>ce(e,this.size)).filter(e=>e!==null)),this.setInteractive(!0)}unmount(){}}function ge(){return new fe}const ot="liars-dice-frontend-style",be=`
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

@media (prefers-reduced-motion: reduce) {
  .ld-turn .ld-pod,
  .ld-roll .ld-cup,
  .ld-flip,
  .ld-lose .ld-pod {
    animation: none;
  }
}
`;function it(n){return typeof n=="object"&&n!==null&&Array.isArray(n.hands)}function me(n){if(typeof n!="object"||n===null)return!1;const t=n.kind;return t==="liar"||t==="exact"}const xe={1:["c"],2:["ne","sw"],3:["ne","c","sw"],4:["nw","ne","sw","se"],5:["nw","ne","c","sw","se"],6:["nw","ne","w","e","sw","se"],7:["nw","ne","w","c","e","sw","se"],8:["nw","n","ne","w","e","sw","s","se"],9:["nw","n","ne","w","c","e","sw","s","se"]};function q(n,t=""){const e=xe[n],s=e?e.map(o=>`<i class="ld-pip-${o}"></i>`).join(""):`<b class="ld-die-num">${n}</b>`;return`<span class="ld-die${t}" data-v="${n}">${s}</span>`}function ve(n){return`<span class="ld-cup"><span class="ld-cup-count">${n}</span></span>`}function ye(n,t){const e=Math.PI/180*(90+360*n/t);return{x:50+39*Math.cos(e),y:50+36*Math.sin(e)}}class we{ctx;tableEl;seatsEl;centerEl;bannerEl;controlsEl;view=null;ladder=[];ladderRound=-1;dead=!1;openQty=1;openFace=1;mount(t,e){if(this.ctx=e,!document.getElementById(ot)){const s=document.createElement("style");s.id=ot,s.textContent=be,document.head.append(s)}t.innerHTML=`
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
      </div>`,this.tableEl=t.querySelector(".ld-table"),this.seatsEl=t.querySelector(".ld-seats"),this.centerEl=t.querySelector(".ld-center"),this.bannerEl=t.querySelector(".ld-banner"),this.controlsEl=t.querySelector(".ld-controls")}render(t){if(!it(t.viewData)){const s=document.createElement("pre");s.className="ld-fallback",s.textContent=t.view,this.seatsEl.replaceChildren(s);return}const e=t.viewData;this.view=e,this.syncLadder(e),this.renderSeats(e),this.renderCenter(e),(t.toAct!==t.humanSeat||t.isOver)&&this.controlsEl.replaceChildren()}async animate(t,e){const s=this.ctx.animationScale();if(me(t.data)){s>0&&!this.dead&&await this.playReveal(t.data,s),this.render(e);return}s>0&&!this.dead&&await this.animateBid(t.seat,e,s),this.render(e),s>0&&!this.dead&&(this.centerEl.querySelector(".ld-bid-main")?.animate([{transform:"scale(0.8)"},{transform:"scale(1.12)"},{transform:"scale(1)"}],{duration:300*s,easing:"ease-out"}),await b(150*s))}promptAction(t){this.ctx.humanSeat<0||(t.some(e=>e.startsWith("open "))?this.renderOpenControls(t):this.renderResponseControls(t))}unmount(){this.dead=!0}name(t){return t===this.ctx.humanSeat?"You":`Player ${t}`}syncLadder(t){if(t.round!==this.ladderRound)this.ladderRound=t.round,this.ladder=[...t.history];else if(t.history.length>this.ladder.length)this.ladder=[...t.history];else if(t.bid&&!t.bid.forced){const e=this.ladder[this.ladder.length-1];(!e||e.qty!==t.bid.qty||e.face!==t.bid.face)&&this.ladder.push({seat:t.bid.by,qty:t.bid.qty,face:t.bid.face})}}handHtml(t){return t.alive?t.dice===null||t.dice.length===0?ve(t.count):t.dice.map(e=>q(e)).join(""):'<span class="ld-out-mark">×</span>'}renderSeats(t){const e=t.players,s=this.ctx.humanSeat>=0?this.ctx.humanSeat:0,o=[];for(const a of t.hands){const i=ye((a.seat-s+e)%e,e),r=["ld-seat"];a.alive||r.push("ld-out"),a.alive&&t.phase==="bidding"&&t.turn===a.seat&&r.push("ld-turn"),a.alive&&t.phase==="rolling"&&r.push("ld-roll");const c=t.phase==="over"&&t.winner===a.seat;c&&r.push("ld-winner");const l=t.bid&&!t.bid.forced&&t.phase==="bidding"&&t.bid.by===a.seat?`<span class="ld-bubble">${t.bid.qty}×${q(t.bid.face)}</span>`:"",h=c?'<span class="ld-crown">★</span>':"",d=a.alive?`<span class="ld-tag">${a.count} ${a.count===1?"die":"dice"}</span>`:'<span class="ld-out-tag">OUT</span>';o.push(`
        <div class="${r.join(" ")}" data-seat="${a.seat}"
             style="left:${i.x.toFixed(2)}%;top:${i.y.toFixed(2)}%">
          <div class="ld-pod">
            ${l}
            <div class="ld-hand">${this.handHtml(a)}</div>
            <div class="ld-name">${h}${this.name(a.seat)} ${d}</div>
          </div>
        </div>`)}this.seatsEl.innerHTML=o.join("")}renderCenter(t){const e=this.centerEl.querySelector(".ld-round"),s=this.centerEl.querySelector(".ld-bid-box"),o=this.centerEl.querySelector(".ld-ladder");if(e.textContent=`round ${t.round} · ${t.totalDice} dice in play`,t.phase==="over"&&t.winner!==null){const i=t.winner===this.ctx.humanSeat?"win":"wins";s.innerHTML=`<span class="ld-win-text">★ ${this.name(t.winner)} ${i}</span>`}else if(t.phase==="rolling")s.innerHTML='<span class="ld-open-hint">shaking the cups…</span>';else if(t.bid){const i=t.bid.by===this.ctx.humanSeat?"bid":"bids",r=t.bid.forced?"forced opening bid":`${this.name(t.bid.by)} ${i}`;s.innerHTML=`
        <div class="ld-bid-main">${t.bid.qty}<span class="ld-x">×</span>${q(t.bid.face)}</div>
        <div class="ld-bid-sub">${r}</div>`}else{const i=t.turn===this.ctx.humanSeat?"open":"opens";s.innerHTML=`<span class="ld-open-hint">${this.name(t.turn)} ${i} the round…</span>`}const a=this.ladder.slice(-6);o.innerHTML=a.map((i,r)=>{const c=i.seat===this.ctx.humanSeat?"you":`P${i.seat}`;return`<div class="ld-rung${r===a.length-1?" ld-rung-now":""}"><span>${c}</span> ${i.qty}×${q(i.face)}</div>`}).join("")}submit(t){for(const e of this.controlsEl.querySelectorAll("button"))e.disabled=!0;this.ctx.submit(String(t))}renderResponseControls(t){const e=this.view?.bid,s=this.view?.faces??6,o=t.map((a,i)=>{const r=document.createElement("button");if(r.type="button",r.className="ld-btn",a==="raise quantity"&&e)r.innerHTML=`Raise to ${e.qty+1}×${q(e.face)}`;else if(a==="raise face"&&e){const[c,l]=e.face<s?[e.qty,e.face+1]:[e.qty+1,1];r.innerHTML=`Raise to ${c}×${q(l)}`}else a==="call LIAR"?(r.classList.add("ld-btn-liar"),r.textContent="LIAR!"):a==="call EXACT"?(r.classList.add("ld-btn-exact"),r.textContent="EXACT"):r.textContent=a;return r.onclick=()=>this.submit(i),r});this.controlsEl.replaceChildren(...o)}renderOpenControls(t){const e=new Map;let s=1;for(const[u,w]of t.entries()){const v=/^open (\d+)x(\d+)$/.exec(w);v&&(e.set(`${v[1]}x${v[2]}`,u),s=Math.max(s,Number(v[1])))}const o=this.view?.faces??6,a=this.view?.hands.find(u=>u.seat===this.ctx.humanSeat)?.dice??[],i=new Array(o+1).fill(0);for(const u of a)i[u]++;let r=1;for(let u=1;u<=o;u++)i[u]>=i[r]&&(r=u);this.openFace=r,this.openQty=Math.min(s,Math.max(1,i[r]));const c=document.createElement("div");c.className="ld-open",c.innerHTML=`
      <span class="ld-open-label">open the round</span>
      <div class="ld-qty">
        <button type="button" class="ld-step ld-minus">−</button>
        <span class="ld-qty-n"></span>
        <button type="button" class="ld-step ld-plus">+</button>
      </div>
      <div class="ld-faces"></div>
      <button type="button" class="ld-btn ld-go"></button>`;const l=c.querySelector(".ld-qty-n"),h=c.querySelector(".ld-faces"),d=c.querySelector(".ld-go"),p=c.querySelector(".ld-minus"),g=c.querySelector(".ld-plus"),m=[],y=()=>{l.textContent=String(this.openQty),p.disabled=this.openQty<=1,g.disabled=this.openQty>=s,m.forEach((u,w)=>u.classList.toggle("ld-sel",w+1===this.openFace)),d.innerHTML=`Bid ${this.openQty}×${q(this.openFace)}`,d.disabled=!e.has(`${this.openQty}x${this.openFace}`)};for(let u=1;u<=o;u++){const w=document.createElement("button");w.type="button",w.className="ld-face-btn",w.innerHTML=q(u),w.onclick=()=>{this.openFace=u,y()},m.push(w),h.append(w)}p.onclick=()=>{this.openQty=Math.max(1,this.openQty-1),y()},g.onclick=()=>{this.openQty=Math.min(s,this.openQty+1),y()},d.onclick=()=>{const u=e.get(`${this.openQty}x${this.openFace}`);u!==void 0&&this.submit(u)},y(),this.controlsEl.replaceChildren(c)}showBanner(t,e){this.bannerEl.textContent=t,this.bannerEl.className=`ld-banner ld-banner-${e} ld-show`}hideBanner(){this.bannerEl.classList.remove("ld-show")}async animateBid(t,e,s){const o=it(e.viewData)?e.viewData.bid:null,a=this.seatsEl.querySelector(`[data-seat="${t}"]`);if(!o||!a)return;const i=document.createElement("div");i.className="ld-fly",i.innerHTML=`${o.qty}×${q(o.face)}`,i.style.left=a.style.left,i.style.top=a.style.top,this.tableEl.append(i);const r=this.tableEl.getBoundingClientRect(),c=(50-parseFloat(a.style.left))/100*r.width,l=(46-parseFloat(a.style.top))/100*r.height;await i.animate([{transform:"translate(-50%, -50%)",opacity:1},{transform:`translate(calc(-50% + ${c}px), calc(-50% + ${l}px))`,opacity:.15}],{duration:480*s,easing:"cubic-bezier(0.3, 0.7, 0.4, 1)",fill:"forwards"}).finished.catch(()=>{}),i.remove()}setTally(t,e,s){const o=this.centerEl.querySelector(".ld-bid-box");o&&(o.innerHTML=`
      <div class="ld-bid-main">
        <span class="ld-tally-n">${t??"?"}</span><span class="ld-x">/</span>${s}<span class="ld-x">×</span>${q(e)}
      </div>
      <div class="ld-bid-sub">counting ${e}s across the table…</div>`)}async playReveal(t,e){const s=p=>p*e,o=t.hands.length,a=t.bid.face,i=this.ctx.humanSeat,r=t.caller===i?"call":"calls",c=t.kind==="liar"?"LIAR":"EXACT";if(this.showBanner(`${this.name(t.caller)} ${r} ${c} on ${t.bid.qty}×${a}!`,t.kind==="liar"?"liar":"exact"),this.setTally(null,a,t.bid.qty),await b(s(900)),this.dead)return;let l=0;for(let p=0;p<o;p++){const g=(t.caller+p)%o,m=t.hands[g];if(!m.length)continue;const y=this.seatsEl.querySelector(`[data-seat="${g}"] .ld-hand`);if(y&&(y.innerHTML=m.map(u=>q(u,u===a?" ld-hit ld-flip":" ld-flip")).join("")),l+=m.filter(u=>u===a).length,this.setTally(l,a,t.bid.qty),await b(s(380)),this.dead)return}if(await b(s(250)),this.dead)return;const h=t.loser===null?"":` ${this.name(t.loser)} ${t.loser===i?"lose":"loses"} a die.`;let d;if(t.kind==="liar"?d=t.actual<t.bid.qty?`A lie — only ${t.actual}!${h}`:`The bid was good — ${t.actual} on the table.${h}`:d=t.loser===null?`EXACT — dead on ${t.actual}! Nobody loses a die.`:`Not exact — ${t.actual}, not ${t.bid.qty}.${h}`,this.showBanner(d,t.loser===null?"good":"liar"),t.loser!==null){const p=this.seatsEl.querySelector(`[data-seat="${t.loser}"]`);p?.classList.add("ld-lose");const g=document.createElement("span");g.className="ld-float",g.textContent="−1 die",p?.querySelector(".ld-pod")?.append(g)}else this.seatsEl.querySelector(`[data-seat="${t.caller}"]`)?.classList.add("ld-safe");if(await b(s(1100)),!this.dead&&!(t.loser!==null&&t.diceLeft[t.loser]===0&&!t.gameOver&&(this.showBanner(`${this.name(t.loser)} ${t.loser===i?"are":"is"} out of the game!`,"liar"),await b(s(900)),this.dead))){if(t.gameOver&&t.winner!==null){this.seatsEl.querySelector(`[data-seat="${t.winner}"]`)?.classList.add("ld-winner");const p=t.adjudicated?" on dice count (round cap reached)":"";this.showBanner(`★ ${this.name(t.winner)} ${t.winner===i?"win":"wins"} the game${p}!`,"good"),await b(s(1200))}this.hideBanner()}}}function ke(){return new we}const E=8,H=["Black","White"];function at(n){if(!n||typeof n!="object")return null;const t=n;return typeof t.cells=="string"&&t.cells.length===E*E&&Array.isArray(t.counts)&&Array.isArray(t.legal)?t:null}function Ee(n){if(!n||typeof n!="object")return null;const t=n;return typeof t.move=="string"&&typeof t.player=="number"&&Array.isArray(t.flipped)?t:null}function nt(n){return/^[a-h][1-8]$/.test(n)?(n.charCodeAt(1)-49)*E+(n.charCodeAt(0)-97):null}function Se(n,t){return Math.max(Math.abs(Math.floor(n/E)-Math.floor(t/E)),Math.abs(n%E-t%E))}function $e(n,t){if(!n||!t)return null;let e=null;const s=[];for(let o=0;o<E*E;o++)n.cells[o]!==t.cells[o]&&(n.cells[o]==="."?e=o:s.push(o));return e===null?{move:"pass",player:n.turn,placed:null,flipped:[]}:{move:"place",player:t.cells[e]==="b"?0:1,placed:e,flipped:s}}const rt="othello-frontend-style",qe=`
.ot-root {
  align-self: center;
  width: min(100%, 520px);
  display: flex;
  flex-direction: column;
  gap: 14px;
  user-select: none;
}
.ot-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
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
`;function Me(){if(document.getElementById(rt))return;const n=document.createElement("style");n.id=rt,n.textContent=qe,document.head.append(n)}class Le{ctx;rootEl;boardEl;msgEl;toastEl;passEl;fallbackEl;scoreEls=[];countEls=[];cells=[];discs=new Map;view=null;actionBySq=null;anims=new Set;mount(t,e){this.ctx=e,Me();const s=o=>`
      <div class="ot-score ot-score-${o}">
        <span class="ot-mini ot-mini-${o===0?"b":"w"}"></span>
        <span>${H[o]} · ${e.humanSeat===o?"You":"Bot"}</span>
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
      </div>`,this.rootEl=t.querySelector(".ot-root"),this.boardEl=t.querySelector(".ot-board"),this.msgEl=t.querySelector(".ot-msg"),this.toastEl=t.querySelector(".ot-toast"),this.passEl=t.querySelector(".ot-pass"),this.fallbackEl=t.querySelector(".ot-fallback"),this.scoreEls=[t.querySelector(".ot-score-0"),t.querySelector(".ot-score-1")],this.countEls=this.scoreEls.map(o=>o.querySelector(".ot-count")),this.cells=[...this.boardEl.querySelectorAll(".ot-cell")];for(const o of[25,75])for(const a of[25,75]){const i=document.createElement("div");i.className="ot-star",i.style.left=`${o}%`,i.style.top=`${a}%`,this.boardEl.append(i)}this.boardEl.addEventListener("click",o=>{const a=o.target.closest(".ot-cell");a&&this.clickSquare(this.cells.indexOf(a))})}render(t){this.disableInput();const e=at(t.viewData);if(this.view=e,!e){this.rootEl.classList.add("ot-text-only"),this.fallbackEl.textContent=t.view;return}this.rootEl.classList.remove("ot-text-only"),this.rebuildDiscs(e);for(let a=0;a<2;a++)this.countEls[a].textContent=String(e.counts[a]),this.scoreEls[a].classList.toggle("ot-active",!t.isOver&&e.turn===a);if(this.ctx.humanSeat<0&&!t.isOver)for(const a of e.legal){const i=nt(a);i!==null&&this.cells[i].classList.add("ot-hint")}const[s,o]=e.counts;this.msgEl.textContent=t.isOver?s===o?`Draw, ${s}–${o}`:`${H[s>o?0:1]} wins ${Math.max(s,o)}–${Math.min(s,o)}`:`${H[e.turn]} to move`}async animate(t,e){const s=this.view,o=at(e.viewData),a=Ee(t.data)??$e(s,o),i=this.ctx.animationScale();if(a?.move==="pass"||a?.placed==null){i>0&&a&&(this.toastEl.textContent=`${H[a.player]} passes`,this.toastEl.classList.add("ot-show"),await b(800*i),this.toastEl.classList.remove("ot-show")),this.render(e);return}if(this.render(e),!o||i<=0)return;const r=[],c=this.discs.get(a.placed);c&&r.push(this.run(c.animate([{transform:"scale(0.2)",opacity:.4,offset:0},{transform:"scale(1.14)",opacity:1,offset:.7},{transform:"scale(1)",offset:1}],{duration:240*i,easing:"ease-out"})));for(const l of a.flipped){const h=this.discs.get(l)?.querySelector(".ot-flip");if(!h)continue;const d=o.cells[l]==="w",[p,g,m]=d?[0,90,180]:[180,270,360];r.push(this.run(h.animate([{transform:`rotateY(${p}deg) scale(1)`},{transform:`rotateY(${g}deg) scale(1.18)`},{transform:`rotateY(${m}deg) scale(1)`}],{duration:340*i,delay:(110+85*(Se(a.placed,l)-1))*i,easing:"ease-in-out",fill:"backwards"})))}await Promise.all(r),await b(70*i)}promptAction(t){const e=t.indexOf("pass");if(e>=0&&t.length===1){this.passEl.classList.add("ot-show"),this.passEl.onclick=()=>{this.disableInput(),this.ctx.submit(String(e))};return}const s=new Map;t.forEach((o,a)=>{const i=nt(o);i!==null&&(s.set(i,a),this.cells[i].classList.add("ot-legal"))}),this.actionBySq=s,this.boardEl.classList.add("ot-live",this.ctx.humanSeat===1?"ot-human-w":"ot-human-b")}unmount(){for(const t of this.anims)t.cancel();this.anims.clear()}rebuildDiscs(t){this.discs.clear();for(let e=0;e<E*E;e++){const s=t.cells[e];if(s==="."){this.cells[e].replaceChildren();continue}const o=document.createElement("div");o.className=`ot-disc ${s==="b"?"ot-b":"ot-w"}`,o.innerHTML='<div class="ot-flip"><div class="ot-face ot-face-b"></div><div class="ot-face ot-face-w"></div></div>',this.cells[e].replaceChildren(o),this.discs.set(e,o)}}clickSquare(t){const e=this.actionBySq?.get(t);e!==void 0&&(this.disableInput(),this.ctx.submit(String(e)))}disableInput(){this.actionBySq=null,this.passEl.classList.remove("ot-show"),this.passEl.onclick=null,this.boardEl.classList.remove("ot-live","ot-human-b","ot-human-w");for(const t of this.cells)t.classList.remove("ot-legal","ot-hint")}async run(t){this.anims.add(t);try{await t.finished}catch{}finally{this.anims.delete(t)}}}function Ce(){return new Le}function lt(n){if(!n||typeof n!="object")return null;const t=n;return typeof t.width!="number"||typeof t.height!="number"||!Array.isArray(t.snake)||t.snake.length===0||t.dir!=="n"&&t.dir!=="e"&&t.dir!=="s"&&t.dir!=="w"?null:t}const Te={ArrowUp:"n",ArrowRight:"e",ArrowDown:"s",ArrowLeft:"w",w:"n",d:"e",s:"s",a:"w",W:"n",D:"e",S:"s",A:"w"},_={n:"w",w:"s",s:"e",e:"n"},Ae={n:"e",e:"s",s:"w",w:"n"},ct={n:[0,-1],e:[1,0],s:[0,1],w:[-1,0]};function ze(n,t){return t===n?"straight":_[n]===t?"left":_[t]===n?"right":null}const Be=[{rel:"left",glyph:"↶"},{rel:"straight",glyph:"↑"},{rel:"right",glyph:"↷"}],dt=120,ht=560,De=180,Pe=90,Re=3,Ie=2,He=`
.snake {
  margin: auto;
  width: min(100%, 520px);
  display: flex;
  flex-direction: column;
  gap: 14px;
  user-select: none;
}
.snake-top {
  display: flex;
  align-items: stretch;
  gap: 10px;
}
.snake-logo {
  margin-right: auto;
  align-self: center;
  font-size: 1.5rem;
  font-weight: 850;
  letter-spacing: 0.18em;
  background: linear-gradient(135deg, var(--good), var(--accent));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.snake-stat {
  min-width: 76px;
  padding: 6px 14px;
  text-align: center;
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) - 2px);
}
.snake-stat small {
  display: block;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.62rem;
  color: var(--text-dim);
}
.snake-stat b {
  font-size: 1.15rem;
  font-variant-numeric: tabular-nums;
}
.snake-frame {
  display: flex;
  justify-content: center;
  padding: 10px;
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
.snake-canvas {
  display: block;
  border-radius: calc(var(--radius) - 4px);
}
.snake-pad {
  display: flex;
  justify-content: center;
  gap: 8px;
}
.snake-pad.snake-hidden { display: none; }
.snake-btn {
  min-width: 96px;
  padding: 9px 0;
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) - 2px);
  color: var(--text);
  font-size: 0.92rem;
  transition: border-color 0.12s, color 0.12s;
}
.snake-btn:not(:disabled):hover { border-color: var(--good); color: var(--good); }
.snake-btn:disabled { opacity: 0.32; cursor: default; }
@media (max-width: 480px) {
  .snake-btn { min-width: 0; flex: 1; }
}
`;function je(){if(document.getElementById("snake-frontend-style"))return;const n=document.createElement("style");n.id="snake-frontend-style",n.textContent=He,document.head.append(n)}class Ne{ctx;view=null;tween=null;flash=null;overState=null;pending=null;live=!1;armed=!1;queue=[];tickTimer=0;nextTickAt=0;canvas;c2d;frameEl;scoreEl;lenEl;padBtns=new Map;cssW=0;cssH=0;rafId=0;resizeObs=null;colors={bg:"#010409",snake:"#3fb950",headGlow:"#8aff9f",food:"#f85149",win:"#edc22e"};mount(t,e){this.ctx=e,this.live=e.humanSeat>=0,je(),t.innerHTML=`
      <div class="snake">
        <div class="snake-top">
          <span class="snake-logo">SNAKE</span>
          <div class="snake-stat"><small>score</small><b class="snake-score">0</b></div>
          <div class="snake-stat"><small>length</small><b class="snake-len">0</b></div>
        </div>
        <div class="snake-frame"><canvas class="snake-canvas"></canvas></div>
        <div class="snake-pad"></div>
      </div>`,this.frameEl=t.querySelector(".snake-frame"),this.canvas=t.querySelector(".snake-canvas"),this.c2d=this.canvas.getContext("2d"),this.scoreEl=t.querySelector(".snake-score"),this.lenEl=t.querySelector(".snake-len");const s=t.querySelector(".snake-pad");if(e.humanSeat<0)s.classList.add("snake-hidden");else for(const{rel:o,glyph:a}of Be){const i=document.createElement("button");i.type="button",i.className="snake-btn",i.textContent=`${a} ${o}`,i.disabled=!0,i.onclick=()=>this.onInput(o),s.append(i),this.padBtns.set(o,i)}this.resizeObs=new ResizeObserver(()=>this.layout()),this.resizeObs.observe(this.frameEl),window.addEventListener("keydown",this.onKey),document.addEventListener("visibilitychange",this.onVisibility),this.rafId=requestAnimationFrame(this.loop)}render(t){const e=lt(t.viewData);e&&(this.view=e,this.tween=null,this.layout(),this.updateStats(e),t.isOver&&(this.overState=e.status==="won"?"win":"dead",clearTimeout(this.tickTimer)),t.toAct!==t.humanSeat&&this.setPending(null))}async animate(t,e){const s=lt(e.viewData);if(!s)return;const o=this.view;this.view=s,this.layout(),this.updateStats(s);const a=this.ctx.animationScale(),i=this.live?Math.min(dt*a,this.tickMs()*.8):dt*a;i>0&&o&&(this.tween={from:o,start:performance.now(),dur:i},await b(i)),e.isOver&&(this.overState=s.status==="won"?"win":"dead",a>0&&(this.flash={start:performance.now(),dur:ht*a},await b(ht*a)))}promptAction(t){this.setPending(t),this.live&&this.armed&&this.scheduleTick()}unmount(){cancelAnimationFrame(this.rafId),clearTimeout(this.tickTimer),window.removeEventListener("keydown",this.onKey),document.removeEventListener("visibilitychange",this.onVisibility),this.resizeObs?.disconnect(),this.resizeObs=null}onKey=t=>{if(this.ctx.humanSeat<0||t.metaKey||t.ctrlKey||t.altKey)return;const e=t.target;if(e&&(e.tagName==="INPUT"||e.tagName==="TEXTAREA"||e.isContentEditable))return;const s=Te[t.key];if(!s||(t.preventDefault(),!this.view))return;const o=ze(this.predictedDir(),s);o&&this.onInput(o)};predictedDir(){let t=this.view.dir;for(const e of this.queue)t=e==="left"?_[t]:e==="right"?Ae[t]:t;return t}onInput(t){if(!(!this.live||this.overState)){if(!this.armed){if(!this.pending)return;this.armed=!0,this.nextTickAt=performance.now()+this.tickMs(),this.submitRel(t);return}t!=="straight"&&this.queue.length<Ie&&this.queue.push(t)}}scheduleTick(){clearTimeout(this.tickTimer);const t=Math.max(0,this.nextTickAt-performance.now());this.tickTimer=window.setTimeout(()=>this.tick(),t)}tick(){!this.pending||document.hidden||(this.nextTickAt=performance.now()+this.tickMs(),this.submitRel(this.queue.shift()??"straight"))}tickMs(){const t=Math.max(0,(this.view?.score??3)-3);return Math.max(Pe,De-Re*t)}onVisibility=()=>{document.hidden||!this.live||!this.armed||!this.pending||(this.nextTickAt=performance.now()+this.tickMs(),this.scheduleTick())};submitRel(t){if(!this.pending)return;const e=this.pending.indexOf(t);e<0||(this.setPending(null),this.ctx.submit(String(e)))}setPending(t){this.pending=t;const e=this.armed&&!this.overState;for(const[s,o]of this.padBtns)o.disabled=e?!1:!t||!t.includes(s)}updateStats(t){this.scoreEl.textContent=String(Math.max(0,t.score-3)),this.lenEl.textContent=`${t.score}/${t.width*t.height}`}layout(){const t=this.view;if(!t)return;const e=Math.max(120,this.frameEl.clientWidth-22),s=Math.min(e/t.width,440/t.height),o=Math.round(s*t.width),a=Math.round(s*t.height);if(o===this.cssW&&a===this.cssH)return;this.cssW=o,this.cssH=a;const i=window.devicePixelRatio||1;this.canvas.style.width=`${o}px`,this.canvas.style.height=`${a}px`,this.canvas.width=Math.round(o*i),this.canvas.height=Math.round(a*i)}loop=()=>{this.drawFrame(performance.now()),this.rafId=requestAnimationFrame(this.loop)};drawFrame(t){const e=this.view;if(!e||this.canvas.width===0)return;const s=this.c2d,o=this.canvas.width,a=this.canvas.height,i=o/e.width,r=this.ctx.animationScale();let c=1,l=e.snake;const h=this.tween;if(h){c=Math.min(1,(t-h.start)/Math.max(1,h.dur)),c>=1&&(this.tween=null);const f=h.from.snake;l=e.snake.map(([S,x],C)=>{const[z,A]=f[Math.min(C,f.length-1)];return[z+(S-z)*c,A+(x-A)*c]})}if(e.status==="crashed"&&l.length>0){const f=(h?Math.sin(Math.min(1,c)*Math.PI):0)*.4,[S,x]=ct[e.dir];l=l.slice(),l[0]=[l[0][0]+S*f,l[0][1]+x*f]}s.clearRect(0,0,o,a),s.fillStyle=this.colors.bg,s.fillRect(0,0,o,a),s.strokeStyle="rgba(255, 255, 255, 0.05)",s.lineWidth=1,s.beginPath();for(let f=1;f<e.width;f++)s.moveTo(f*i,0),s.lineTo(f*i,a);for(let f=1;f<e.height;f++)s.moveTo(0,f*i),s.lineTo(o,f*i);if(s.stroke(),e.food){const f=h&&!h.from.food?c:1,S=r>0?1+.08*Math.sin(t/260):1,x=i*.3*S*f,[C,z]=e.food,A=(C+.5)*i,B=(z+.5)*i,P=s.createRadialGradient(A,B,0,A,B,x*3);P.addColorStop(0,"rgba(248, 81, 73, 0.35)"),P.addColorStop(1,"rgba(248, 81, 73, 0)"),s.fillStyle=P,s.fillRect(A-x*3,B-x*3,x*6,x*6),s.save(),s.shadowColor=this.colors.food,s.shadowBlur=i*.5,s.fillStyle=this.colors.food,s.beginPath(),s.arc(A,B,x,0,Math.PI*2),s.fill(),s.restore()}const d=this.overState==="dead",p=this.overState==="win",g=d?this.colors.food:p?this.colors.win:this.colors.snake;s.save(),d&&!this.flash&&(s.globalAlpha=.65),s.shadowColor=g,s.shadowBlur=i*.65,s.strokeStyle=g,s.lineWidth=i*.64,s.lineJoin="round",s.lineCap="round",s.beginPath(),s.moveTo((l[0][0]+.5)*i,(l[0][1]+.5)*i);for(let f=1;f<l.length;f++)s.lineTo((l[f][0]+.5)*i,(l[f][1]+.5)*i);s.stroke();const m=(l[0][0]+.5)*i,y=(l[0][1]+.5)*i;s.fillStyle=d?this.colors.food:this.colors.headGlow,s.beginPath(),s.arc(m,y,i*.36,0,Math.PI*2),s.fill(),s.shadowBlur=0;const[u,w]=ct[e.dir];s.fillStyle="#04140a";for(const f of[-1,1])s.beginPath(),s.arc(m+u*i*.14-w*f*i*.16,y+w*i*.14+u*f*i*.16,i*.07,0,Math.PI*2),s.fill();s.restore(),this.live&&!this.armed&&!this.overState&&(s.save(),s.fillStyle="rgba(1, 4, 9, 0.55)",s.fillRect(0,0,o,a),s.textAlign="center",s.textBaseline="middle",s.fillStyle="rgba(230, 237, 243, 0.92)",s.font=`600 ${Math.round(i*.5)}px system-ui, sans-serif`,s.fillText("press an arrow to start",o/2,a/2-i*1.6),s.fillStyle="rgba(230, 237, 243, 0.55)",s.font=`${Math.round(i*.36)}px system-ui, sans-serif`,s.fillText("it won't wait for you",o/2,a/2-i*.85),s.restore());const v=this.flash;if(v){const f=(t-v.start)/Math.max(1,v.dur);if(f>=1)this.flash=null;else{const S=.38*Math.abs(Math.sin(f*Math.PI*3));s.fillStyle=p?`rgba(63, 185, 80, ${S})`:`rgba(248, 81, 73, ${S})`,s.fillRect(0,0,o,a)}}}}function Fe(){return new Ne}const pt="twentyone-frontend-style",Oe=`
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
`;function Ge(){if(document.getElementById(pt))return;const n=document.createElement("style");n.id=pt,n.textContent=Oe,document.head.append(n)}const ut=`
  <div class="t21-seat-bar">
    <span class="t21-name"></span>
    <span class="t21-hearts"></span>
    <span class="t21-badge" hidden>stood</span>
  </div>
  <div class="t21-cards"></div>
  <div class="t21-total"></div>`;function ft(n){return n.reduce((t,e)=>t+e,0)}class Ye{ctx;seatEls=[];roundEl;stakeEl;deckEl;deckPile;deckCountEl;bannerEl;bannerChip;actionsEl;prevCounts=[0,0];lastRound=0;prevHearts=null;roundEndSeen=!1;mount(t,e){this.ctx=e,Ge(),t.innerHTML=`
      <div class="t21">
        <div class="t21-table">
          <section class="t21-seat" data-pos="top">${ut}</section>
          <div class="t21-mid">
            <div class="t21-round"><b></b><span class="t21-stake"></span></div>
            <div class="t21-deck">
              <span class="t21-deck-count"></span>
              <div class="t21-deck-pile"><i></i><i></i><i></i></div>
            </div>
          </div>
          <section class="t21-seat" data-pos="bottom">${ut}</section>
          <div class="t21-banner" hidden><div class="t21-banner-chip"></div></div>
        </div>
        <div class="t21-actions"></div>
      </div>`;const s=a=>{const i=t.querySelector(`[data-pos="${a}"]`);return{bar:i.querySelector(".t21-seat-bar"),name:i.querySelector(".t21-name"),hearts:i.querySelector(".t21-hearts"),badge:i.querySelector(".t21-badge"),cards:i.querySelector(".t21-cards"),total:i.querySelector(".t21-total")}},o=e.humanSeat>=0?e.humanSeat:0;this.seatEls=[],this.seatEls[o]=s("bottom"),this.seatEls[1-o]=s("top"),this.roundEl=t.querySelector(".t21-round b"),this.stakeEl=t.querySelector(".t21-stake"),this.deckEl=t.querySelector(".t21-deck"),this.deckPile=t.querySelector(".t21-deck-pile"),this.deckCountEl=t.querySelector(".t21-deck-count"),this.bannerEl=t.querySelector(".t21-banner"),this.bannerChip=t.querySelector(".t21-banner-chip"),this.actionsEl=t.querySelector(".t21-actions"),e.humanSeat<0&&(this.actionsEl.style.display="none")}seatName(t){return t===this.ctx.humanSeat?"You":this.ctx.humanSeat>=0?"Bot":`Player ${t}`}cardEl(t,e){const s=document.createElement("div");return t===null?(s.className="t21-card t21-back t21-hole",s.textContent="?"):(s.className=e?"t21-card t21-hole":"t21-card",s.innerHTML=`<span class="t21-pip">${t}</span><b>${t}</b><span class="t21-pip t21-pip-br">${t}</span>`),s}setTotal(t,e,s){if(s===null){t.innerHTML="&nbsp;",t.classList.remove("t21-bust","t21-sweet");return}t.innerHTML=`${e} <b>${s}</b>`,t.classList.toggle("t21-bust",s>21),t.classList.toggle("t21-sweet",s===21)}renderHearts(t,e,s){const o=[];for(let a=0;a<s;a++){const i=document.createElement("span");i.className=a<e?"t21-heart":"t21-heart t21-lost",i.textContent="♥",o.push(i)}t.replaceChildren(...o)}renderSeat(t,e,s){const o=this.seatEls[t],a=e.players[t],i=this.ctx.animationScale();if(o.name.textContent=this.seatName(t),this.renderHearts(o.hearts,e.hearts[t],e.maxHearts),o.badge.hidden=!(e.roundActive&&a.stood),o.bar.classList.toggle("t21-active",e.roundActive&&!s.isOver&&e.toAct===t),e.roundActive){let r=0;const c=this.prevCounts[t],l=(d,p)=>{const g=this.cardEl(d,p);return r>=c&&i>0&&(g.classList.add("t21-deal"),g.style.animationDuration=`${320*i}ms`,g.style.animationDelay=`${(r-c)*80*i}ms`),r++,g},h=[l(a.up[0]??null,!1),l(a.down,!0)];for(const d of a.up.slice(1))h.push(l(d,!1));o.cards.replaceChildren(...h),a.total!==null?this.setTotal(o.total,"total",a.total):this.setTotal(o.total,"showing",ft(a.up)),this.prevCounts[t]=a.up.length+1}else if(e.lastReveal){const r=e.lastReveal.up[t],c=e.lastReveal.downs[t],l=[this.cardEl(r[0]??null,!1),this.cardEl(c,!0)];for(const h of r.slice(1))l.push(this.cardEl(h,!1));for(const h of l)h.classList.add("t21-spent");o.cards.replaceChildren(...l),this.setTotal(o.total,"total",ft(r)+c),this.prevCounts[t]=0}else{const r=document.createElement("div");r.className="t21-placeholder",r.textContent="waiting for the deal…",o.cards.replaceChildren(r),this.setTotal(o.total,"",null),this.prevCounts[t]=0}}showBanner(t,e){this.bannerChip.textContent=t,this.bannerEl.className=`t21-banner${e?` t21-banner-${e}`:""}`,this.bannerEl.hidden=!1}endText(t,e,s){return t===null?`Round ${s}: push — no damage`:`${t===this.ctx.humanSeat?"You win":`${this.seatName(t)} wins`} round ${s} · −${e} ♥`}endClass(t){return this.ctx.humanSeat<0||t===null?"":t===this.ctx.humanSeat?"good":"bad"}render(t){const e=t.viewData;if(!e)return;const s=e.round!==this.lastRound;if(s&&(this.prevCounts=[0,0]),t.isOver){const o=this.ctx.humanSeat>=0?e.hearts[this.ctx.humanSeat]>0?"good":"bad":"";this.showBanner(t.result??"Game over",o)}else if(e.roundActive)this.bannerEl.hidden=!0,this.roundEndSeen=!1;else if(s&&this.lastRound>0&&!this.roundEndSeen&&this.prevHearts){const o=e.hearts[0]<this.prevHearts[0]?0:e.hearts[1]<this.prevHearts[1]?1:null,a=o===null?null:1-o,i=o===null?0:this.prevHearts[o]-e.hearts[o];this.showBanner(this.endText(a,i,this.lastRound),this.endClass(a)),this.roundEndSeen=!0}this.roundEl.textContent=`Round ${e.round}`,this.stakeEl.textContent=`${e.round} ♥ at stake`,this.deckCountEl.textContent=e.roundActive?`${e.deckCount} in deck`:"",this.deckEl.style.visibility=e.roundActive?"visible":"hidden",this.renderSeat(0,e,t),this.renderSeat(1,e,t),t.toAct!==t.humanSeat&&this.actionsEl.replaceChildren(),this.lastRound=e.round,this.prevHearts=[e.hearts[0],e.hearts[1]]}async showdown(t,e){for(const s of[0,1]){const o=this.seatEls[s],a=o.cards.querySelector(".t21-back");if(a){const i=this.cardEl(t.downs[s],!0);i.classList.add("t21-flip"),i.style.animationDuration=`${500*e}ms`,a.replaceWith(i)}this.setTotal(o.total,"total",t.totals[s])}if(await b(700*e),this.showBanner(this.endText(t.winner,t.damage,this.lastRound),this.endClass(t.winner)),t.winner!==null){const s=1-t.winner,o=[...this.seatEls[s].hearts.querySelectorAll(".t21-heart:not(.t21-lost)")];for(const a of o.slice(Math.max(0,o.length-t.damage)))a.style.animationDuration=`${800*e}ms`,a.classList.add("t21-heart-break")}await b(900*e)}async animate(t,e){const s=this.ctx.animationScale(),o=t.data??null;if(o?.kind==="draw")s>0&&(this.deckPile.classList.add("t21-deck-pulse"),await b(260*s),this.deckPile.classList.remove("t21-deck-pulse")),this.render(e),await b(160*s);else if(o?.kind==="stand"){if(this.render(e),s>0){const a=this.seatEls[o.seat].bar;a.classList.add("t21-stand-flash"),await b(340*s),a.classList.remove("t21-stand-flash")}}else o?.kind==="roundEnd"?(this.roundEndSeen=!0,s>0?await this.showdown(o,s):this.showBanner(this.endText(o.winner,o.damage,this.lastRound),this.endClass(o.winner)),this.render(e),await b(250*s)):(this.render(e),await b(200*s))}promptAction(t){const e={draw:"take a card",stand:"hold your total"},s=t.map((o,a)=>{const i=document.createElement("button");i.type="button",i.className=o==="draw"?"t21-btn t21-btn-draw":"t21-btn";const r=e[o];return i.innerHTML=r?`${o}<span>${r}</span>`:o,i.onclick=()=>{for(const c of s)c.disabled=!0;this.ctx.submit(String(a))},i});this.actionsEl.replaceChildren(...s)}unmount(){}}function _e(){return new Ye}const We={2048:Nt,chess:Jt,connect4:ne,go:ge,"liars-dice":ke,othello:Ce,snake:Fe,twentyone:_e};function Ve(n){const t=We[n];return t?t():new re}const gt={chess:{bots:["alphabeta:depth=4","alphabeta-rich:depth=4"],opts:{}},othello:{bots:["alphabeta:depth=5","mcts:sims=2000"],opts:{}},connect4:{bots:["alphabeta:depth=7","alphabeta:depth=5","mcts:sims=2000"],opts:{}},go:{bots:["mcts:sims=800","mcts-eval:sims=800"],opts:{size:"9"}},"liars-dice":{bots:["rollout:rollouts=300","belief","random"],opts:{players:"2",dice:"5"}}};class Ue{constructor(t,e,s,o){this.root=t,this.compare=e,this.statsHost=s,this.onBack=o}hosts=[];running=!1;gen=0;render(){const t=this.compare.map(o=>`<option value="${o.id}">${o.id}</option>`).join("");this.root.innerHTML=`
      <div class="tourney">
        <button type="button" class="link back">&larr; arcade</button>
        <h2>Tournament lab</h2>
        <p class="muted">Round-robin between bot specs, paired seat-swapped games on a pool of
           engine workers, Bradley-Terry Elo fitted live. Same statistics as the lab's CLI.</p>
        <div class="tourney-form">
          <label class="opt-row"><span>game</span>
            <select class="t-game">${t}</select></label>
          <label class="opt-row"><span>bots</span>
            <textarea class="t-bots" rows="4" spellcheck="false"></textarea></label>
          <div class="bots-help muted"></div>
          <label class="opt-row"><span>games / pairing</span>
            <input class="t-games" value="8" /></label>
          <label class="opt-row"><span>seed</span>
            <input class="t-seed" value="${(Math.floor(Math.random()*2147483647)|1)>>>0}" /></label>
          <button type="button" class="primary t-run">Run tournament</button>
        </div>
        <div class="t-progress"></div>
        <div class="t-standings"></div>
        <div class="t-matrix"></div>
      </div>`,this.root.querySelector(".back").onclick=()=>{this.destroy(),this.onBack()};const e=this.root.querySelector(".t-game"),s=()=>{const o=gt[e.value]??{bots:[]};this.root.querySelector(".t-bots").value=o.bots.join(`
`);const a=this.compare.find(i=>i.id===e.value);this.root.querySelector(".bots-help").textContent=a?`available: ${a.bots}`:""};e.onchange=s,s(),this.root.querySelector(".t-run").onclick=()=>void this.run()}async run(){if(this.running){this.gen++,this.stopPool(),this.running=!1,this.root.querySelector(".t-run").textContent="Run tournament";return}const t=++this.gen,e=this.root.querySelector(".t-game").value,s=this.root.querySelector(".t-bots").value.split(`
`).map(v=>v.trim()).filter(Boolean),o=Math.max(2,Number(this.root.querySelector(".t-games").value)||8),a=Number(this.root.querySelector(".t-seed").value)>>>0||1,i=gt[e]?.opts??{};if(s.length<2){this.progress("Need at least two bot specs (one per line).");return}this.running=!0,this.root.querySelector(".t-run").textContent="Stop";const r=s.length,c=Math.max(1,Math.floor(o/2)),l=Array.from({length:r},()=>Array.from({length:r},()=>({w:0,d:0,l:0}))),h=[];for(let v=0;v<r;v++)for(let f=v+1;f<r;f++)for(let S=0;S<c;S++)h.push({i:v,j:f,k:S});let d=0;const p=h.length;this.renderTables(s,l,null),this.progress(`0 / ${p} pairs`);const g=navigator.hardwareConcurrency||4,m=Math.max(1,Math.min(4,g-2,p));this.hosts=Array.from({length:m},()=>new vt);let y=0,u=Promise.resolve();const w=async v=>{for(;this.gen===t&&y<h.length;){const f=h[y++],S=(a^f.i*r+f.j<<16)>>>0;try{const x=await v.pairs(e,i,s[f.i],s[f.j],S,f.k,f.k+1);if(this.gen!==t)return;const C=l[f.i][f.j];C.w+=x.w,C.d+=x.d,C.l+=x.l;const z=l[f.j][f.i];z.w+=x.l,z.d+=x.d,z.l+=x.w,d++,this.progress(`${d} / ${p} pairs`),u=u.then(async()=>{if(this.gen!==t)return;const A=l.map(P=>P.map(N=>[N.w,N.d,N.l])),B=await this.statsHost.fitElo(A);this.gen===t&&this.renderTables(s,l,B)})}catch(x){if(this.gen!==t)return;this.progress(`error: ${x instanceof Error?x.message:x}`),this.gen++,this.running=!1;const C=this.root.querySelector(".t-run");C&&(C.textContent="Run tournament");return}}};if(await Promise.all(this.hosts.map(w)),await u.catch(()=>{}),this.gen===t){this.progress(`done — ${d*2} games across ${p} pairs on ${m} workers`),this.running=!1;const v=this.root.querySelector(".t-run");v&&(v.textContent="Run tournament")}this.stopPool()}renderTables(t,e,s){const o=t.length,a=t.map((l,h)=>e[h].reduce((d,p)=>({w:d.w+p.w,d:d.d+p.d,l:d.l+p.l}),{w:0,d:0,l:0})),i=t.map((l,h)=>h);s&&i.sort((l,h)=>s[h]-s[l]);const r=i.map((l,h)=>{const d=a[l],p=s?`${s[l]>=0?"+":""}${s[l].toFixed(0)}`:"—";return`<tr><td>${h+1}</td><td class="t-spec">${bt(t[l])}</td>
                <td class="t-elo">${p}</td><td>${d.w}-${d.d}-${d.l}</td></tr>`}).join("");this.root.querySelector(".t-standings").innerHTML=`
      <table class="t-table">
        <thead><tr><th>#</th><th>bot</th><th>elo</th><th>W-D-L</th></tr></thead>
        <tbody>${r}</tbody>
      </table>`;let c='<table class="t-table t-grid"><thead><tr><th></th>';for(let l=0;l<o;l++)c+=`<th>${l+1}</th>`;c+="</tr></thead><tbody>";for(let l=0;l<o;l++){c+=`<tr><th>${l+1}. ${bt(Xe(t[l]))}</th>`;for(let h=0;h<o;h++){const d=e[l][h];c+=l===h?'<td class="t-self">·</td>':`<td>${d.w+d.d+d.l?`${d.w}-${d.d}-${d.l}`:""}</td>`}c+="</tr>"}c+="</tbody></table>",this.root.querySelector(".t-matrix").innerHTML=c}progress(t){const e=this.root.querySelector(".t-progress");e&&(e.textContent=t)}stopPool(){for(const t of this.hosts)t.terminate();this.hosts=[]}destroy(){this.gen++,this.stopPool()}}function Xe(n){return n.length>24?`${n.slice(0,22)}…`:n}function bt(n){return n.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}const Ke={chess:{depth:"4"},"liars-dice":{players:"5",dice:"5",rollouts:"400"},twentyone:{hearts:"3"},othello:{depth:"5"},connect4:{depth:"7"},go:{size:"9",sims:"1500"},2048:{},snake:{}},wt={"data/azero/chess.bin":"artifacts/azero-chess.bin","data/twentyone/solver-h3.bin":"artifacts/t21-solver-h3.bin","data/twentyone/solver-h6.bin":"artifacts/t21-solver-h6.bin"};function Qe(n,t){const e=[];if(n==="chess"){const s=t.net??(t.bot==="azero"?"data/azero/chess.bin":null);s&&e.push(s)}return n==="twentyone"&&e.push(`data/twentyone/solver-h${t.hearts??"6"}.bin`),e.filter(s=>s in wt)}function Je(n,t){return n.filter(e=>e.key!=="seed"&&e.key!=="seat"&&e.key!=="bot"&&!e.nativeOnly).map(e=>({key:e.key,value:t[e.key]??e.value.split("|")[0].replace(/\.{3}$/,""),note:e.note,bots:e.bots}))}function mt(n,t){const e=n.optsSchema.find(s=>s.key==="bot");return e?t.bot??(n.solo?"":e.value.split("|")[0]):""}function j(){return(Math.floor(Math.random()*2147483647)|1)>>>0}function M(n){return n.replace(/[&<>"']/g,t=>`&#${t.charCodeAt(0)};`)}function Ze(n){switch(n){case"chess":return'<div class="mini mini-chess"><span class="mini-pc" style="left:12%;top:8%">♞</span><span class="mini-pc mini-pc-w" style="left:58%;top:52%">♙</span></div>';case"liars-dice":return`<div class="mini mini-dice">
        <span class="mini-die"><i style="left:25%;top:25%"></i><i style="left:65%;top:65%"></i></span>
        <span class="mini-die mini-die-2"><i style="left:45%;top:45%"></i><i style="left:18%;top:18%"></i><i style="left:72%;top:72%"></i></span>
        <span class="mini-cup"></span></div>`;case"twentyone":return'<div class="mini mini-t21"><span class="mini-card">7♠</span><span class="mini-card mini-card-2">9♦</span><span class="mini-heart">♥♥♥</span></div>';case"othello":return'<div class="mini mini-othello"><span class="mini-disc mini-disc-b" style="left:28%;top:28%"></span><span class="mini-disc mini-disc-w" style="left:52%;top:28%"></span><span class="mini-disc mini-disc-w" style="left:28%;top:52%"></span><span class="mini-disc mini-disc-b" style="left:52%;top:52%"></span></div>';case"connect4":return'<div class="mini mini-c4"></div>';case"go":return'<div class="mini mini-go"><span class="mini-stone mini-stone-b" style="left:30%;top:30%"></span><span class="mini-stone mini-stone-w" style="left:55%;top:47%"></span><span class="mini-stone mini-stone-b" style="left:38%;top:63%"></span></div>';case"2048":return'<div class="mini mini-2048"><span>2</span><span class="v4">4</span><span class="v8">8</span><span class="v16">16</span></div>';case"snake":return'<div class="mini mini-snake"><span class="mini-seg" style="left:18%;top:55%"></span><span class="mini-seg" style="left:33%;top:55%"></span><span class="mini-seg" style="left:48%;top:55%"></span><span class="mini-seg mini-head" style="left:48%;top:38%"></span><span class="mini-food" style="left:72%;top:25%"></span></div>';default:return'<div class="mini"></div>'}}class ts{constructor(t){this.root=t}host=new vt;manifest;frontend=null;clientBot=null;tourney=null;gen=0;speedScale=1;submitResolve=null;logEl=null;statusEl=null;async start(){this.root.innerHTML='<div class="boot">Waking the engine…</div>',this.manifest=await this.host.manifest(),this.renderHome()}renderHome(){this.teardown();const t=this.manifest.games.map(e=>`
        <div class="card" data-game="${e.id}" role="button" tabindex="0">
          ${Ze(e.id)}
          <div class="card-text">
            <span class="card-name">${M(e.name||e.id)}</span>
          </div>
          <button type="button" class="card-watch" title="Watch bots play">watch</button>
        </div>`).join("");this.root.innerHTML=`
      <div class="home">
        <header class="home-head">
          <h1><a href="https://henrilemoine.com/">Games Room</a></h1>
        </header>
        <div class="card-grid">${t}</div>
        <div class="home-foot">
          <button type="button" class="link tourney-link">Bot tournament lab &rarr;</button>
        </div>
        <footer class="home-footer">
          <nav>
            <a href="https://github.com/henri123lemoine/games">GitHub</a>
            <a href="https://henrilemoine.com/">henrilemoine.com</a>
          </nav>
          <span class="muted">Rust compiled to WebAssembly — everything runs on your device.</span>
        </footer>
      </div>`;for(const e of this.root.querySelectorAll(".card")){const s=this.manifest.games.find(a=>a.id===e.dataset.game);if(!s)continue;const o=()=>void this.startMatch(s,"play");e.onclick=o,e.onkeydown=a=>{(a.key==="Enter"||a.key===" ")&&(a.preventDefault(),o())},e.querySelector(".card-watch").onclick=a=>{a.stopPropagation(),this.startMatch(s,"watch")}}this.root.querySelector(".tourney-link").onclick=()=>this.renderTournament()}renderTournament(){this.teardown(),this.tourney=new Ue(this.root,this.manifest.compare,this.host,()=>this.renderHome()),this.tourney.render()}buildOpts(t,e,s){const o={...Ke[t.id],...s};e==="watch"?t.solo?o.bot||=t.watchBot:o.seat="watch":t.solo?delete o.bot:o.seat==="watch"&&(o.seat="0");const a=mt(t,o);for(const i of t.optsSchema)i.bots.length>0&&!i.bots.includes(a)&&delete o[i.key];return o.seed||=String(j()),o}async startMatch(t,e,s={}){const o=++this.gen;this.teardownMatch();const a=this.buildOpts(t,e,s);if(this.renderMatchSkeleton(t,e,a),a.bot==="azero-gpu"&&!("gpu"in navigator)){this.setStatus("WebGPU is unavailable in this browser — pick another bot in settings (azero runs the same net on CPU).","error");return}try{await this.loadArtifacts(t,a);const i=await this.host.create(t.id,a);if(o!==this.gen)return;const r=At(t.id,a.bot);if(this.clientBot=r?await r(this.host,a):null,o!==this.gen)return;const c=this.root.querySelector(".board");this.frontend=Ve(t.id);const l={gameId:t.id,opts:a,humanSeat:i.humanSeat,numSeats:i.numSeats,submit:h=>this.submit(h),animationScale:()=>this.animationScale()};this.frontend.mount(c,l),this.frontend.render(i),this.setStatus(i.humanSeat<0?"Bots playing…":"Thinking…"),this.runLoop(o)}catch(i){o===this.gen&&this.setStatus(`Could not start: ${xt(i)}`,"error")}}renderMatchSkeleton(t,e,s){const o=e==="watch"?"take a seat":"watch bots";this.root.innerHTML=`
      <div class="match">
        <header class="match-bar">
          <button type="button" class="link back">&larr; games</button>
          <span class="match-title">${M(t.name||t.id)}</span>
          <span class="spacer"></span>
          <label class="speed-label">speed
            <select class="speed">
              <option value="2">slow</option>
              <option value="1" selected>normal</option>
              <option value="0.4">fast</option>
              <option value="0">instant</option>
            </select>
          </label>
          <button type="button" class="link mode-toggle">${o}</button>
          <button type="button" class="link again">rematch</button>
          <button type="button" class="link gear" title="Match settings">⚙</button>
        </header>
        <div class="match-body">
          <section class="board"></section>
          <aside class="side">
            <div class="status">Starting…</div>
            <div class="log" aria-live="polite"></div>
            <form class="free-input">
              <input placeholder="or type a move…" autocomplete="off" />
              <button type="submit">send</button>
            </form>
          </aside>
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
      </div>`,this.logEl=this.root.querySelector(".log"),this.statusEl=this.root.querySelector(".status"),this.root.querySelector(".back").onclick=()=>this.renderHome(),this.root.querySelector(".again").onclick=()=>void this.startMatch(t,e,{...s,seed:String(j())}),this.root.querySelector(".mode-toggle").onclick=()=>void this.startMatch(t,e==="watch"?"play":"watch",{seed:String(j())}),this.root.querySelector(".speed").onchange=i=>{this.speedScale=Number(i.target.value)};const a=this.root.querySelector(".free-input");a.onsubmit=i=>{i.preventDefault();const r=a.querySelector("input");r.value.trim()&&(this.submit(r.value.trim()),r.value="")},this.wireDrawer(t,s)}wireDrawer(t,e){const s=this.root.querySelector(".drawer"),o=s.querySelector(".drawer-fields"),a=r=>r?`<small class="opt-note">${M(r)}</small>`:"",i=()=>{const r=t.optsSchema.find(u=>u.key==="bot"),c=r?r.value.split("|"):[],l=mt(t,e),h=t.optsSchema.find(u=>u.key==="seat"),d=t.solo?"":`<label class="opt-row"><span>seat</span>
             <input name="d-seat" value="${M(e.seat??"0")}" autocomplete="off" />
             ${a(h?.note??"")}</label>`,p=r?`<label class="opt-row"><span>bot</span>
             <select name="d-bot">
               ${t.solo?`<option value=""${l===""?" selected":""}>— you play —</option>`:""}
               ${c.map(u=>`<option value="${M(u)}"${u===l?" selected":""}>${M(u)}</option>`).join("")}
             </select>
             ${a(t.solo?"":r.note)}</label>`:"",g=Je(t.optsSchema,e);o.innerHTML=`
        ${d}
        ${p}
        ${g.map(u=>`<label class="opt-row"${u.bots.length?` data-bots="${M(u.bots.join(" "))}"`:""}>
              <span>${M(u.key)}</span>
              <input name="d-${M(u.key)}" value="${M(u.value)}" autocomplete="off" />
              ${a(u.note)}</label>`).join("")}
        <label class="opt-row"><span>seed</span>
          <input name="d-seed" value="${M(String(e.seed??j()))}" autocomplete="off" /></label>`;const m=o.querySelector('select[name="d-bot"]'),y=()=>{const u=m?.value??"";for(const w of o.querySelectorAll(".opt-row[data-bots]"))w.hidden=!w.dataset.bots.split(" ").includes(u)};m&&(m.onchange=y),y(),s.hidden=!1};this.root.querySelector(".gear").onclick=i,s.querySelector(".drawer-close").onclick=()=>{s.hidden=!0},s.onclick=r=>{r.target===s&&(s.hidden=!0)},s.querySelector(".drawer-apply").onclick=()=>{const r={},c=o.querySelectorAll("input, select");for(const h of c){if(h.closest(".opt-row")?.hidden)continue;const d=h.name.replace(/^d-/,"");h.value.trim()!==""&&(r[d]=h.value.trim())}const l=t.solo?r.bot?"watch":"play":r.seat==="watch"?"watch":"play";this.startMatch(t,l,r)}}async runLoop(t){const e=s=>{t===this.gen&&this.setStatus(xt(s),"error")};for(;t===this.gen;){let s;try{s=await this.host.step()}catch(i){e(i);return}if(t!==this.gen)return;if(s){try{this.log(s),await this.clientBot?.onMove(s);const i=await this.host.state();if(t!==this.gen)return;await this.frontend.animate(s,i)}catch(i){e(i);return}continue}const o=await this.host.state();if(t!==this.gen)return;if(this.frontend.render(o),o.isOver){this.setStatus(o.result??"Game over","result"),this.logText(`— ${o.result??"game over"}`);return}if(this.clientBot&&o.toAct>=0&&o.toAct!==o.humanSeat){this.setStatus("Thinking…");try{const i=await this.clientBot.chooseMove(o);if(t!==this.gen)return;const r=await this.host.apply(i);if(t!==this.gen)return;this.log(r),await this.clientBot.onMove(r);const c=await this.host.state();if(t!==this.gen)return;await this.frontend.animate(r,c)}catch(i){e(i);return}continue}this.setStatus("Your turn"),this.frontend.promptAction(o.labels);const a=await new Promise(i=>this.submitResolve=i);if(t!==this.gen)return;o.numSeats>1&&this.setStatus("Thinking…");try{const i=await this.host.apply(a);if(t!==this.gen)return;this.log(i),await this.clientBot?.onMove(i);const r=await this.host.state();if(t!==this.gen)return;await this.frontend.animate(i,r)}catch(i){e(i)}}}async loadArtifacts(t,e){for(const s of Qe(t.id,e)){const o=`./${wt[s]}`,a=await fetch(o);if(!a.ok)throw new Error(`artifact ${o} missing (HTTP ${a.status})`);await this.host.artifact(s,await a.arrayBuffer())}}submit(t){const e=this.submitResolve;e&&(this.submitResolve=null,e(t))}animationScale(){return window.matchMedia("(prefers-reduced-motion: reduce)").matches?0:this.speedScale}log(t){this.logText(t.text),t.detail&&this.logText(t.detail,!0)}logText(t,e=!1){if(!this.logEl)return;const s=document.createElement("div");s.className=e?"log-line log-detail":"log-line",s.textContent=t,this.logEl.append(s),this.logEl.scrollTop=this.logEl.scrollHeight}setStatus(t,e="info"){this.statusEl&&(this.statusEl.textContent=t,this.statusEl.className=`status status-${e}`)}teardownMatch(){this.clientBot?.cancel(),this.clientBot=null,this.frontend?.unmount(),this.frontend=null,this.submitResolve=null}teardown(){this.gen++,this.tourney?.destroy(),this.tourney=null,this.teardownMatch(),this.logEl=null,this.statusEl=null}}function xt(n){return n instanceof Error?n.message:String(n)}new ts(document.getElementById("app")).start().catch(n=>{document.getElementById("app").innerHTML=`<div class="boot">Failed to start the engine: ${n instanceof Error?n.message:n}</div>`});
