// Go frontend: a wood-toned goban with slate & shell stones, soft drop and
// capture animations, last-move marker, and capture counts.
//
// View schema (games/go/src/ui.rs::view_data):
//   {size, cells: "<size*size chars b/w/.>", turn, captures: [b,w], lastMove, komi}
//   cells index = row * size + col, row 0 = board row 1 (bottom).
// Transition schema (transition_data):
//   {move: "c3"|"pass", seat, point?, captured?: number[]}

import type { MatchEventData, ViewState } from '../../engine/protocol';
import type { FrontendCtx, GameFrontend } from '../types';
import { sleep } from '../types';

interface GoView {
  size: number;
  cells: string;
  turn: number;
  captures: [number, number];
  lastMove: string | null;
  komi: number;
}

function parseView(data: unknown): GoView | null {
  if (!data || typeof data !== 'object') return null;
  const v = data as GoView;
  return typeof v.size === 'number' &&
    typeof v.cells === 'string' &&
    v.cells.length === v.size * v.size &&
    Array.isArray(v.captures)
    ? v
    : null;
}

interface GoMoveData {
  move: string;
  seat: number;
  point?: number;
  captured?: number[];
}

const STYLE_ID = 'go-frontend-style';
const PAD = 1.0;

function colLetter(col: number): string {
  return String.fromCharCode(97 + col + (col >= 8 ? 1 : 0));
}

function coordLabel(p: number, size: number): string {
  return `${colLetter(p % size)}${Math.floor(p / size) + 1}`;
}

function parseCoord(label: string, size: number): number | null {
  const c = label.charCodeAt(0) - 97;
  if (c < 0 || c > 25 || label[0] === 'i') return null;
  const col = c > 8 ? c - 1 : c;
  const row = parseInt(label.slice(1), 10);
  if (!Number.isFinite(row) || col >= size || row < 1 || row > size) return null;
  return (row - 1) * size + col;
}

function gridPath(size: number): string {
  const lines: string[] = [];
  const end = PAD + size - 1;
  for (let i = 0; i < size; i++) {
    const v = PAD + i;
    lines.push(`M ${v} ${PAD} L ${v} ${end}`, `M ${PAD} ${v} L ${end} ${v}`);
  }
  return lines.join(' ');
}

function hoshiPoints(size: number): number[] {
  const pts: number[] = [];
  const edge = size >= 13 ? 3 : 2;
  if (size >= 7) {
    for (const r of [edge, size - 1 - edge]) {
      for (const c of [edge, size - 1 - edge]) pts.push(r * size + c);
    }
  }
  if (size % 2 === 1 && size >= 5) {
    const mid = (size - 1) / 2;
    pts.push(mid * size + mid);
    if (size >= 15) {
      pts.push(edge * size + mid, (size - 1 - edge) * size + mid);
      pts.push(mid * size + edge, mid * size + (size - 1 - edge));
    }
  }
  return pts;
}

const CSS = `
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
`;

function ensureStyle(): void {
  if (document.getElementById(STYLE_ID)) return;
  const el = document.createElement('style');
  el.id = STYLE_ID;
  el.textContent = CSS;
  document.head.append(el);
}

class GoFrontend implements GameFrontend {
  private ctx!: FrontendCtx;
  private svg!: SVGSVGElement;
  private stonesG!: SVGGElement;
  private fxG!: SVGGElement;
  private ghostEl!: SVGCircleElement;
  private markerEl!: SVGCircleElement;
  private toastEl!: HTMLElement;
  private passBtn!: HTMLButtonElement;
  private turnChip!: HTMLElement;
  private plaques: HTMLElement[] = [];
  private capEls: HTMLElement[] = [];

  private size = 0;
  private view: GoView | null = null;
  private lastMove: number | null = null;
  private interactive = false;
  private labelIndex = new Map<string, number>();
  private legalPoints = new Set<number>();
  private stoneEls = new Map<number, SVGCircleElement>();

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    ensureStyle();
    host.innerHTML = `
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
      </div>`;
    this.svg = host.querySelector('.go-svg')!;
    this.toastEl = host.querySelector('.go-toast')!;
    this.passBtn = host.querySelector('.go-pass')!;
    this.turnChip = host.querySelector('.go-turn-chip')!;
    this.plaques = [...host.querySelectorAll<HTMLElement>('.go-player')];
    this.capEls = this.plaques.map((p) => p.querySelector<HTMLElement>('.go-pcaps b')!);
    for (const [seat, plaque] of this.plaques.entries()) {
      const sub = plaque.querySelector<HTMLElement>('.go-psub')!;
      const bits = [seat === ctx.humanSeat ? 'you' : 'bot'];
      if (seat === 1) bits.push(`+${this.view?.komi ?? 7.5} komi`);
      sub.textContent = bits.join(' · ');
    }
    if (ctx.humanSeat < 0) this.passBtn.style.display = 'none';
    this.passBtn.onclick = () => {
      const idx = this.labelIndex.get('pass');
      if (!this.interactive || idx === undefined) return;
      this.setInteractive(false);
      this.ctx.submit(String(idx));
    };
  }

  private xy(p: number): { x: number; y: number } {
    return { x: PAD + (p % this.size), y: PAD + (this.size - 1 - Math.floor(p / this.size)) };
  }

  private buildBoard(size: number): void {
    this.size = size;
    const ext = size - 1 + 2 * PAD;
    this.svg.setAttribute('viewBox', `0 0 ${ext} ${ext}`);
    const hoshi = hoshiPoints(size)
      .map((p) => {
        const { x, y } = this.xy(p);
        return `<circle cx="${x}" cy="${y}" r="${size > 13 ? 0.08 : 0.1}" fill="rgba(40,24,8,.78)"/>`;
      })
      .join('');
    const labels: string[] = [];
    for (let c = 0; c < size; c++) {
      labels.push(
        `<text x="${PAD + c}" y="${PAD + size - 1 + 0.72}">${colLetter(c)}</text>`,
        `<text x="${PAD - 0.66}" y="${PAD + (size - 1 - c) + 0.11}">${c + 1}</text>`,
      );
    }
    const hits: string[] = [];
    for (let p = 0; p < size * size; p++) {
      const { x, y } = this.xy(p);
      hits.push(`<rect class="go-hit" data-p="${p}" x="${x - 0.5}" y="${y - 0.5}" width="1" height="1"/>`);
    }
    this.svg.innerHTML = `
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
      <rect width="${ext}" height="${ext}" rx="0.32" fill="url(#go-wood)"/>
      <rect width="${ext}" height="${ext}" rx="0.32" fill="url(#go-sheen)"/>
      <path d="${gridPath(size)}" stroke="rgba(46,28,8,.8)" stroke-width="0.032" fill="none" stroke-linecap="square"/>
      ${hoshi}
      <g fill="rgba(46,28,8,.55)" font-size="0.32" text-anchor="middle" font-family="inherit">${labels.join('')}</g>
      <g class="go-stones" filter="url(#go-shadow)"></g>
      <g class="go-fx"></g>
      <circle class="go-marker" r="0.17" fill="none" stroke-width="0.07" opacity="0"/>
      <circle class="go-ghost" r="0.45" opacity="0"/>
      <g class="go-hits"></g>`;
    this.stonesG = this.svg.querySelector('.go-stones')!;
    this.fxG = this.svg.querySelector('.go-fx')!;
    this.markerEl = this.svg.querySelector('.go-marker')!;
    this.ghostEl = this.svg.querySelector('.go-ghost')!;
    const hitsG = this.svg.querySelector<SVGGElement>('.go-hits')!;
    hitsG.innerHTML = hits.join('');
    const pointOf = (e: Event): number | null => {
      const attr = (e.target as Element).getAttribute?.('data-p');
      return attr === null || attr === undefined ? null : Number(attr);
    };
    hitsG.addEventListener('click', (e) => {
      const p = pointOf(e);
      if (p !== null) this.tryPlay(p);
    });
    hitsG.addEventListener('pointerover', (e) => this.showGhost(pointOf(e)));
    hitsG.addEventListener('pointerout', () => this.showGhost(null));
  }

  private tryPlay(p: number): void {
    if (!this.interactive || !this.legalPoints.has(p)) return;
    const idx = this.labelIndex.get(coordLabel(p, this.size));
    if (idx === undefined) return;
    this.setInteractive(false);
    this.ctx.submit(String(idx));
  }

  private showGhost(p: number | null): void {
    if (
      p === null ||
      !this.interactive ||
      !this.legalPoints.has(p) ||
      this.view?.cells[p] !== '.'
    ) {
      this.ghostEl.setAttribute('opacity', '0');
      return;
    }
    const { x, y } = this.xy(p);
    this.ghostEl.setAttribute('cx', String(x));
    this.ghostEl.setAttribute('cy', String(y));
    this.ghostEl.setAttribute(
      'fill',
      this.ctx.humanSeat === 1 ? 'rgba(250,248,238,.62)' : 'rgba(12,12,16,.55)',
    );
    this.ghostEl.setAttribute('opacity', '1');
  }

  private setInteractive(on: boolean): void {
    this.interactive = on;
    this.passBtn.disabled = !on || !this.labelIndex.has('pass');
    if (!on) this.ghostEl.setAttribute('opacity', '0');
    this.svg
      .querySelectorAll('.go-hit')
      .forEach((el) => el.classList.toggle('go-hit-on', on && this.legalPoints.has(Number(el.getAttribute('data-p')))));
  }

  private drawStones(v: GoView): void {
    this.stoneEls.clear();
    this.stonesG.replaceChildren();
    for (let p = 0; p < v.cells.length; p++) {
      const ch = v.cells[p];
      if (ch !== 'b' && ch !== 'w') continue;
      this.stonesG.append(this.makeStone(p, ch === 'b' ? 0 : 1));
    }
    if (this.lastMove !== null && v.cells[this.lastMove] !== '.') {
      const { x, y } = this.xy(this.lastMove);
      this.markerEl.setAttribute('cx', String(x));
      this.markerEl.setAttribute('cy', String(y));
      this.markerEl.setAttribute(
        'stroke',
        v.cells[this.lastMove] === 'b' ? '#f2f0e4' : '#1c1c20',
      );
      this.markerEl.setAttribute('opacity', '1');
    } else {
      this.markerEl.setAttribute('opacity', '0');
    }
  }

  private makeStone(p: number, color: number): SVGCircleElement {
    const { x, y } = this.xy(p);
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('cx', String(x));
    c.setAttribute('cy', String(y));
    c.setAttribute('r', '0.47');
    c.setAttribute('fill', color === 0 ? 'url(#go-stone-b)' : 'url(#go-stone-w)');
    this.stoneEls.set(p, c);
    return c;
  }

  render(state: ViewState): void {
    const v = parseView(state.viewData);
    if (!v) return;
    if (v.size !== this.size) this.buildBoard(v.size);
    this.view = v;
    this.drawStones(v);
    this.capEls[0].textContent = String(v.captures[0]);
    this.capEls[1].textContent = String(v.captures[1]);
    const dot = this.turnChip.querySelector<HTMLElement>('.go-turn-dot')!;
    const text = this.turnChip.querySelector<HTMLElement>('.go-turn-text')!;
    if (state.isOver) {
      text.textContent = 'Game over';
      dot.style.background = 'var(--text-dim)';
      this.plaques.forEach((pl) => pl.classList.remove('go-active'));
    } else {
      text.textContent = v.turn === 0 ? 'Black to move' : 'White to move';
      dot.style.background =
        v.turn === 0
          ? 'radial-gradient(circle at 35% 30%, #7c8088, #0a0a0d)'
          : 'radial-gradient(circle at 35% 30%, #ffffff, #c4c0ae)';
      this.plaques.forEach((pl, seat) => pl.classList.toggle('go-active', seat === v.turn));
    }
    if (state.toAct !== state.humanSeat) this.setInteractive(false);
  }

  async animate(event: MatchEventData, after: ViewState): Promise<void> {
    const d = (event.data ?? null) as GoMoveData | null;
    const scale = this.ctx.animationScale();
    const v = parseView(after.viewData);
    if (v && v.size !== this.size) this.buildBoard(v.size);
    if (d && typeof d.point === 'number') {
      this.lastMove = d.point;
      this.render(after);
      if (scale > 0) {
        const stone = this.stoneEls.get(d.point);
        if (stone) {
          stone.style.animationDuration = `${280 * scale}ms`;
          stone.classList.add('go-drop');
        }
        const captured = d.captured ?? [];
        for (const q of captured) {
          const dying = this.makeStone(q, d.seat ^ 1);
          this.stoneEls.delete(q);
          dying.style.animationDuration = `${340 * scale}ms`;
          dying.style.animationDelay = `${120 * scale}ms`;
          dying.classList.add('go-die');
          this.fxG.append(dying);
        }
        await sleep((captured.length > 0 ? 500 : 300) * scale);
        this.fxG.replaceChildren();
      }
    } else if (d && d.move === 'pass') {
      this.lastMove = null;
      this.render(after);
      if (scale > 0) {
        this.toastEl.textContent = `${d.seat === 0 ? 'Black' : 'White'} passes`;
        this.toastEl.classList.add('go-toast-show');
        await sleep(650 * scale);
        this.toastEl.classList.remove('go-toast-show');
      }
    } else {
      this.render(after);
      await sleep(200 * scale);
    }
  }

  promptAction(labels: string[]): void {
    this.labelIndex = new Map(labels.map((l, i) => [l, i]));
    this.legalPoints = new Set(
      labels
        .map((l) => parseCoord(l, this.size))
        .filter((p): p is number => p !== null),
    );
    this.setInteractive(true);
  }

  unmount(): void {}
}

export function createGoFrontend(): GameFrontend {
  return new GoFrontend();
}
