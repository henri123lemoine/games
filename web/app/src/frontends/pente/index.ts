// Pente frontend: a cool slate-and-indigo board with carved grid lines, glassy
// stones, soft placement and capture-pair animations, a last-move marker, and
// twin "captured pairs" tracks (first to five pairs — or five in a row — wins).
// The opening is forced to the center, so the board hints that until Black has
// played it.
//
// View schema (games/pente/src/ui.rs::view_data):
//   {size, cells: "<size*size chars b/w/.>", turn, pairs: [b,w], last, winner}
//   cells index = row * size + col, row 0 = board row 1 (bottom).
// Transition schema (transition_data):
//   {move: "g7", seat, point, captured: number[]}   // captured = removed indices

import type { MatchEventData, ViewState } from '../../engine/protocol';
import type { FrontendCtx, GameFrontend } from '../types';
import { sleep } from '../types';

const PAIRS_TO_WIN = 5;
const LINE_TO_WIN = 5;

interface PenteView {
  size: number;
  cells: string;
  turn: number;
  pairs: [number, number];
  last: number | null;
  winner: number | null;
}

function parseView(data: unknown): PenteView | null {
  if (!data || typeof data !== 'object') return null;
  const v = data as PenteView;
  return typeof v.size === 'number' &&
    typeof v.cells === 'string' &&
    v.cells.length === v.size * v.size &&
    Array.isArray(v.pairs)
    ? v
    : null;
}

interface PenteMoveData {
  move: string;
  seat: number;
  point?: number;
  captured?: number[];
}

const STYLE_ID = 'pente-frontend-style';
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

/** Dotted reference points (corners, midlines, center) like a Pente board. */
function dots(size: number): number[] {
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
  }
  return pts;
}

/** The five-in-a-row line through `p` for `color`, as cell indices, or null.
 * The board never carries a win-line, so it is recovered from the stones when
 * a line win is shown. */
function winLine(cells: string, size: number, color: string): number[] | null {
  const dirs: [number, number][] = [
    [0, 1],
    [1, 0],
    [1, 1],
    [1, -1],
  ];
  for (let p = 0; p < cells.length; p++) {
    if (cells[p] !== color) continue;
    const row = Math.floor(p / size);
    const col = p % size;
    for (const [dr, dc] of dirs) {
      const run = [p];
      let r = row + dr;
      let c = col + dc;
      while (r >= 0 && c >= 0 && r < size && c < size && cells[r * size + c] === color) {
        run.push(r * size + c);
        r += dr;
        c += dc;
      }
      if (run.length >= LINE_TO_WIN) return run;
    }
  }
  return null;
}

const CSS = `
.pente { display: flex; flex-direction: column; gap: 14px; }
.pente-hud { display: grid; grid-template-columns: 1fr auto 1fr; align-items: stretch; gap: 10px; }
.pente-player { display: flex; align-items: center; gap: 10px; padding: 8px 12px; min-width: 0;
  border-radius: var(--radius); background: var(--bg-raised); border: 1px solid var(--border);
  transition: border-color .2s, box-shadow .2s; }
.pente-player.pente-active { border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent), 0 0 18px rgba(99, 102, 241, .26); }
.pente-stone-icon { width: 22px; height: 22px; border-radius: 50%; flex: none;
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, .25), 0 1px 3px rgba(0, 0, 0, .5); }
.pente-stone-icon-b { background: radial-gradient(circle at 34% 28%, #5b6478, #24293a 44%, #070910); }
.pente-stone-icon-w { background: radial-gradient(circle at 34% 28%, #ffffff, #e7eaf4 58%, #b9bfd2); }
.pente-pinfo { display: flex; flex-direction: column; min-width: 0; }
.pente-pname { font-weight: 600; line-height: 1.2; }
.pente-psub { font-size: 12px; color: var(--text-dim); white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.pente-pcaps { margin-left: auto; text-align: right; font-size: 11px; color: var(--text-dim);
  line-height: 1.15; white-space: nowrap; }
.pente-pcaps b { display: block; font-size: 16px; color: var(--text); }
.pente-pips { display: inline-flex; gap: 3px; margin-top: 3px; justify-content: flex-end; }
.pente-pip { width: 7px; height: 7px; border-radius: 50%; background: var(--border);
  transition: background .25s, box-shadow .25s; }
.pente-pip.pente-pip-on { background: var(--accent);
  box-shadow: 0 0 6px rgba(99, 102, 241, .7); }
.pente-turn-chip { align-self: center; display: flex; align-items: center; gap: 8px; padding: 7px 14px;
  border-radius: 999px; background: var(--bg-inset); border: 1px solid var(--border);
  font-size: 13px; color: var(--text-dim); white-space: nowrap; }
.pente-turn-dot { width: 11px; height: 11px; border-radius: 50%; flex: none;
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, .25), 0 1px 2px rgba(0, 0, 0, .4); }
.pente-board-wrap { position: relative; width: 100%; max-width: min(74vh, 640px); margin: 0 auto; }
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
}
`;

function ensureStyle(): void {
  if (document.getElementById(STYLE_ID)) return;
  const el = document.createElement('style');
  el.id = STYLE_ID;
  el.textContent = CSS;
  document.head.append(el);
}

class PenteFrontend implements GameFrontend {
  private ctx!: FrontendCtx;
  private svg!: SVGSVGElement;
  private stonesG!: SVGGElement;
  private fxG!: SVGGElement;
  private ghostEl!: SVGCircleElement;
  private markerEl!: SVGCircleElement;
  private winLineEl!: SVGPathElement;
  private toastEl!: HTMLElement;
  private turnChip!: HTMLElement;
  private plaques: HTMLElement[] = [];
  private capEls: HTMLElement[] = [];
  private pipRows: HTMLElement[] = [];

  private size = 0;
  private view: PenteView | null = null;
  private lastMove: number | null = null;
  private interactive = false;
  private labelIndex = new Map<string, number>();
  private legalPoints = new Set<number>();
  private stoneEls = new Map<number, SVGCircleElement>();

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    ensureStyle();
    host.innerHTML = `
      <div class="pente">
        <div class="pente-hud">
          <div class="pente-player" data-seat="0">
            <span class="pente-stone-icon pente-stone-icon-b"></span>
            <span class="pente-pinfo"><span class="pente-pname">Black</span><span class="pente-psub"></span><span class="seat-slot" data-seat="0"></span></span>
            <span class="pente-pcaps"><b>0</b>pairs<span class="pente-pips" data-seat="0"></span></span>
          </div>
          <div class="pente-turn-chip"><span class="pente-turn-dot"></span><span class="pente-turn-text"></span></div>
          <div class="pente-player" data-seat="1">
            <span class="pente-stone-icon pente-stone-icon-w"></span>
            <span class="pente-pinfo"><span class="pente-pname">White</span><span class="pente-psub"></span><span class="seat-slot" data-seat="1"></span></span>
            <span class="pente-pcaps"><b>0</b>pairs<span class="pente-pips" data-seat="1"></span></span>
          </div>
        </div>
        <div class="pente-board-wrap">
          <svg class="pente-svg" role="img" aria-label="Pente board"></svg>
          <div class="pente-toast"></div>
        </div>
      </div>`;
    this.svg = host.querySelector('.pente-svg')!;
    this.toastEl = host.querySelector('.pente-toast')!;
    this.turnChip = host.querySelector('.pente-turn-chip')!;
    this.plaques = [...host.querySelectorAll<HTMLElement>('.pente-player')];
    this.capEls = this.plaques.map((p) => p.querySelector<HTMLElement>('.pente-pcaps b')!);
    this.pipRows = [...host.querySelectorAll<HTMLElement>('.pente-pips')];
    for (const row of this.pipRows) {
      for (let i = 0; i < PAIRS_TO_WIN; i++) {
        const pip = document.createElement('span');
        pip.className = 'pente-pip';
        row.append(pip);
      }
    }
    for (const [seat, plaque] of this.plaques.entries()) {
      const sub = plaque.querySelector<HTMLElement>('.pente-psub')!;
      sub.textContent = seat === ctx.humanSeat ? 'you' : 'bot';
    }
  }

  private xy(p: number): { x: number; y: number } {
    return { x: PAD + (p % this.size), y: PAD + (this.size - 1 - Math.floor(p / this.size)) };
  }

  private buildBoard(size: number): void {
    this.size = size;
    const ext = size - 1 + 2 * PAD;
    this.svg.setAttribute('viewBox', `0 0 ${ext} ${ext}`);
    const dotMarks = dots(size)
      .map((p) => {
        const { x, y } = this.xy(p);
        return `<circle cx="${x}" cy="${y}" r="${size > 13 ? 0.08 : 0.1}" fill="rgba(150,160,210,.5)"/>`;
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
      hits.push(`<rect class="pente-hit" data-p="${p}" x="${x - 0.5}" y="${y - 0.5}" width="1" height="1"/>`);
    }
    this.svg.innerHTML = `
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
      <rect width="${ext}" height="${ext}" rx="0.32" fill="url(#pente-board)"/>
      <rect width="${ext}" height="${ext}" rx="0.32" fill="url(#pente-sheen)"/>
      <path d="${gridPath(size)}" stroke="rgba(150,162,210,.4)" stroke-width="0.028" fill="none" stroke-linecap="square"/>
      ${dotMarks}
      <g fill="rgba(170,180,222,.5)" font-size="0.32" text-anchor="middle" font-family="inherit">${labels.join('')}</g>
      <g class="pente-stones" filter="url(#pente-shadow)"></g>
      <g class="pente-fx"></g>
      <path class="pente-winline" fill="none" stroke="rgba(129,140,248,.9)" stroke-width="0.12" stroke-linecap="round" opacity="0"/>
      <circle class="pente-marker" r="0.17" fill="none" stroke-width="0.07" opacity="0"/>
      <circle class="pente-ghost" r="0.45" opacity="0"/>
      <g class="pente-hits"></g>`;
    this.stonesG = this.svg.querySelector('.pente-stones')!;
    this.fxG = this.svg.querySelector('.pente-fx')!;
    this.winLineEl = this.svg.querySelector('.pente-winline')!;
    this.markerEl = this.svg.querySelector('.pente-marker')!;
    this.ghostEl = this.svg.querySelector('.pente-ghost')!;
    const hitsG = this.svg.querySelector<SVGGElement>('.pente-hits')!;
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
      this.ctx.humanSeat === 1 ? 'rgba(250,250,255,.6)' : 'rgba(14,16,28,.6)',
    );
    this.ghostEl.setAttribute('opacity', '1');
  }

  private setInteractive(on: boolean): void {
    this.interactive = on;
    if (!on) this.ghostEl.setAttribute('opacity', '0');
    this.svg
      .querySelectorAll('.pente-hit')
      .forEach((el) =>
        el.classList.toggle('pente-hit-on', on && this.legalPoints.has(Number(el.getAttribute('data-p')))),
      );
  }

  private drawStones(v: PenteView): void {
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
      this.markerEl.setAttribute('stroke', v.cells[this.lastMove] === 'b' ? '#eef0fb' : '#1a1c2c');
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
    c.setAttribute('r', '0.46');
    c.setAttribute('fill', color === 0 ? 'url(#pente-stone-b)' : 'url(#pente-stone-w)');
    this.stoneEls.set(p, c);
    return c;
  }

  /** Light the winning line and pulse its stones when the game ends on five. */
  private showWin(v: PenteView): void {
    if (v.winner === null) return;
    if (v.pairs[v.winner] >= PAIRS_TO_WIN) return; // a capture win has no line
    const line = winLine(v.cells, v.size, v.winner === 0 ? 'b' : 'w');
    if (!line) return;
    const d = line
      .map((p, i) => {
        const { x, y } = this.xy(p);
        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
      })
      .join(' ');
    this.winLineEl.setAttribute('d', d);
    this.winLineEl.setAttribute('opacity', '1');
    for (const p of line) this.stoneEls.get(p)?.classList.add('pente-win-stone');
  }

  render(state: ViewState): void {
    const v = parseView(state.viewData);
    if (!v) return;
    if (v.size !== this.size) this.buildBoard(v.size);
    this.view = v;
    this.winLineEl.setAttribute('opacity', '0');
    this.drawStones(v);
    for (let seat = 0; seat < 2; seat++) {
      this.capEls[seat].textContent = String(v.pairs[seat]);
      const pips = this.pipRows[seat].children;
      for (let i = 0; i < pips.length; i++) {
        pips[i].classList.toggle('pente-pip-on', i < v.pairs[seat]);
      }
    }
    const dot = this.turnChip.querySelector<HTMLElement>('.pente-turn-dot')!;
    const text = this.turnChip.querySelector<HTMLElement>('.pente-turn-text')!;
    if (state.isOver) {
      this.showWin(v);
      text.textContent =
        v.winner === null
          ? 'Draw — board full'
          : `${v.winner === 0 ? 'Black' : 'White'} wins`;
      dot.style.background = 'var(--text-dim)';
      this.plaques.forEach((pl) => pl.classList.remove('pente-active'));
    } else {
      const center = v.cells.split('').every((ch) => ch === '.');
      text.textContent = center
        ? 'Black opens at the center'
        : v.turn === 0
          ? 'Black to move'
          : 'White to move';
      dot.style.background =
        v.turn === 0
          ? 'radial-gradient(circle at 35% 30%, #5b6478, #070910)'
          : 'radial-gradient(circle at 35% 30%, #ffffff, #b9bfd2)';
      this.plaques.forEach((pl, seat) => pl.classList.toggle('pente-active', seat === v.turn));
    }
    if (state.toAct !== state.humanSeat) this.setInteractive(false);
  }

  async animate(event: MatchEventData, after: ViewState): Promise<void> {
    const d = (event.data ?? null) as PenteMoveData | null;
    const scale = this.ctx.animationScale();
    const v = parseView(after.viewData);
    if (v && v.size !== this.size) this.buildBoard(v.size);
    if (d && typeof d.point === 'number') {
      this.lastMove = d.point;
      this.render(after);
      if (scale > 0) {
        const stone = this.stoneEls.get(d.point);
        if (stone) {
          stone.style.animationDuration = `${260 * scale}ms`;
          stone.classList.add('pente-drop');
        }
        const captured = d.captured ?? [];
        for (const q of captured) {
          const dying = this.makeStone(q, d.seat ^ 1);
          this.stoneEls.delete(q);
          dying.style.animationDuration = `${360 * scale}ms`;
          dying.style.animationDelay = `${110 * scale}ms`;
          dying.classList.add('pente-cap');
          this.fxG.append(dying);
        }
        if (captured.length > 0 && !after.isOver) {
          const pairs = captured.length / 2;
          this.toastEl.textContent = `${d.seat === 0 ? 'Black' : 'White'} captures ${pairs} pair${pairs === 1 ? '' : 's'}`;
          this.toastEl.classList.add('pente-toast-show');
        }
        await sleep((captured.length > 0 ? 520 : 300) * scale);
        this.fxG.replaceChildren();
        this.toastEl.classList.remove('pente-toast-show');
      }
    } else {
      this.render(after);
      await sleep(200 * scale);
    }
  }

  promptAction(labels: string[]): void {
    this.labelIndex = new Map(labels.map((l, i) => [l, i]));
    this.legalPoints = new Set(
      labels.map((l) => parseCoord(l, this.size)).filter((p): p is number => p !== null),
    );
    this.setInteractive(true);
  }

  unmount(): void {}
}

export function createPenteFrontend(): GameFrontend {
  return new PenteFrontend();
}
