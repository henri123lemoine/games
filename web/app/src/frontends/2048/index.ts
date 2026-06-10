// 2048 frontend: authentic tile board adapted to the dark arcade shell.
//
// View JSON (contract with games/g2048/src/ui.rs):
//   {"cells":[16 ints row-major from top-left, 0 = empty, else tile value],
//    "score":n,"over":bool}
// Transition JSON for shifts: {"dir":"up|down|left|right","gained":n}.
// Tile spawns are chance moves resolved without events, so new tiles are
// found by diffing consecutive views (or, in spectate mode, by searching for
// the spawn that makes the slide reproduce the post-move board).

import type { MatchEventData, ViewState } from '../../engine/protocol';
import type { FrontendCtx, GameFrontend } from '../types';
import { sleep } from '../types';

type Dir = 'up' | 'down' | 'left' | 'right';

const DIRS: Dir[] = ['up', 'down', 'left', 'right'];

const KEY_DIRS: Record<string, Dir> = {
  ArrowUp: 'up',
  ArrowDown: 'down',
  ArrowLeft: 'left',
  ArrowRight: 'right',
  w: 'up',
  s: 'down',
  a: 'left',
  d: 'right',
  W: 'up',
  S: 'down',
  A: 'left',
  D: 'right',
};

const PAD_ARROWS: Record<Dir, string> = { up: '↑', down: '↓', left: '←', right: '→' };

const SLIDE_MS = 110;
const POP_MS = 140;

interface G2048View {
  cells: number[];
  score: number;
  over: boolean;
}

function asView(data: unknown): G2048View | null {
  if (!data || typeof data !== 'object') return null;
  const v = data as Partial<G2048View>;
  if (!Array.isArray(v.cells) || v.cells.length !== 16) return null;
  if (typeof v.score !== 'number' || typeof v.over !== 'boolean') return null;
  return v as G2048View;
}

function isDir(x: unknown): x is Dir {
  return typeof x === 'string' && (DIRS as string[]).includes(x);
}

/** Cell indices of lane `lane` for a shift toward `dir`, front first —
 * mirrors `lane_indices` in games/g2048/src/lib.rs. */
function laneIndices(dir: Dir, lane: number): number[] {
  const idx: number[] = [];
  for (let k = 0; k < 4; k++) {
    if (dir === 'left') idx.push(lane * 4 + k);
    else if (dir === 'right') idx.push(lane * 4 + (3 - k));
    else if (dir === 'up') idx.push(k * 4 + lane);
    else idx.push((3 - k) * 4 + lane);
  }
  return idx;
}

interface TileMove {
  from: number;
  to: number;
  value: number;
  merged: boolean;
}

/** Replays the game's merge rule to learn where every tile travels. */
function computeSlide(cells: number[], dir: Dir): { moves: TileMove[]; after: number[] } {
  const after = new Array<number>(16).fill(0);
  const moves: TileMove[] = [];
  for (let lane = 0; lane < 4; lane++) {
    const idx = laneIndices(dir, lane);
    let len = 0;
    let open = 0;
    for (const i of idx) {
      const v = cells[i];
      if (v === 0) continue;
      if (open === v) {
        after[idx[len - 1]] = v * 2;
        moves.push({ from: i, to: idx[len - 1], value: v, merged: true });
        open = 0;
      } else {
        after[idx[len]] = v;
        moves.push({ from: i, to: idx[len], value: v, merged: false });
        len++;
        open = v;
      }
    }
  }
  return { moves, after };
}

function sameCells(a: number[], b: number[]): boolean {
  return a.every((v, i) => v === b[i]);
}

function tileClasses(value: number): string {
  const ramp = value <= 2048 ? `g2048-v${value}` : 'g2048-vmax';
  const digits = String(value).length;
  const size = digits <= 2 ? 'g2048-d2' : digits === 3 ? 'g2048-d3' : digits === 4 ? 'g2048-d4' : 'g2048-d5';
  return `g2048-tile-inner ${ramp} ${size}`;
}

const STYLE = `
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
  transition: transform ${SLIDE_MS}ms ease-in-out;
  will-change: transform;
}
.g2048-cell-inner, .g2048-tile-inner {
  position: absolute;
  inset: 5px;
  border-radius: 6px;
}
.g2048-cell-inner {
  background: rgba(255, 255, 255, 0.045);
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
  color: var(--accent);
  border: 1px solid var(--accent);
  box-shadow: 0 0 18px rgba(88, 166, 255, 0.4);
}
.g2048-pop { animation: g2048-pop ${POP_MS}ms ease-out backwards; }
@keyframes g2048-pop {
  from { transform: scale(0); }
  to { transform: scale(1); }
}
.g2048-merge { animation: g2048-merge ${POP_MS}ms ease-in-out; }
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
  background: rgba(13, 17, 23, 0.74);
  backdrop-filter: blur(2px);
}
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
`;

function injectStyle(): void {
  if (document.getElementById('g2048-frontend-style')) return;
  const style = document.createElement('style');
  style.id = 'g2048-frontend-style';
  style.textContent = STYLE;
  document.head.append(style);
}

class G2048Frontend implements GameFrontend {
  private ctx!: FrontendCtx;
  private cells: number[] = new Array<number>(16).fill(0);
  private score = 0;
  private pending: string[] | null = null;
  private boardEl!: HTMLElement;
  private tilesEl!: HTMLElement;
  private scoreEl!: HTMLElement;
  private scoreBoxEl!: HTMLElement;
  private bestEl!: HTMLElement;
  private overlayEl!: HTMLElement;
  private overlaySubEl!: HTMLElement;
  private padBtns = new Map<Dir, HTMLButtonElement>();

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    injectStyle();
    host.innerHTML = `
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
      </div>`;
    this.boardEl = host.querySelector('.g2048-board')!;
    this.tilesEl = host.querySelector('.g2048-tiles')!;
    this.scoreEl = host.querySelector('.g2048-score')!;
    this.scoreBoxEl = host.querySelector('.g2048-scorebox-score')!;
    this.bestEl = host.querySelector('.g2048-best')!;
    this.overlayEl = host.querySelector('.g2048-overlay')!;
    this.overlaySubEl = host.querySelector('.g2048-overlay-sub')!;

    const cellsEl = host.querySelector('.g2048-cells')!;
    for (let i = 0; i < 16; i++) {
      const cell = document.createElement('div');
      cell.className = 'g2048-cell';
      this.setPos(cell, i);
      cell.innerHTML = '<div class="g2048-cell-inner"></div>';
      cellsEl.append(cell);
    }

    const pad = host.querySelector<HTMLElement>('.g2048-pad')!;
    if (ctx.humanSeat < 0) {
      pad.classList.add('g2048-hidden');
    } else {
      for (const dir of ['up', 'left', 'down', 'right'] as Dir[]) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = `g2048-btn g2048-btn-${dir}`;
        b.textContent = PAD_ARROWS[dir];
        b.title = dir;
        b.disabled = true;
        b.onclick = () => this.trySubmit(dir);
        pad.append(b);
        this.padBtns.set(dir, b);
      }
    }
    window.addEventListener('keydown', this.onKey);
  }

  render(state: ViewState): void {
    const v = asView(state.viewData);
    if (!v) return;
    if (state.toAct !== state.humanSeat) this.setPending(null);
    const popped: number[] = [];
    let othersEqual = true;
    for (let i = 0; i < 16; i++) {
      if (this.cells[i] === 0 && v.cells[i] !== 0) popped.push(i);
      else if (this.cells[i] !== v.cells[i]) othersEqual = false;
    }
    const popSet =
      othersEqual && popped.length > 0 && popped.length <= 2 && this.ctx.animationScale() > 0
        ? new Set(popped)
        : undefined;
    this.applyView(v, popSet);
  }

  async animate(event: MatchEventData, after: ViewState): Promise<void> {
    const v = asView(after.viewData);
    if (!v) return;
    const scale = this.ctx.animationScale();
    const dir = this.eventDir(event);
    const gained = v.score - this.score;
    if (!dir || scale === 0) {
      this.applyView(v);
      return;
    }
    if (gained > 0) this.showGain(gained);

    let base = this.cells;
    let slide = computeSlide(base, dir);
    if (!sameCells(slide.after, v.cells)) {
      // Spectate mode: a chance spawn resolved silently before this move.
      // Find it, pop it in, then slide from the spawned board.
      const spawned = this.findSpawn(base, dir, v.cells);
      if (!spawned) {
        const allNew = base.every((c) => c === 0)
          ? new Set(v.cells.flatMap((c, i) => (c !== 0 ? [i] : [])))
          : undefined;
        this.applyView(v, allNew);
        return;
      }
      base = spawned.cells;
      slide = computeSlide(base, dir);
      this.cells = base;
      this.rebuildTiles(base, new Set([spawned.cell]));
      await sleep(POP_MS * scale);
    }

    this.tilesEl.replaceChildren();
    const movers: { el: HTMLElement; move: TileMove }[] = [];
    for (const move of slide.moves) {
      const el = this.makeTile(move.value, move.from);
      el.classList.add('g2048-slide');
      el.style.transitionDuration = `${SLIDE_MS * scale}ms`;
      this.tilesEl.append(el);
      movers.push({ el, move });
    }
    void this.tilesEl.offsetWidth;
    for (const { el, move } of movers) this.setPos(el, move.to);
    await sleep(SLIDE_MS * scale + 25);

    const mergedTo = new Set(slide.moves.filter((m) => m.merged).map((m) => m.to));
    this.applyView(v, undefined, mergedTo);
    if (mergedTo.size > 0) await sleep(POP_MS * scale);
  }

  promptAction(labels: string[]): void {
    this.setPending(labels);
  }

  unmount(): void {
    window.removeEventListener('keydown', this.onKey);
  }

  private onKey = (e: KeyboardEvent): void => {
    if (this.ctx.humanSeat < 0 || e.metaKey || e.ctrlKey || e.altKey) return;
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    const dir = KEY_DIRS[e.key];
    if (!dir) return;
    e.preventDefault();
    this.trySubmit(dir);
  };

  private trySubmit(dir: Dir): void {
    if (!this.pending) return;
    const i = this.pending.indexOf(dir);
    if (i < 0) {
      this.shake();
      return;
    }
    this.setPending(null);
    this.ctx.submit(String(i));
  }

  private setPending(labels: string[] | null): void {
    this.pending = labels;
    for (const [dir, btn] of this.padBtns) btn.disabled = !labels || !labels.includes(dir);
  }

  private eventDir(event: MatchEventData): Dir | null {
    const d = event.data;
    if (d && typeof d === 'object' && 'dir' in d) {
      const dir = (d as { dir?: unknown }).dir;
      if (isDir(dir)) return dir;
    }
    return isDir(event.label) ? event.label : null;
  }

  /** Searches for the silent chance spawn (one empty cell becoming 2 or 4)
   * that makes `dir`'s slide reproduce `target`. */
  private findSpawn(
    base: number[],
    dir: Dir,
    target: number[],
  ): { cells: number[]; cell: number } | null {
    for (let cell = 0; cell < 16; cell++) {
      if (base[cell] !== 0) continue;
      for (const value of [2, 4]) {
        const candidate = base.slice();
        candidate[cell] = value;
        if (sameCells(computeSlide(candidate, dir).after, target)) {
          return { cells: candidate, cell };
        }
      }
    }
    return null;
  }

  private applyView(v: G2048View, popSet?: Set<number>, mergeSet?: Set<number>): void {
    this.cells = v.cells.slice();
    this.score = v.score;
    this.rebuildTiles(this.cells, popSet, mergeSet);
    this.scoreEl.textContent = String(v.score);
    this.bestEl.textContent = String(Math.max(0, ...v.cells));
    this.overlayEl.classList.toggle('g2048-show', v.over);
    if (v.over) this.overlaySubEl.textContent = `score ${v.score}`;
  }

  private rebuildTiles(cells: number[], popSet?: Set<number>, mergeSet?: Set<number>): void {
    const scale = this.ctx.animationScale();
    this.tilesEl.replaceChildren();
    for (let i = 0; i < 16; i++) {
      if (cells[i] === 0) continue;
      const tile = this.makeTile(cells[i], i);
      const inner = tile.firstElementChild as HTMLElement;
      if (scale > 0 && popSet?.has(i)) {
        inner.classList.add('g2048-pop');
        inner.style.animationDuration = `${POP_MS * scale}ms`;
      } else if (scale > 0 && mergeSet?.has(i)) {
        inner.classList.add('g2048-merge');
        inner.style.animationDuration = `${POP_MS * scale}ms`;
      }
      this.tilesEl.append(tile);
    }
  }

  private makeTile(value: number, cell: number): HTMLElement {
    const tile = document.createElement('div');
    tile.className = 'g2048-tile';
    this.setPos(tile, cell);
    const inner = document.createElement('div');
    inner.className = tileClasses(value);
    inner.textContent = String(value);
    tile.append(inner);
    return tile;
  }

  private setPos(el: HTMLElement, cell: number): void {
    const row = Math.floor(cell / 4);
    const col = cell % 4;
    el.style.transform = `translate(${col * 100}%, ${row * 100}%)`;
  }

  private showGain(gained: number): void {
    const el = document.createElement('span');
    el.className = 'g2048-gain';
    el.textContent = `+${gained}`;
    el.addEventListener('animationend', () => el.remove());
    this.scoreBoxEl.append(el);
    setTimeout(() => el.remove(), 900);
  }

  private shake(): void {
    this.boardEl.classList.remove('g2048-shake');
    void this.boardEl.offsetWidth;
    this.boardEl.classList.add('g2048-shake');
  }
}

export function createG2048Frontend(): GameFrontend {
  return new G2048Frontend();
}
