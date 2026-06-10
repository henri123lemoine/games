// Othello frontend: green felt board with star points, 3D discs, legal-move
// dots on your turn, a pop-in for the placed disc and a 3D flip wave that
// radiates outward from it, plus live disc counts per side.
//
// View schema (games/othello/src/ui.rs::view_data):
//   { cells: string,        // 64 chars, square 0 (a1) .. 63 (h8), row 1
//                           //   first: '.' empty, 'b' Black (seat 0), 'w' White
//     turn: 0 | 1,          // side to move
//     counts: [number, number],   // [black, white]
//     legal: string[] }     // square names ("c4") the side to move can play
//
// Transition schema (ui.rs::transition_data):
//   { move: string,         // square name, or "pass"
//     player: 0 | 1,
//     placed: number | null,   // square index
//     flipped: number[] }      // square indices turned over

import type { MatchEventData, ViewState } from '../../engine/protocol';
import type { FrontendCtx, GameFrontend } from '../types';
import { sleep } from '../types';

const SIZE = 8;
const PLAYER_NAMES = ['Black', 'White'];

interface OtView {
  cells: string;
  turn: number;
  counts: [number, number];
  legal: string[];
}

interface OtMove {
  move: string;
  player: number;
  placed: number | null;
  flipped: number[];
}

function parseView(data: unknown): OtView | null {
  if (!data || typeof data !== 'object') return null;
  const v = data as OtView;
  return typeof v.cells === 'string' &&
    v.cells.length === SIZE * SIZE &&
    Array.isArray(v.counts) &&
    Array.isArray(v.legal)
    ? v
    : null;
}

function parseMove(data: unknown): OtMove | null {
  if (!data || typeof data !== 'object') return null;
  const m = data as OtMove;
  return typeof m.move === 'string' && typeof m.player === 'number' && Array.isArray(m.flipped)
    ? m
    : null;
}

/** "c4" → square index 0..63, matching the view's `cells` order. */
function squareIndex(name: string): number | null {
  if (!/^[a-h][1-8]$/.test(name)) return null;
  return (name.charCodeAt(1) - 49) * SIZE + (name.charCodeAt(0) - 97);
}

function chebyshev(a: number, b: number): number {
  return Math.max(
    Math.abs(Math.floor(a / SIZE) - Math.floor(b / SIZE)),
    Math.abs((a % SIZE) - (b % SIZE)),
  );
}

/** Recover the transition from a state diff when the event carries no data. */
function diffMove(prev: OtView | null, after: OtView | null): OtMove | null {
  if (!prev || !after) return null;
  let placed: number | null = null;
  const flipped: number[] = [];
  for (let sq = 0; sq < SIZE * SIZE; sq++) {
    if (prev.cells[sq] === after.cells[sq]) continue;
    if (prev.cells[sq] === '.') placed = sq;
    else flipped.push(sq);
  }
  if (placed === null) return { move: 'pass', player: prev.turn, placed: null, flipped: [] };
  return {
    move: 'place',
    player: after.cells[placed] === 'b' ? 0 : 1,
    placed,
    flipped,
  };
}

const STYLE_ID = 'othello-frontend-style';

const CSS = `
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
  grid-template-columns: repeat(${SIZE}, 1fr);
  grid-template-rows: repeat(${SIZE}, 1fr);
  aspect-ratio: 1;
  border: 10px solid #18221b;
  border-radius: var(--radius);
  background:
    repeating-linear-gradient(48deg, rgba(255, 255, 255, 0.02) 0 2px, transparent 2px 5px),
    linear-gradient(158deg, #31894e, #1d5c31);
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
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text);
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
`;

function injectStyle(): void {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.append(style);
}

class OthelloFrontend implements GameFrontend {
  private ctx!: FrontendCtx;
  private rootEl!: HTMLElement;
  private boardEl!: HTMLElement;
  private msgEl!: HTMLElement;
  private toastEl!: HTMLElement;
  private passEl!: HTMLButtonElement;
  private fallbackEl!: HTMLElement;
  private scoreEls: HTMLElement[] = [];
  private countEls: HTMLElement[] = [];
  private cells: HTMLElement[] = [];
  private discs = new Map<number, HTMLElement>();
  private view: OtView | null = null;
  private actionBySq: Map<number, number> | null = null;
  private anims = new Set<Animation>();

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    injectStyle();
    const score = (p: number) => `
      <div class="ot-score ot-score-${p}">
        <span class="ot-mini ot-mini-${p === 0 ? 'b' : 'w'}"></span>
        <span>${PLAYER_NAMES[p]} · ${ctx.humanSeat === p ? 'You' : 'Bot'}</span>
        <span class="ot-count">0</span>
      </div>`;
    host.innerHTML = `
      <div class="ot-root">
        <div class="ot-bar">${score(0)}<div class="ot-msg"></div>${score(1)}</div>
        <div class="ot-board">
          ${'<div class="ot-cell"></div>'.repeat(SIZE * SIZE)}
          <div class="ot-toast"></div>
          <button type="button" class="ot-pass">Pass</button>
        </div>
        <pre class="ot-fallback"></pre>
      </div>`;
    this.rootEl = host.querySelector('.ot-root')!;
    this.boardEl = host.querySelector('.ot-board')!;
    this.msgEl = host.querySelector('.ot-msg')!;
    this.toastEl = host.querySelector('.ot-toast')!;
    this.passEl = host.querySelector('.ot-pass')!;
    this.fallbackEl = host.querySelector('.ot-fallback')!;
    this.scoreEls = [host.querySelector('.ot-score-0')!, host.querySelector('.ot-score-1')!];
    this.countEls = this.scoreEls.map((el) => el.querySelector('.ot-count')!);
    this.cells = [...this.boardEl.querySelectorAll<HTMLElement>('.ot-cell')];
    for (const x of [25, 75]) {
      for (const y of [25, 75]) {
        const star = document.createElement('div');
        star.className = 'ot-star';
        star.style.left = `${x}%`;
        star.style.top = `${y}%`;
        this.boardEl.append(star);
      }
    }
    this.boardEl.addEventListener('click', (e) => {
      const cell = (e.target as HTMLElement).closest('.ot-cell');
      if (cell) this.clickSquare(this.cells.indexOf(cell as HTMLElement));
    });
  }

  render(state: ViewState): void {
    this.disableInput();
    const view = parseView(state.viewData);
    this.view = view;
    if (!view) {
      this.rootEl.classList.add('ot-text-only');
      this.fallbackEl.textContent = state.view;
      return;
    }
    this.rootEl.classList.remove('ot-text-only');
    this.rebuildDiscs(view);
    for (let p = 0; p < 2; p++) {
      this.countEls[p].textContent = String(view.counts[p]);
      this.scoreEls[p].classList.toggle('ot-active', !state.isOver && view.turn === p);
    }
    if (this.ctx.humanSeat < 0 && !state.isOver) {
      for (const name of view.legal) {
        const sq = squareIndex(name);
        if (sq !== null) this.cells[sq].classList.add('ot-hint');
      }
    }
    const [b, w] = view.counts;
    this.msgEl.textContent = state.isOver
      ? b === w
        ? `Draw, ${b}–${w}`
        : `${PLAYER_NAMES[b > w ? 0 : 1]} wins ${Math.max(b, w)}–${Math.min(b, w)}`
      : `${PLAYER_NAMES[view.turn]} to move`;
  }

  async animate(event: MatchEventData, after: ViewState): Promise<void> {
    const prev = this.view;
    const view = parseView(after.viewData);
    const move = parseMove(event.data) ?? diffMove(prev, view);
    const scale = this.ctx.animationScale();

    if (move?.move === 'pass' || move?.placed == null) {
      if (scale > 0 && move) {
        this.toastEl.textContent = `${PLAYER_NAMES[move.player]} passes`;
        this.toastEl.classList.add('ot-show');
        await sleep(800 * scale);
        this.toastEl.classList.remove('ot-show');
      }
      this.render(after);
      return;
    }

    this.render(after);
    if (!view || scale <= 0) return;

    const waits: Promise<void>[] = [];
    const placedDisc = this.discs.get(move.placed);
    if (placedDisc) {
      waits.push(
        this.run(
          placedDisc.animate(
            [
              { transform: 'scale(0.2)', opacity: 0.4, offset: 0 },
              { transform: 'scale(1.14)', opacity: 1, offset: 0.7 },
              { transform: 'scale(1)', offset: 1 },
            ],
            { duration: 240 * scale, easing: 'ease-out' },
          ),
        ),
      );
    }
    for (const sq of move.flipped) {
      const flip = this.discs.get(sq)?.querySelector<HTMLElement>('.ot-flip');
      if (!flip) continue;
      const toWhite = view.cells[sq] === 'w';
      const [from, mid, to] = toWhite ? [0, 90, 180] : [180, 270, 360];
      waits.push(
        this.run(
          flip.animate(
            [
              { transform: `rotateY(${from}deg) scale(1)` },
              { transform: `rotateY(${mid}deg) scale(1.18)` },
              { transform: `rotateY(${to}deg) scale(1)` },
            ],
            {
              duration: 340 * scale,
              delay: (110 + 85 * (chebyshev(move.placed, sq) - 1)) * scale,
              easing: 'ease-in-out',
              fill: 'backwards',
            },
          ),
        ),
      );
    }
    await Promise.all(waits);
    await sleep(70 * scale);
  }

  promptAction(labels: string[]): void {
    const passIndex = labels.indexOf('pass');
    if (passIndex >= 0 && labels.length === 1) {
      this.passEl.classList.add('ot-show');
      this.passEl.onclick = () => {
        this.disableInput();
        this.ctx.submit(String(passIndex));
      };
      return;
    }
    const map = new Map<number, number>();
    labels.forEach((label, i) => {
      const sq = squareIndex(label);
      if (sq !== null) {
        map.set(sq, i);
        this.cells[sq].classList.add('ot-legal');
      }
    });
    this.actionBySq = map;
    this.boardEl.classList.add('ot-live', this.ctx.humanSeat === 1 ? 'ot-human-w' : 'ot-human-b');
  }

  unmount(): void {
    for (const a of this.anims) a.cancel();
    this.anims.clear();
  }

  private rebuildDiscs(view: OtView): void {
    this.discs.clear();
    for (let sq = 0; sq < SIZE * SIZE; sq++) {
      const ch = view.cells[sq];
      if (ch === '.') {
        this.cells[sq].replaceChildren();
        continue;
      }
      const disc = document.createElement('div');
      disc.className = `ot-disc ${ch === 'b' ? 'ot-b' : 'ot-w'}`;
      disc.innerHTML =
        '<div class="ot-flip"><div class="ot-face ot-face-b"></div><div class="ot-face ot-face-w"></div></div>';
      this.cells[sq].replaceChildren(disc);
      this.discs.set(sq, disc);
    }
  }

  private clickSquare(sq: number): void {
    const action = this.actionBySq?.get(sq);
    if (action === undefined) return;
    this.disableInput();
    this.ctx.submit(String(action));
  }

  private disableInput(): void {
    this.actionBySq = null;
    this.passEl.classList.remove('ot-show');
    this.passEl.onclick = null;
    this.boardEl.classList.remove('ot-live', 'ot-human-b', 'ot-human-w');
    for (const cell of this.cells) cell.classList.remove('ot-legal', 'ot-hint');
  }

  private async run(anim: Animation): Promise<void> {
    this.anims.add(anim);
    try {
      await anim.finished;
    } catch {
      /* cancelled on unmount */
    } finally {
      this.anims.delete(anim);
    }
  }
}

export function createOthelloFrontend(): GameFrontend {
  return new OthelloFrontend();
}
