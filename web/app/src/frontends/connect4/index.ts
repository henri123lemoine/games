// Connect Four frontend: a blue arcade rig with punched holes, gravity-fed
// disc drops (discs fall behind the frame and show through the holes), a
// hover ghost over the target column, and a pulsing win-line glow.
//
// View schema (games/connect4/src/ui.rs::view_data):
//   { cells: string,             // 42 chars, row-major, TOP row first:
//                                //   '.' empty, 'x' seat 0, 'o' seat 1
//     turn: 0 | 1,               // side to move
//     winner: 0 | 1 | null,
//     winLine: number[] | null } // 4 indices into `cells`
//
// Transition schema (ui.rs::transition_data):
//   { col: 0-6, row: 0-5, player: 0 | 1 }   // landing cell, row 0 = BOTTOM

import type { MatchEventData, ViewState } from '../../engine/protocol';
import type { FrontendCtx, GameFrontend } from '../types';
import { sleep } from '../types';

const COLS = 7;
const ROWS = 6;
const COLOR_NAMES = ['Red', 'Yellow'];

interface C4View {
  cells: string;
  turn: number;
  winner: number | null;
  winLine: number[] | null;
}

interface C4Drop {
  col: number;
  row: number;
  player: number;
}

function parseView(data: unknown): C4View | null {
  if (!data || typeof data !== 'object') return null;
  const v = data as C4View;
  return typeof v.cells === 'string' && v.cells.length === COLS * ROWS ? v : null;
}

function parseDrop(data: unknown): C4Drop | null {
  if (!data || typeof data !== 'object') return null;
  const d = data as C4Drop;
  return typeof d.col === 'number' && typeof d.row === 'number' && typeof d.player === 'number'
    ? d
    : null;
}

/** Index into the cells string for (col, row-from-bottom). */
function cellIndex(col: number, row: number): number {
  return (ROWS - 1 - row) * COLS + col;
}

/** Recover the drop from a state diff when the event carries no data. */
function diffDrop(prev: C4View | null, after: C4View | null): C4Drop | null {
  if (!prev || !after) return null;
  for (let i = 0; i < COLS * ROWS; i++) {
    if (prev.cells[i] === '.' && after.cells[i] !== '.') {
      return {
        col: i % COLS,
        row: ROWS - 1 - Math.floor(i / COLS),
        player: after.cells[i] === 'x' ? 0 : 1,
      };
    }
  }
  return null;
}

const STYLE_ID = 'connect4-frontend-style';

const CSS = `
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
  aspect-ratio: ${COLS} / ${ROWS};
  border-radius: calc(var(--radius) + 4px);
  overflow: hidden;
  background: #0b1020;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45), 0 0 0 2px rgba(10, 24, 64, 0.9);
}
.c4-layer {
  position: absolute;
  inset: 0;
}
.c4-backdrop {
  background:
    radial-gradient(circle closest-side at 50% 44%, #131a26 0 70%, #060910 100%)
    0 0 / calc(100% / ${COLS}) calc(100% / ${ROWS});
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
    0 0 / calc(100% / ${COLS}) calc(100% / ${ROWS});
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
  width: calc(100% / ${COLS});
  height: calc(100% / ${ROWS});
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
`;

function injectStyle(): void {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.append(style);
}

class Connect4Frontend implements GameFrontend {
  private ctx!: FrontendCtx;
  private rootEl!: HTMLElement;
  private discsEl!: HTMLElement;
  private hitsEl!: HTMLElement;
  private msgEl!: HTMLElement;
  private fallbackEl!: HTMLElement;
  private chips: HTMLElement[] = [];
  private discs = new Map<number, HTMLElement>();
  private view: C4View | null = null;
  private colToAction: Map<number, number> | null = null;
  private ghost: HTMLElement | null = null;
  private anims = new Set<Animation>();

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    injectStyle();
    host.innerHTML = `
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
      </div>`;
    this.rootEl = host.querySelector('.c4-root')!;
    this.discsEl = host.querySelector('.c4-discs')!;
    this.hitsEl = host.querySelector('.c4-hits')!;
    this.msgEl = host.querySelector('.c4-msg')!;
    this.fallbackEl = host.querySelector('.c4-fallback')!;
    this.chips = [host.querySelector('.c4-chip-0')!, host.querySelector('.c4-chip-1')!];
    for (let seat = 0; seat < 2; seat++) {
      const who = ctx.humanSeat === seat ? 'You' : 'Bot';
      this.chips[seat].lastElementChild!.textContent = `${COLOR_NAMES[seat]} · ${who}`;
    }
    for (let col = 0; col < COLS; col++) {
      const hit = document.createElement('div');
      hit.className = 'c4-hit';
      hit.addEventListener('pointerenter', () => this.showGhost(col));
      hit.addEventListener('pointerleave', () => this.hideGhost());
      hit.addEventListener('click', () => this.clickColumn(col));
      this.hitsEl.append(hit);
    }
  }

  render(state: ViewState): void {
    this.disableInput();
    const view = parseView(state.viewData);
    this.view = view;
    if (!view) {
      this.rootEl.classList.add('c4-text-only');
      this.fallbackEl.textContent = state.view;
      return;
    }
    this.rootEl.classList.remove('c4-text-only');
    this.rebuildDiscs(view);
    this.decorateWin(view, true);
    for (let seat = 0; seat < 2; seat++) {
      this.chips[seat].classList.toggle('c4-active', !state.isOver && view.turn === seat);
    }
    this.msgEl.textContent = state.isOver
      ? view.winner !== null
        ? `${COLOR_NAMES[view.winner]} connects four!`
        : 'Draw — board full'
      : `${COLOR_NAMES[view.turn]} to move`;
  }

  async animate(event: MatchEventData, after: ViewState): Promise<void> {
    const prev = this.view;
    const view = parseView(after.viewData);
    const drop = parseDrop(event.data) ?? diffDrop(prev, view);
    this.render(after);
    const scale = this.ctx.animationScale();
    if (!view || !drop || scale <= 0) return;
    const disc = this.discs.get(cellIndex(drop.col, drop.row));
    if (!disc) return;

    const hasWin = view.winLine !== null;
    if (hasWin) this.decorateWin(view, false);

    const travel = ROWS - drop.row;
    const fallMs = (150 + 100 * Math.sqrt(travel)) * scale;
    await this.run(
      disc.animate(
        [
          {
            transform: `translateY(${-travel * 100 - 30}%)`,
            offset: 0,
            easing: 'cubic-bezier(0.5, 0, 0.9, 0.6)',
          },
          { transform: 'translateY(0%)', offset: 0.62, easing: 'cubic-bezier(0.1, 0.5, 0.5, 1)' },
          { transform: 'translateY(-17%)', offset: 0.8, easing: 'cubic-bezier(0.5, 0, 0.9, 0.6)' },
          { transform: 'translateY(0%)', offset: 1 },
        ],
        { duration: fallMs / 0.62 },
      ),
    );
    if (hasWin) {
      this.decorateWin(view, true);
      await sleep(650 * scale);
    }
  }

  promptAction(labels: string[]): void {
    const map = new Map<number, number>();
    labels.forEach((label, i) => {
      const m = /(\d+)/.exec(label);
      if (m) map.set(Number(m[1]) - 1, i);
    });
    this.colToAction = map;
    this.hitsEl.classList.add('c4-live');
  }

  unmount(): void {
    for (const a of this.anims) a.cancel();
    this.anims.clear();
  }

  private rebuildDiscs(view: C4View): void {
    this.discsEl.replaceChildren();
    this.discs.clear();
    this.ghost = null;
    for (let i = 0; i < COLS * ROWS; i++) {
      const ch = view.cells[i];
      if (ch === '.') continue;
      const disc = this.makeDisc(ch === 'x' ? 0 : 1, i % COLS, Math.floor(i / COLS));
      this.discs.set(i, disc);
      this.discsEl.append(disc);
    }
  }

  private makeDisc(player: number, col: number, rowFromTop: number): HTMLElement {
    const disc = document.createElement('div');
    disc.className = `c4-disc c4-p${player}`;
    disc.style.left = `${(col * 100) / COLS}%`;
    disc.style.top = `${(rowFromTop * 100) / ROWS}%`;
    return disc;
  }

  private decorateWin(view: C4View, on: boolean): void {
    if (!view.winLine) return;
    const winning = new Set(view.winLine);
    for (const [idx, disc] of this.discs) {
      disc.classList.toggle('c4-win', on && winning.has(idx));
      disc.classList.toggle('c4-dim', on && !winning.has(idx));
    }
  }

  private showGhost(col: number): void {
    this.hideGhost();
    const view = this.view;
    if (!view || !this.colToAction?.has(col) || this.ctx.humanSeat < 0) return;
    for (let row = 0; row < ROWS; row++) {
      const idx = cellIndex(col, row);
      if (view.cells[idx] === '.') {
        this.ghost = this.makeDisc(this.ctx.humanSeat, col, Math.floor(idx / COLS));
        this.ghost.classList.add('c4-ghost');
        this.discsEl.append(this.ghost);
        return;
      }
    }
  }

  private hideGhost(): void {
    this.ghost?.remove();
    this.ghost = null;
  }

  private clickColumn(col: number): void {
    const action = this.colToAction?.get(col);
    if (action === undefined) return;
    this.disableInput();
    this.ctx.submit(String(action));
  }

  private disableInput(): void {
    this.colToAction = null;
    this.hideGhost();
    this.hitsEl.classList.remove('c4-live');
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

export function createConnect4Frontend(): GameFrontend {
  return new Connect4Frontend();
}
