// Kamisado frontend: the full-color lacquered board, octagonal dragon towers,
// click-to-slide input, an obligation pulse that hops along blocked towers,
// and a far-rank win glow. The board's colors are 180°-symmetric, so the
// White seat sees it rotated with its home rank at the bottom and the tiles
// unchanged.
//
// View schema (games/kamisado/src/ui.rs::view_data):
//   { cells: string,               // 64 chars, row-major, TOP rank (8) first:
//                                  //   '.' empty, 'A'-'H' Black tower color 0-7,
//                                  //   'a'-'h' White tower color 0-7
//     turn: 0 | 1,
//     required: number | null,     // color `turn` must move
//     requiredCell: number | null, // its index in `cells`
//     winner: 0 | 1 | null,
//     deadlock: boolean }
//
// Transition schema (ui.rs::transition_data):
//   { from, to: number,            // cell indices
//     player: 0 | 1, color: number, landColor: number,
//     passes: [player, color][],   // blocked towers, in obligation-chain order
//     next: [player, color] | null,
//     win: boolean, deadlock: boolean }

import type { MatchEventData, ViewState } from '../../engine/protocol';
import type { FrontendCtx, GameFrontend } from '../types';
import { sleep } from '../types';

const COLOR_NAMES = ['Brown', 'Green', 'Red', 'Yellow', 'Pink', 'Purple', 'Blue', 'Orange'];
const TILE = ['#8a5a33', '#3f9d42', '#d23c32', '#e8c33c', '#e77fb0', '#8050c0', '#3a6fd8', '#e8862f'];
const CORE = ['#a4713f', '#4dbb50', '#ec554a', '#f4d34f', '#f394c2', '#9668d6', '#5288e8', '#f89a44'];

// The official grid, same cell order as the view's `cells` (rank 8 first).
// Single source: games/kamisado/src/lib.rs::BOARD_COLOR, top rank first.
// prettier-ignore
const BOARD: number[] = [
  7, 6, 5, 4, 3, 2, 1, 0,
  2, 7, 4, 1, 6, 3, 0, 5,
  1, 4, 7, 2, 5, 0, 3, 6,
  4, 5, 6, 7, 0, 1, 2, 3,
  3, 2, 1, 0, 7, 6, 5, 4,
  6, 3, 0, 5, 2, 7, 4, 1,
  5, 0, 3, 6, 1, 4, 7, 2,
  0, 1, 2, 3, 4, 5, 6, 7,
];

interface KView {
  cells: string;
  turn: number;
  required: number | null;
  requiredCell: number | null;
  winner: number | null;
  deadlock: boolean;
}

interface KSlide {
  from: number;
  to: number;
  player: number;
  color: number;
  landColor: number;
  passes: [number, number][];
  next: [number, number] | null;
  win: boolean;
  deadlock: boolean;
}

function parseView(data: unknown): KView | null {
  if (!data || typeof data !== 'object') return null;
  const v = data as KView;
  return typeof v.cells === 'string' && v.cells.length === 64 ? v : null;
}

function parseSlide(data: unknown): KSlide | null {
  if (!data || typeof data !== 'object') return null;
  const s = data as KSlide;
  return typeof s.from === 'number' && typeof s.to === 'number' ? s : null;
}

/** "d1" → index into `cells` (rank 8 is row 0). */
function nameToCell(name: string): number | null {
  const file = name.charCodeAt(0) - 97;
  const rank = name.charCodeAt(1) - 49;
  if (file < 0 || file > 7 || rank < 0 || rank > 7) return null;
  return (7 - rank) * 8 + file;
}

/** The tower of (player, color) in a view, if on the board (it always is). */
function towerCell(view: KView, player: number, color: number): number {
  return view.cells.indexOf(String.fromCharCode((player === 0 ? 65 : 97) + color));
}

const STYLE_ID = 'kamisado-frontend-style';

const CSS = `
.km-root {
  align-self: center;
  width: min(100%, var(--board-fit));
  display: flex;
  flex-direction: column;
  gap: 14px;
  user-select: none;
}
.km-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px 10px;
}
.km-chip {
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
.km-chip.km-active {
  border-color: var(--accent);
  color: var(--text);
  box-shadow: 0 0 12px rgba(88, 166, 255, 0.3);
}
.km-swatch {
  width: 14px;
  height: 14px;
  border-radius: 3px;
  flex: none;
  clip-path: polygon(30% 0, 70% 0, 100% 30%, 100% 70%, 70% 100%, 30% 100%, 0 70%, 0 30%);
}
.km-chip-0 .km-swatch { background: linear-gradient(145deg, #3a3a42, #101014); }
.km-chip-1 .km-swatch { background: linear-gradient(145deg, #fbf6ea, #d9d0bc); }
.km-msg {
  flex: 1;
  text-align: center;
  color: var(--text-dim);
  font-size: 0.92rem;
  min-width: 0;
}
.km-msg b { color: var(--text); font-weight: 600; }
.km-board {
  position: relative;
  aspect-ratio: 1;
  border-radius: calc(var(--radius) + 4px);
  overflow: hidden;
  padding: 1.2%;
  background: #14100c;
  box-shadow: 0 10px 26px rgba(0, 0, 0, 0.35), inset 0 0 0 2px rgba(255, 255, 255, 0.04);
}
.km-grid {
  position: relative;
  width: 100%;
  height: 100%;
  display: grid;
  grid-template: repeat(8, 1fr) / repeat(8, 1fr);
  gap: 1%;
}
.km-tile {
  position: relative;
  border-radius: 8%;
  box-shadow: inset 0 2px 6px rgba(255, 255, 255, 0.22), inset 0 -3px 7px rgba(0, 0, 0, 0.3);
}
.km-tile.km-target { cursor: pointer; }
.km-tile.km-target::after {
  content: '';
  position: absolute;
  inset: 32%;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.85);
  box-shadow: 0 0 10px rgba(255, 255, 255, 0.8), inset 0 0 4px rgba(0, 0, 0, 0.25);
  opacity: 0.85;
  transition: transform 0.12s;
}
.km-tile.km-target:hover::after { transform: scale(1.25); }
.km-towers {
  position: absolute;
  inset: 0;
  pointer-events: none;
}
.km-tower {
  position: absolute;
  width: 12.5%;
  height: 12.5%;
  pointer-events: auto;
  will-change: transform;
}
.km-tower .km-oct {
  position: absolute;
  inset: 12%;
  clip-path: polygon(30% 0, 70% 0, 100% 30%, 100% 70%, 70% 100%, 30% 100%, 0 70%, 0 30%);
  transition: filter 0.3s;
}
.km-tower.km-p0 .km-oct {
  background: linear-gradient(145deg, #46464f 0%, #232329 45%, #0c0c10 100%);
  box-shadow: inset 0 3px 5px rgba(255, 255, 255, 0.18);
}
.km-tower.km-p1 .km-oct {
  background: linear-gradient(145deg, #fffdf5 0%, #ece4d2 45%, #c9bfa8 100%);
  box-shadow: inset 0 3px 5px rgba(255, 255, 255, 0.6), inset 0 -3px 6px rgba(0, 0, 0, 0.18);
}
.km-tower .km-core {
  position: absolute;
  inset: 30%;
  border-radius: 50%;
  box-shadow: inset 0 -2px 4px rgba(0, 0, 0, 0.35), inset 0 2px 4px rgba(255, 255, 255, 0.45);
}
.km-tower.km-p0 .km-core { outline: 2px solid rgba(255, 255, 255, 0.25); }
.km-tower.km-p1 .km-core { outline: 2px solid rgba(0, 0, 0, 0.2); }
.km-tower.km-pickable { cursor: pointer; }
.km-tower.km-pickable .km-oct { filter: drop-shadow(0 0 6px rgba(255, 255, 255, 0.45)); }
.km-tower.km-selected .km-oct { filter: drop-shadow(0 0 9px rgba(255, 255, 255, 0.9)); }
.km-tower.km-obliged::before {
  content: '';
  position: absolute;
  inset: 2%;
  border-radius: 30%;
  border: 3px solid rgba(255, 255, 255, 0.75);
  animation: km-obliged 1.5s ease-in-out infinite;
}
@keyframes km-obliged {
  0%, 100% { opacity: 0.25; transform: scale(0.96); }
  50% { opacity: 0.9; transform: scale(1.04); }
}
.km-tower.km-blocked .km-oct { animation: km-blocked 0.5s ease-in-out; }
@keyframes km-blocked {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-6%); }
  75% { transform: translateX(6%); }
}
.km-tower.km-winner .km-oct {
  animation: km-win 1.1s ease-in-out infinite;
}
@keyframes km-win {
  0%, 100% { filter: drop-shadow(0 0 4px rgba(255, 255, 255, 0.5)); }
  50% { filter: drop-shadow(0 0 16px rgba(255, 255, 255, 1)) brightness(1.25); }
}
.km-board.km-deadlock { animation: km-shake 0.45s ease-in-out; }
@keyframes km-shake {
  0%, 100% { transform: translateX(0); }
  20%, 60% { transform: translateX(-1.2%); }
  40%, 80% { transform: translateX(1.2%); }
}
@media (prefers-reduced-motion: reduce) {
  .km-tower.km-obliged::before, .km-tower.km-winner .km-oct { animation: none; }
  .km-board.km-deadlock { animation: none; }
}
.km-fallback {
  display: none;
  margin: 0;
  font-family: ui-monospace, monospace;
  color: var(--text);
  white-space: pre;
}
.km-root.km-text-only .km-bar, .km-root.km-text-only .km-board { display: none; }
.km-root.km-text-only .km-fallback { display: block; }
`;

function injectStyle(): void {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.append(style);
}

class KamisadoFrontend implements GameFrontend {
  private ctx!: FrontendCtx;
  private rootEl!: HTMLElement;
  private boardEl!: HTMLElement;
  private gridEl!: HTMLElement;
  private towersEl!: HTMLElement;
  private msgEl!: HTMLElement;
  private fallbackEl!: HTMLElement;
  private chips: HTMLElement[] = [];
  private tiles: HTMLElement[] = [];
  private towers = new Map<number, HTMLElement>();
  private view: KView | null = null;
  /** Board rotated 180° so the human's home rank sits at the bottom. */
  private flipped = false;
  /** cell of a movable tower → (destination cell → action index). */
  private moves: Map<number, Map<number, number>> | null = null;
  private selected: number | null = null;
  private anims = new Set<Animation>();

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    this.flipped = ctx.humanSeat === 1;
    injectStyle();
    host.innerHTML = `
      <div class="km-root">
        <div class="km-bar">
          <div class="km-chip km-chip-0"><span class="km-swatch"></span><span>Black</span><span class="seat-slot" data-seat="0"></span></div>
          <div class="km-msg"></div>
          <div class="km-chip km-chip-1"><span class="km-swatch"></span><span>White</span><span class="seat-slot" data-seat="1"></span></div>
        </div>
        <div class="km-board">
          <div class="km-grid"></div>
          <div class="km-towers"></div>
        </div>
        <pre class="km-fallback"></pre>
      </div>`;
    this.rootEl = host.querySelector('.km-root')!;
    this.boardEl = host.querySelector('.km-board')!;
    this.gridEl = host.querySelector('.km-grid')!;
    this.towersEl = host.querySelector('.km-towers')!;
    this.msgEl = host.querySelector('.km-msg')!;
    this.fallbackEl = host.querySelector('.km-fallback')!;
    this.chips = [host.querySelector('.km-chip-0')!, host.querySelector('.km-chip-1')!];
    for (let screen = 0; screen < 64; screen++) {
      const cell = this.cellAt(screen);
      const tile = document.createElement('div');
      tile.className = 'km-tile';
      tile.style.background = TILE[BOARD[cell]];
      tile.addEventListener('click', () => this.clickCell(cell));
      this.gridEl.append(tile);
      this.tiles[cell] = tile;
    }
  }

  render(state: ViewState): void {
    this.disableInput();
    const view = parseView(state.viewData);
    this.view = view;
    if (!view) {
      this.rootEl.classList.add('km-text-only');
      this.fallbackEl.textContent = state.view;
      return;
    }
    this.rootEl.classList.remove('km-text-only');
    this.rebuildTowers(view);
    for (let seat = 0; seat < 2; seat++) {
      this.chips[seat].classList.toggle('km-active', !state.isOver && view.turn === seat);
    }
    this.msgEl.innerHTML = this.status(view, state);
  }

  async animate(event: MatchEventData, after: ViewState): Promise<void> {
    const slide = parseSlide(event.data);
    this.render(after);
    const view = this.view;
    const scale = this.ctx.animationScale();
    if (!view || !slide || scale <= 0) return;
    const tower = this.towers.get(slide.to);
    if (tower) {
      const [fc, fr] = this.screenColRow(slide.from);
      const [tc, tr] = this.screenColRow(slide.to);
      const dist = Math.max(Math.abs(fc - tc), Math.abs(fr - tr));
      await this.run(
        tower.animate(
          [
            { transform: `translate(${(fc - tc) * 100}%, ${(fr - tr) * 100}%)` },
            { transform: 'translate(0, 0)' },
          ],
          { duration: (120 + 60 * dist) * scale, easing: 'cubic-bezier(0.2, 0.7, 0.3, 1)' },
        ),
      );
    }
    for (const [p, c] of slide.passes ?? []) {
      const blocked = this.towers.get(towerCell(view, p, c));
      if (!blocked) continue;
      blocked.classList.add('km-blocked');
      await sleep(320 * scale);
      blocked.classList.remove('km-blocked');
    }
    if (slide.win && tower) {
      await sleep(500 * scale);
    } else if (slide.deadlock) {
      this.boardEl.classList.add('km-deadlock');
      await sleep(500 * scale);
      this.boardEl.classList.remove('km-deadlock');
    }
  }

  promptAction(labels: string[]): void {
    const moves = new Map<number, Map<number, number>>();
    labels.forEach((label, i) => {
      const m = /^([a-h][1-8])-([a-h][1-8])$/.exec(label);
      if (!m) return;
      const from = nameToCell(m[1]);
      const to = nameToCell(m[2]);
      if (from === null || to === null) return;
      if (!moves.has(from)) moves.set(from, new Map());
      moves.get(from)!.set(to, i);
    });
    this.moves = moves;
    for (const cell of moves.keys()) this.towers.get(cell)?.classList.add('km-pickable');
    // The obligated tower is the only choice on every move but the first —
    // select it so its destinations light up immediately.
    if (moves.size === 1) this.select([...moves.keys()][0]);
  }

  unmount(): void {
    for (const a of this.anims) a.cancel();
    this.anims.clear();
  }

  /** cells index rendered at this screen grid position (row-major). */
  private cellAt(screen: number): number {
    return this.flipped ? 63 - screen : screen;
  }

  private screenColRow(cell: number): [number, number] {
    const s = this.flipped ? 63 - cell : cell;
    return [s % 8, Math.floor(s / 8)];
  }

  private rebuildTowers(view: KView): void {
    this.towersEl.replaceChildren();
    this.towers.clear();
    for (let cell = 0; cell < 64; cell++) {
      const ch = view.cells[cell];
      if (ch === '.') continue;
      const player = ch >= 'a' ? 1 : 0;
      const color = ch.toLowerCase().charCodeAt(0) - 97;
      const tower = document.createElement('div');
      tower.className = `km-tower km-p${player}`;
      const [col, row] = this.screenColRow(cell);
      tower.style.left = `${col * 12.5}%`;
      tower.style.top = `${row * 12.5}%`;
      tower.innerHTML = `<div class="km-oct"></div><div class="km-core"></div>`;
      (tower.querySelector('.km-core') as HTMLElement).style.background = CORE[color];
      tower.title = `${player === 0 ? 'Black' : 'White'} ${COLOR_NAMES[color]}`;
      tower.addEventListener('click', () => this.clickCell(cell));
      if (view.winner !== null) {
        if (player === view.winner && !view.deadlock && this.onFarRank(cell, player)) {
          tower.classList.add('km-winner');
        }
      } else if (cell === view.requiredCell) {
        tower.classList.add('km-obliged');
      }
      this.towers.set(cell, tower);
      this.towersEl.append(tower);
    }
  }

  /** Whether `cell` is on `player`'s goal rank (rank 8 for Black, 1 for White). */
  private onFarRank(cell: number, player: number): boolean {
    const row = Math.floor(cell / 8);
    return player === 0 ? row === 0 : row === 7;
  }

  private status(view: KView, state: ViewState): string {
    const name = (p: number) => (p === 0 ? 'Black' : 'White');
    if (state.isOver && view.winner !== null) {
      return view.deadlock
        ? `Deadlock — <b>${name(1 - view.winner)}</b> caused it. <b>${name(view.winner)}</b> wins!`
        : `<b>${name(view.winner)}</b> reaches the far rank!`;
    }
    if (view.required === null) return `<b>${name(view.turn)}</b> opens — any tower`;
    return `<b>${name(view.turn)}</b> must move <b>${COLOR_NAMES[view.required]}</b>`;
  }

  private select(cell: number): void {
    this.clearTargets();
    this.selected = cell;
    this.towers.get(cell)?.classList.add('km-selected');
    for (const to of this.moves?.get(cell)?.keys() ?? []) {
      this.tiles[to].classList.add('km-target');
    }
  }

  private clearTargets(): void {
    if (this.selected !== null) this.towers.get(this.selected)?.classList.remove('km-selected');
    this.selected = null;
    for (const tile of this.tiles) tile.classList.remove('km-target');
  }

  private clickCell(cell: number): void {
    if (!this.moves) return;
    if (this.moves.has(cell)) {
      this.select(cell);
      return;
    }
    if (this.selected !== null) {
      const action = this.moves.get(this.selected)?.get(cell);
      if (action !== undefined) {
        this.disableInput();
        this.ctx.submit(String(action));
        return;
      }
    }
    if (this.moves.size > 1) this.clearTargets();
  }

  private disableInput(): void {
    this.clearTargets();
    for (const tower of this.towers.values()) tower.classList.remove('km-pickable');
    this.moves = null;
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

export function createKamisadoFrontend(): GameFrontend {
  return new KamisadoFrontend();
}
