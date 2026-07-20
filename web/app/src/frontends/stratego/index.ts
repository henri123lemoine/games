// Stratego frontend: a campaign-map board with classic rank-badge pieces,
// click-or-drag movement, a serialized deployment flow with a supply tray, and
// the battle face-off reveal (both ranks flip up, the loser sinks).
//
// Consumes the game-private `view_data`/`transition_data` JSON emitted by
// `games/stratego/src/game.rs` (see the schema doc there). Legality comes
// entirely from `promptAction` labels: `"30->40"` absolute-cell moves in the
// move phase, `"S (Spy)"` piece-type placements during deployment.

import type { MatchEventData, ViewState } from '../../engine/protocol';
import type { FrontendCtx, GameFrontend } from '../types';
import { sleep } from '../types';

interface CellPiece {
  o: 0 | 1;
  r: string | null;
  v: boolean;
  m: boolean;
}
type Cell = CellPiece | '~' | null;

interface View {
  phase: 'deploy' | 'play';
  viewer: number;
  toAct: 0 | 1;
  cells: Cell[];
  nextSquare: number | null;
  supply: number[] | null;
  deployed: [number, number] | null;
  lastMove: { from: number; to: number } | null;
  captured: [string[], string[]] | null;
}

interface Transition {
  from: number;
  to: number;
  mover: { o: 0 | 1; r: string | null };
  battle: {
    attacker: string;
    defender: string;
    outcome: 'win' | 'loss' | 'tie';
    flag: boolean;
  } | null;
}

const SLIDE_MS = 260;
const REVEAL_MS = 520;
const SINK_MS = 340;
const RANK_NAMES: Record<string, string> = {
  '10': 'Marshal', '9': 'General', '8': 'Colonel', '7': 'Major',
  '6': 'Captain', '5': 'Lieutenant', '4': 'Sergeant', '3': 'Miner',
  '2': 'Scout', S: 'Spy', B: 'Bomb', F: 'Flag',
};
/** Supply-tray order: strongest first, specials last — the classic rulebook
 * listing players scan when planning a setup. */
const TRAY_ORDER = ['10', '9', '8', '7', '6', '5', '4', '3', '2', 'S', 'B', 'F'];
/** PieceType index (the `supply` array order) per rank glyph. */
const TYPE_INDEX: Record<string, number> = {
  S: 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8,
  '10': 9, F: 10, B: 11,
};
/** Deploy-label arrangement characters (`type_to_char`, an A-M bijection where
 * e.g. 'F' is the Sergeant) → rank glyph. */
const CHAR_TO_GLYPH: Record<string, string> = {
  C: 'S', D: '2', E: '3', F: '4', G: '5', H: '6', I: '7', J: '8', K: '9',
  L: '10', M: 'F', B: 'B',
};

export function createStrategoFrontend(): GameFrontend {
  return new StrategoFrontend();
}

class StrategoFrontend implements GameFrontend {
  private ctx!: FrontendCtx;
  private root!: HTMLElement;
  private squaresEl!: HTMLDivElement;
  private piecesEl!: HTMLDivElement;
  private trayEl!: HTMLDivElement;
  private fallbackEl!: HTMLPreElement;
  private flipped = false;
  private view: View | null = null;
  /** abs from-cell → (abs to-cell → label to submit). */
  private moves = new Map<number, Map<number, string>>();
  /** Deploy labels: rank glyph → label to submit. */
  private placements = new Map<string, string>();
  private selected: number | null = null;
  private myTurn = false;
  private drag: {
    piece: HTMLElement;
    ghost: HTMLElement;
    from: number;
    moved: boolean;
  } | null = null;

  // ---------- mount ----------

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    this.flipped = ctx.humanSeat === 1;
    injectStyle();
    host.innerHTML = `
      <div class="sg-root">
        <div class="sg-bar sg-bar-top">
          <span class="seat-slot" data-seat="${this.flipped ? 0 : 1}"></span>
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
          <span class="seat-slot" data-seat="${this.flipped ? 1 : 0}"></span>
          <div class="sg-tray sg-tray-bottom" title="Captured pieces"></div>
        </div>
        <pre class="sg-fallback" hidden></pre>
      </div>`;
    this.root = host.querySelector('.sg-root')!;
    this.squaresEl = host.querySelector('.sg-squares')!;
    this.piecesEl = host.querySelector('.sg-pieces')!;
    this.trayEl = host.querySelector('.sg-supply')!;
    this.fallbackEl = host.querySelector('.sg-fallback')!;
    this.buildSquares();

    const board = host.querySelector<HTMLElement>('.sg-board')!;
    board.addEventListener('pointerdown', (e) => this.onPointerDown(e));
    board.addEventListener('pointermove', (e) => this.onPointerMove(e));
    board.addEventListener('pointerup', (e) => this.onPointerUp(e));
    board.addEventListener('pointercancel', () => this.cancelDrag());
  }

  private buildSquares(): void {
    const frag = document.createDocumentFragment();
    for (let y = 0; y < 10; y++) {
      for (let x = 0; x < 10; x++) {
        const cell = this.cellAt(x, y);
        const sq = document.createElement('div');
        sq.className = 'sg-sq';
        sq.dataset.cell = String(cell);
        if (LAKES.has(cell)) {
          sq.classList.add('sg-lake');
          sq.innerHTML = WAVES_SVG;
        }
        frag.append(sq);
      }
    }
    this.squaresEl.replaceChildren(frag);
  }

  // ---------- geometry ----------

  /** Screen grid position (x right, y down) → absolute cell. */
  private cellAt(x: number, y: number): number {
    const col = this.flipped ? 9 - x : x;
    const row = this.flipped ? y : 9 - y;
    return row * 10 + col;
  }

  private xyOf(cell: number): { x: number; y: number } {
    const row = Math.floor(cell / 10);
    const col = cell % 10;
    return {
      x: this.flipped ? 9 - col : col,
      y: this.flipped ? row : 9 - row,
    };
  }

  private squareEl(cell: number): HTMLElement | null {
    return this.squaresEl.querySelector(`[data-cell="${cell}"]`);
  }

  private pieceEl(cell: number): HTMLElement | null {
    return this.piecesEl.querySelector(`[data-cell="${cell}"]`);
  }

  // ---------- render ----------

  render(state: ViewState): void {
    const view = parseView(state.viewData);
    if (!view) {
      this.fallbackEl.hidden = false;
      this.fallbackEl.textContent = state.view;
      return;
    }
    this.fallbackEl.hidden = true;
    this.view = view;
    this.syncPieces(view);
    this.syncTrays(view);
    this.syncSupply(view);
    this.syncHighlights(view);
  }

  private syncPieces(view: View): void {
    const frag = document.createDocumentFragment();
    view.cells.forEach((cell, i) => {
      if (cell === null || cell === '~') return;
      frag.append(this.makePiece(i, cell));
    });
    this.piecesEl.replaceChildren(frag);
  }

  private makePiece(cell: number, p: CellPiece): HTMLElement {
    const el = document.createElement('div');
    el.className = `sg-piece sg-${p.o === 0 ? 'red' : 'blue'}`;
    el.dataset.cell = String(cell);
    if (p.r === null) {
      el.classList.add('sg-hidden');
      if (p.m) el.classList.add('sg-has-moved');
      el.title = p.m ? 'Hidden enemy (has moved)' : 'Hidden enemy';
    } else {
      if (p.v && this.view && p.o === this.ctx.humanSeat) {
        el.classList.add('sg-known');
        el.title = `${RANK_NAMES[p.r]} (revealed to the enemy)`;
      } else {
        el.title = RANK_NAMES[p.r] ?? p.r;
      }
    }
    el.innerHTML = badgeSvg(p.r, p.o);
    this.place(el, cell);
    return el;
  }

  private place(el: HTMLElement, cell: number): void {
    const { x, y } = this.xyOf(cell);
    el.style.transform = `translate(${x * 100}%, ${y * 100}%)`;
    el.dataset.cell = String(cell);
  }

  private syncTrays(view: View): void {
    // A side's tray shows the pieces *that side* has lost, beside that side's
    // seat bar: bottom bar = the viewer's seat (or red when spectating).
    const bottomSeat = this.flipped ? 1 : 0;
    const lost = view.captured ?? [[], []];
    this.fillTray('.sg-tray-bottom', lost[bottomSeat], bottomSeat as 0 | 1);
    this.fillTray('.sg-tray-top', lost[1 - bottomSeat], (1 - bottomSeat) as 0 | 1);
  }

  private fillTray(sel: string, ranks: string[], owner: 0 | 1): void {
    const tray = this.root.querySelector<HTMLElement>(sel)!;
    const sorted = [...ranks].sort(
      (a, b) => TRAY_ORDER.indexOf(a) - TRAY_ORDER.indexOf(b),
    );
    tray.replaceChildren(
      ...sorted.map((r) => {
        const s = document.createElement('span');
        s.className = `sg-tray-piece sg-${owner === 0 ? 'red' : 'blue'}`;
        s.title = `${RANK_NAMES[r]} (captured)`;
        s.innerHTML = badgeSvg(r, owner);
        return s;
      }),
    );
  }

  private syncSupply(view: View): void {
    const deploying =
      view.phase === 'deploy' &&
      view.supply !== null &&
      view.toAct === this.ctx.humanSeat;
    this.trayEl.hidden = !deploying;
    this.root.classList.toggle('sg-deploying', deploying);
    if (!deploying || !view.supply) return;

    this.trayEl.replaceChildren(
      ...TRAY_ORDER.map((rank) => {
        const count = view.supply![TYPE_INDEX[rank]];
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `sg-supply-btn sg-${this.ctx.humanSeat === 1 ? 'blue' : 'red'}`;
        btn.disabled = count === 0 || !this.placements.has(rank);
        btn.title = `${RANK_NAMES[rank]} — ${count} left`;
        btn.innerHTML = `${badgeSvg(rank, this.ctx.humanSeat === 1 ? 1 : 0)}<span class="sg-supply-count">${count}</span>`;
        btn.onclick = () => {
          const label = this.placements.get(rank);
          if (label) this.submitOnce(label);
        };
        return btn;
      }),
    );
  }

  private syncHighlights(view: View): void {
    for (const sq of this.squaresEl.children) {
      sq.classList.remove(
        'sg-sq-last-from', 'sg-sq-last-to', 'sg-sq-next', 'sg-sq-selected',
        'sg-sq-target', 'sg-sq-capture', 'sg-sq-movable', 'sg-sq-drop',
      );
    }
    if (view.lastMove) {
      this.squareEl(view.lastMove.from)?.classList.add('sg-sq-last-from');
      this.squareEl(view.lastMove.to)?.classList.add('sg-sq-last-to');
    }
    if (
      view.phase === 'deploy' &&
      view.nextSquare !== null &&
      view.toAct === this.ctx.humanSeat
    ) {
      this.squareEl(view.nextSquare)?.classList.add('sg-sq-next');
    }
    if (this.myTurn && this.view?.phase === 'play') {
      for (const from of this.moves.keys()) {
        this.squareEl(from)?.classList.add('sg-sq-movable');
      }
      if (this.selected !== null) this.showSelection(this.selected);
    }
  }

  private showSelection(from: number): void {
    this.squareEl(from)?.classList.add('sg-sq-selected');
    const dests = this.moves.get(from);
    if (!dests || !this.view) return;
    for (const to of dests.keys()) {
      const cell = this.view.cells[to];
      const isCapture = cell !== null && cell !== '~';
      this.squareEl(to)?.classList.add(isCapture ? 'sg-sq-capture' : 'sg-sq-target');
    }
  }

  // ---------- animation ----------

  async animate(event: MatchEventData, after: ViewState): Promise<void> {
    const t = parseTransition(event.data);
    const scale = this.ctx.animationScale();
    if (!t || scale === 0 || this.view?.phase !== 'play') {
      this.render(after);
      if (this.view?.phase === 'deploy' && scale > 0) await sleep(40 * scale);
      return;
    }

    const mover = this.pieceEl(t.from);
    if (mover) {
      mover.style.transitionDuration = `${SLIDE_MS * scale}ms`;
      mover.style.zIndex = '4';
      this.place(mover, t.to);
      await sleep(SLIDE_MS * scale);
    }
    if (t.battle && mover) {
      const defender = this.pieceEl(t.to);
      // Face-off: both flip up to their true ranks…
      this.reveal(mover, t.battle.attacker, t.mover.o);
      if (defender && defender !== mover) {
        this.reveal(defender, t.battle.defender, (1 - t.mover.o) as 0 | 1);
      }
      await sleep(REVEAL_MS * scale);
      // …then the fallen sink away.
      const sink: HTMLElement[] = [];
      if (t.battle.outcome !== 'win') sink.push(mover);
      if (t.battle.outcome !== 'loss' && defender && defender !== mover) {
        sink.push(defender);
      }
      for (const el of sink) el.classList.add('sg-sinking');
      if (sink.length) await sleep(SINK_MS * scale);
    }
    this.render(after);
  }

  private reveal(el: HTMLElement, rank: string, owner: 0 | 1): void {
    el.classList.add('sg-revealing');
    el.innerHTML = badgeSvg(rank, owner);
  }

  // ---------- input ----------

  promptAction(labels: string[]): void {
    this.moves.clear();
    this.placements.clear();
    for (const label of labels) {
      const move = /^(\d+)->(\d+)$/.exec(label);
      if (move) {
        const from = Number(move[1]);
        const to = Number(move[2]);
        if (!this.moves.has(from)) this.moves.set(from, new Map());
        this.moves.get(from)!.set(to, label);
        continue;
      }
      const place = /^([A-M]) \(/.exec(label);
      if (place) {
        const glyph = CHAR_TO_GLYPH[place[1]];
        if (glyph) this.placements.set(glyph, place[1]);
      }
    }
    this.myTurn = true;
    this.selected = null;
    if (this.view) {
      this.syncSupply(this.view);
      this.syncHighlights(this.view);
    }
  }

  private submitOnce(input: string): void {
    if (!this.myTurn) return;
    this.myTurn = false;
    this.selected = null;
    this.moves.clear();
    this.placements.clear();
    this.trayEl.hidden = true;
    this.root.classList.remove('sg-deploying');
    if (this.view) this.syncHighlights(this.view);
    this.ctx.submit(input);
  }

  private cellFromEvent(e: PointerEvent): number | null {
    const rect = this.squaresEl.getBoundingClientRect();
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * 10);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * 10);
    if (x < 0 || x > 9 || y < 0 || y > 9) return null;
    return this.cellAt(x, y);
  }

  private onPointerDown(e: PointerEvent): void {
    if (!this.myTurn || this.view?.phase !== 'play' || e.button !== 0) return;
    const cell = this.cellFromEvent(e);
    if (cell === null) return;

    if (this.selected !== null) {
      const label = this.moves.get(this.selected)?.get(cell);
      if (label) {
        this.submitOnce(label);
        return;
      }
    }
    if (!this.moves.has(cell)) {
      this.selected = null;
      if (this.view) this.syncHighlights(this.view);
      return;
    }
    this.selected = cell;
    if (this.view) this.syncHighlights(this.view);

    const piece = this.pieceEl(cell);
    if (!piece) return;
    const ghost = piece.cloneNode(true) as HTMLElement;
    ghost.classList.add('sg-ghost');
    this.piecesEl.append(ghost);
    piece.classList.add('sg-drag-src');
    this.drag = { piece, ghost, from: cell, moved: false };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    this.moveGhost(e);
  }

  private moveGhost(e: PointerEvent): void {
    if (!this.drag) return;
    const rect = this.squaresEl.getBoundingClientRect();
    const size = rect.width / 10;
    const gx = e.clientX - rect.left - size / 2;
    const gy = e.clientY - rect.top - size / 2;
    this.drag.ghost.style.transform = `translate(${gx}px, ${gy}px)`;
  }

  private onPointerMove(e: PointerEvent): void {
    if (!this.drag) return;
    this.drag.moved = true;
    this.moveGhost(e);
    const over = this.cellFromEvent(e);
    for (const sq of this.squaresEl.children) sq.classList.remove('sg-sq-drop');
    if (over !== null && this.moves.get(this.drag.from)?.has(over)) {
      this.squareEl(over)?.classList.add('sg-sq-drop');
    }
  }

  private onPointerUp(e: PointerEvent): void {
    if (!this.drag) return;
    const { from, moved } = this.drag;
    const over = this.cellFromEvent(e);
    this.cancelDrag();
    if (!moved || over === null || over === from) return; // click-select flow
    const label = this.moves.get(from)?.get(over);
    if (label) this.submitOnce(label);
  }

  private cancelDrag(): void {
    if (!this.drag) return;
    this.drag.ghost.remove();
    this.drag.piece.classList.remove('sg-drag-src');
    this.drag = null;
    for (const sq of this.squaresEl.children) sq.classList.remove('sg-sq-drop');
  }

  unmount(): void {
    this.cancelDrag();
  }
}

// ---------- parsing ----------

function parseView(data: unknown): View | null {
  if (!data || typeof data !== 'object') return null;
  const v = data as View;
  return Array.isArray(v.cells) && v.cells.length === 100 ? v : null;
}

function parseTransition(data: unknown): Transition | null {
  if (!data || typeof data !== 'object') return null;
  const t = data as Transition;
  return typeof t.from === 'number' && typeof t.to === 'number' ? t : null;
}

// ---------- board constants ----------

const LAKES = new Set([42, 43, 46, 47, 52, 53, 56, 57]);

const WAVES_SVG = `<svg class="sg-wave" viewBox="0 0 40 40" aria-hidden="true">
  <path d="M6 16 q4 -4 8 0 t8 0 t8 0" />
  <path d="M10 26 q4 -4 8 0 t8 0" />
</svg>`;

// ---------- piece sprites ----------

/** A classic standing piece: the colored stand, an inner face, and the rank —
 * a numeral roundel, or the special's own emblem (flag, bomb, spy). A `null`
 * rank draws the hidden back: crosshatch and a star. */
function badgeSvg(rank: string | null, owner: 0 | 1): string {
  const side = owner === 0 ? 'r' : 'b';
  const face =
    rank === null ? hiddenFace(side) : FACE_BUILDERS[rank]?.(side) ?? textFace(rank);
  return `<svg class="sg-badge" viewBox="0 0 100 100" aria-hidden="true">
    <rect class="sg-stand" x="8" y="6" width="84" height="88" rx="14" />
    <rect class="sg-stand-lip" x="8" y="6" width="84" height="88" rx="14" />
    ${face}
  </svg>`;
}

function hiddenFace(side: 'r' | 'b'): string {
  return `<g clip-path="url(#sg-clip)">
    <pattern id="sg-hatch-${side}" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <rect width="9" height="9" class="sg-hatch-bg" />
      <line x1="0" y1="0" x2="0" y2="9" class="sg-hatch-line" />
    </pattern>
    <rect x="14" y="12" width="72" height="76" rx="9" fill="url(#sg-hatch-${side})" />
  </g>
  <path class="sg-star" d="M50 30 l5.9 12 13.2 1.9 -9.5 9.3 2.2 13.1 -11.8 -6.2 -11.8 6.2 2.2 -13.1 -9.5 -9.3 13.2 -1.9 z" />
  <clipPath id="sg-clip"><rect x="14" y="12" width="72" height="76" rx="9" /></clipPath>`;
}

function textFace(rank: string): string {
  return `<rect class="sg-face" x="14" y="12" width="72" height="76" rx="9" />
    <text class="sg-rank" x="50" y="64" text-anchor="middle">${rank}</text>`;
}

const FACE_BUILDERS: Record<string, (side: 'r' | 'b') => string> = {
  F: () => `<rect class="sg-face" x="14" y="12" width="72" height="76" rx="9" />
    <line class="sg-pole" x1="38" y1="24" x2="38" y2="78" />
    <path class="sg-flag" d="M40 26 h26 l-7 9 7 9 h-26 z" />`,
  B: () => `<rect class="sg-face" x="14" y="12" width="72" height="76" rx="9" />
    <circle class="sg-bomb" cx="50" cy="58" r="18" />
    <path class="sg-fuse" d="M58 44 q4 -12 14 -14" />
    <path class="sg-spark" d="M72 30 l3 -6 M72 30 l6 -2 M72 30 l6 4" />`,
  S: () => `<rect class="sg-face" x="14" y="12" width="72" height="76" rx="9" />
    <ellipse class="sg-eye" cx="50" cy="50" rx="22" ry="13" />
    <circle class="sg-pupil" cx="50" cy="50" r="6.5" />`,
  '2': () => `<rect class="sg-face" x="14" y="12" width="72" height="76" rx="9" />
    <text class="sg-rank" x="50" y="62" text-anchor="middle">2</text>
    <path class="sg-arrow" d="M26 78 h48 m-8 -6 8 6 -8 6" />`,
  '3': () => `<rect class="sg-face" x="14" y="12" width="72" height="76" rx="9" />
    <text class="sg-rank" x="50" y="62" text-anchor="middle">3</text>
    <path class="sg-pick" d="M28 80 l14 -8 m-14 8 l8 -14" />`,
};

// ---------- styles ----------

const STYLE_ID = 'stratego-frontend-style';

function injectStyle(): void {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = CSS_TEXT;
  document.head.append(style);
}

const CSS_TEXT = `
.sg-root {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: min(100%, var(--board-fit));
  margin: 0 auto;
  --sg-field: #dbe4cf;
  --sg-grid: #b7c6a4;
  --sg-lake: #9fc4dc;
  --sg-lake-wave: #6d9cbc;
  --sg-red: #b23a34;
  --sg-red-deep: #7e211d;
  --sg-red-face: #c9524b;
  --sg-blue: #33589e;
  --sg-blue-deep: #1e3a72;
  --sg-blue-face: #4a6fb5;
  --sg-cream: #f5efdd;
}
.dark .sg-root {
  --sg-field: #2c3529;
  --sg-grid: #414f3b;
  --sg-lake: #24405a;
  --sg-lake-wave: #45719a;
  --sg-red: #a03a35;
  --sg-red-deep: #571713;
  --sg-red-face: #b44e47;
  --sg-blue: #3a5da0;
  --sg-blue-deep: #16294f;
  --sg-blue-face: #4a6fb5;
  --sg-cream: #e8e2d0;
}

.sg-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 34px;
}
.sg-tray {
  display: flex;
  flex-wrap: wrap;
  gap: 2px;
  margin-left: auto;
  max-width: 70%;
}
.sg-tray-piece {
  width: 20px;
  height: 20px;
  opacity: 0.85;
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
.sg-squares {
  position: absolute;
  inset: 0;
  display: grid;
  grid-template: repeat(10, 1fr) / repeat(10, 1fr);
}
.sg-sq {
  background: var(--sg-field);
  box-shadow: inset 0 0 0 0.5px var(--sg-grid);
  position: relative;
}
.sg-lake { background: var(--sg-lake); }
.sg-wave {
  position: absolute;
  inset: 12%;
  width: 76%;
  height: 76%;
}
.sg-wave path {
  fill: none;
  stroke: var(--sg-lake-wave);
  stroke-width: 2.4;
  stroke-linecap: round;
}

/* --- square states --- */
.sg-sq-last-from, .sg-sq-last-to { background: color-mix(in srgb, var(--accent) 22%, var(--sg-field)); }
.sg-sq-next {
  animation: sg-pulse 1.2s ease-in-out infinite;
  box-shadow: inset 0 0 0 3px var(--accent);
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
  width: 26%;
  height: 26%;
  translate: -50% -50%;
  border-radius: 50%;
  background: color-mix(in srgb, var(--accent) 55%, transparent);
}
.sg-sq-capture { box-shadow: inset 0 0 0 3px color-mix(in srgb, var(--bad, #d33) 75%, var(--accent)); }
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
   * 0.7% of the board is 7% of this square. */
  padding: 0.7%;
  box-sizing: border-box;
  transition: transform 0.24s cubic-bezier(0.2, 0.8, 0.3, 1);
  will-change: transform;
}
.sg-piece svg { display: block; width: 100%; height: 100%; filter: drop-shadow(0 1px 1.5px rgba(0,0,0,0.35)); }
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
  right: 12%;
  bottom: 10%;
  width: 12%;
  height: 12%;
  border-radius: 50%;
  background: var(--sg-cream);
  outline: 1px solid rgba(0,0,0,0.4);
}
.sg-known .sg-stand-lip { stroke: var(--accent-2, #d4a72c); stroke-width: 5; }

/* --- badge colors --- */
.sg-red .sg-stand { fill: var(--sg-red-deep); }
.sg-red .sg-face, .sg-red .sg-hatch-bg { fill: var(--sg-red); }
.sg-red .sg-hatch-line { stroke: var(--sg-red-deep); stroke-width: 2.4; }
.sg-blue .sg-stand { fill: var(--sg-blue-deep); }
.sg-blue .sg-face, .sg-blue .sg-hatch-bg { fill: var(--sg-blue); }
.sg-blue .sg-hatch-line { stroke: var(--sg-blue-deep); stroke-width: 2.4; }
.sg-stand-lip { fill: none; stroke: rgba(255,255,255,0.18); stroke-width: 2.5; }
.sg-red .sg-face { fill: var(--sg-red-face); }
.sg-blue .sg-face { fill: var(--sg-blue-face); }
.sg-rank {
  font: 700 44px var(--mono, ui-monospace, monospace);
  fill: var(--sg-cream);
  paint-order: stroke;
  stroke: rgba(0,0,0,0.25);
  stroke-width: 2;
}
.sg-star { fill: var(--sg-cream); opacity: 0.9; }
.sg-pole { stroke: var(--sg-cream); stroke-width: 5; stroke-linecap: round; }
.sg-flag { fill: var(--sg-cream); }
.sg-bomb { fill: #1c1c1e; stroke: var(--sg-cream); stroke-width: 2.5; }
.sg-fuse, .sg-spark { fill: none; stroke: var(--sg-cream); stroke-width: 4; stroke-linecap: round; }
.sg-spark { stroke: #f2b03d; stroke-width: 3; }
.sg-eye { fill: var(--sg-cream); }
.sg-pupil { fill: #1c1c1e; }
.sg-arrow, .sg-pick { fill: none; stroke: var(--sg-cream); stroke-width: 4.5; stroke-linecap: round; stroke-linejoin: round; }
.sg-arrow { stroke-width: 3.5; }

/* --- supply tray (deployment) --- */
.sg-supply[hidden] { display: none; }
.sg-supply {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 6px;
  padding: 10px;
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  width: 128px;
  flex-shrink: 0;
}
.sg-supply-btn {
  position: relative;
  aspect-ratio: 1;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg-inset);
  cursor: pointer;
  padding: 6px;
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
`;
