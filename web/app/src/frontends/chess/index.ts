// Chess frontend: a classic cream-and-green club board with click-to-move and
// pointer drag-and-drop, legal-move hints, sliding piece animation,
// last-move/check highlights, captured-piece trays and a promotion picker.
// Pieces are a hand-drawn inline SVG set. Spectator mode renders and animates
// only.
//
// Game-private JSON contract with games/chess/src/ui.rs:
//   view_data:  {"board":"<64 chars, rank 8 first, file a first; '.' empty,
//                PNBRQK white / pnbrqk black>","turn":"w"|"b","check":bool}
//   transition: {"from":"e2","to":"e4","piece":"P","captured":"p"|null,
//                "capturedSquare":"d5"|null,"promo":"Q"|null,
//                "castleRookFrom":"h1"|null,"castleRookTo":"f1"|null,
//                "check":bool,"mate":bool}
// Action labels are UCI coordinate moves ("e2e4"; promotions "e7e8q").

import type { MatchEventData, ViewState } from '../../engine/protocol';
import type { FrontendCtx, GameFrontend } from '../types';
import { sleep } from '../types';

interface ChessView {
  board: string;
  turn: 'w' | 'b';
  check: boolean;
}

interface ChessTransition {
  from: string;
  to: string;
  capturedSquare: string | null;
  castleRookFrom: string | null;
  castleRookTo: string | null;
}

interface BarEls {
  root: HTMLElement;
  tray: HTMLElement;
  score: HTMLElement;
}

interface DragState {
  pointerId: number;
  from: number;
  el: HTMLElement;
  ghost: HTMLElement;
  wasSelected: boolean;
  hover: number | null;
}

const TRAY_ORDER = ['q', 'r', 'b', 'n', 'p'] as const;
const START_COUNT: Record<string, number> = { q: 1, r: 2, b: 2, n: 2, p: 8 };
const PIECE_POINTS: Record<string, number> = { q: 9, r: 5, b: 3, n: 3, p: 1 };
const PROMO_ORDER = ['q', 'r', 'b', 'n'] as const;
const UCI_RE = /^[a-h][1-8][a-h][1-8][qrbn]?$/;
const SLIDE_MS = 240;
const SETTLE_MS = 120;
const SNAP_MS = 160;

// Hand-drawn minimal piece set on the conventional 45x45 grid. Shape classes:
// pcb body (filled + outlined), pcd detail stroke, pcf detail fill.
const PIECE_SHAPES: Record<string, string> = {
  p: `<circle class="pcb" cx="22.5" cy="15.5" r="4.5"/>
<path class="pcb" d="M22.5 19.7c-3.3 0-5.4 2.4-5.4 5 0 1.8 0.9 3.3 2.3 4.3-2.9 1.6-4.9 4.2-4.9 6.5h16c0-2.3-2-4.9-4.9-6.5 1.4-1 2.3-2.5 2.3-4.3 0-2.6-2.1-5-5.4-5z"/>
<rect class="pcb" x="14" y="35.5" width="17" height="4.5" rx="2"/>`,
  n: `<path class="pcb" d="M14.5 35.5c0-7.5 1-11.5 4-14-2.5 0-6-1-7.5-4l0-2c0-1.5 1.5-3.2 3.5-3.5 2-0.4 3.4-2 4-5l2.1 3 2.4-3.5c1 1.5 1.6 2.7 1.6 3.8 4.4 1.7 8.9 6.7 8.9 13.7v11.5z"/>
<circle class="pcf" cx="16.2" cy="15.4" r="1"/>
<rect class="pcb" x="12" y="35.5" width="21.5" height="4.5" rx="2"/>`,
  b: `<circle class="pcb" cx="22.5" cy="9" r="1.9"/>
<path class="pcb" d="M22.5 11.5c3.4 2.5 5.5 5.6 5.5 8.9 0 2.3-1.1 4.4-2.9 5.7 3 2 5 6 5.4 9.4h-16c0.4-3.4 2.4-7.4 5.4-9.4-1.8-1.3-2.9-3.4-2.9-5.7 0-3.3 2.1-6.4 5.5-8.9z"/>
<path class="pcd" d="M22.5 15v6.4M19.6 18.2h5.8"/>
<rect class="pcb" x="12.5" y="35.5" width="20" height="4.5" rx="2"/>`,
  r: `<path class="pcb" d="M13.5 35.5v-4l2-2.5v-10l-2-2v-7h4v3h3v-3h4v3h3v-3h4v7l-2 2v10l2 2.5v4z"/>
<rect class="pcb" x="11.5" y="35.5" width="22" height="4.5" rx="2"/>`,
  q: `<path class="pcb" d="M14 21l-2.5-9.5 5 4.7 1.6-7.7 3 6.6 1.4-8.1 1.4 8.1 3-6.6 1.6 7.7 5-4.7-2.5 9.5c1 3-0.3 5.2-2.1 6.6 2.6 1.9 4.2 4.3 4.5 7.9h-21.8c0.3-3.6 1.9-6 4.5-7.9-1.8-1.4-3.1-3.6-2.1-6.6z"/>
<rect class="pcb" x="11.5" y="35.5" width="22" height="4.5" rx="2"/>`,
  k: `<path class="pcb" d="M21.3 4h2.4v3h2.9v2.4h-2.9v3h-2.4v-3h-2.9v-2.4h2.9z"/>
<path class="pcb" d="M22.5 12.8c5.3 0 9 3.3 9 7.4 0 2.5-1.4 4.8-3.5 6.2 3.2 2 5.3 5 5.6 9.1h-22.2c0.3-4.1 2.4-7.1 5.6-9.1-2.1-1.4-3.5-3.7-3.5-6.2 0-4.1 3.7-7.4 9-7.4z"/>
<path class="pcd" d="M16.2 21h12.6"/>
<rect class="pcb" x="11" y="35.5" width="23" height="4.5" rx="2"/>`,
};

function pieceSvg(type: string, white: boolean): string {
  const shapes = PIECE_SHAPES[type] ?? '';
  const color = white ? 'chess-pc-w' : 'chess-pc-b';
  return `<svg class="chess-pc ${color}" viewBox="0 0 45 45" aria-hidden="true">${shapes}</svg>`;
}

/** Squares are 0..63, a1 = 0, h1 = 7, a8 = 56 (matches the Rust side). */
function sqIndex(name: string): number {
  return (name.charCodeAt(1) - 49) * 8 + (name.charCodeAt(0) - 97);
}

/** Piece letter at `sq` in the rank-8-first board string ('.' when empty). */
function pieceAt(board: string, sq: number): string {
  return board.charAt((7 - Math.floor(sq / 8)) * 8 + (sq % 8));
}

function parseView(data: unknown): ChessView | null {
  if (typeof data !== 'object' || data === null) return null;
  const v = data as Record<string, unknown>;
  if (typeof v.board !== 'string' || v.board.length !== 64) return null;
  return { board: v.board, turn: v.turn === 'b' ? 'b' : 'w', check: v.check === true };
}

function parseTransition(data: unknown): ChessTransition | null {
  if (typeof data !== 'object' || data === null) return null;
  const t = data as Record<string, unknown>;
  if (typeof t.from !== 'string' || typeof t.to !== 'string') return null;
  if (!UCI_RE.test(t.from + t.to)) return null;
  const sq = (v: unknown) => (typeof v === 'string' && v.length === 2 ? v : null);
  return {
    from: t.from,
    to: t.to,
    capturedSquare: sq(t.capturedSquare),
    castleRookFrom: sq(t.castleRookFrom),
    castleRookTo: sq(t.castleRookTo),
  };
}

function transitionFromLabel(label: string): ChessTransition | null {
  if (!UCI_RE.test(label)) return null;
  return {
    from: label.slice(0, 2),
    to: label.slice(2, 4),
    capturedSquare: null,
    castleRookFrom: null,
    castleRookTo: null,
  };
}

class ChessFrontend implements GameFrontend {
  private ctx!: FrontendCtx;
  private host!: HTMLElement;
  private rootEl!: HTMLElement;
  private boardEl!: HTMLElement;
  private piecesEl!: HTMLElement;
  private promoEl!: HTMLElement;
  private bars!: Record<'w' | 'b', BarEls>;
  private squareEls: HTMLElement[] = [];
  private pieceEls = new Map<number, HTMLElement>();
  private flipped = false;
  private view: ChessView | null = null;
  private lastMove: { from: number; to: number } | null = null;
  private gameOver = false;
  private moves = new Map<number, Map<number, string[]>>();
  private selected: number | null = null;
  private inputArmed = false;
  private drag: DragState | null = null;
  private skipSlide = false;
  private promoFromDrag = false;

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    this.host = host;
    this.flipped = ctx.humanSeat === 1;
    injectStyle();

    const bar = `
      <span class="chess-turn-dot"></span>
      <span class="chess-bar-name"></span>
      <span class="chess-tray"></span>
      <span class="chess-score"></span>`;
    host.innerHTML = `
      <div class="chess-root">
        <div class="chess-bar chess-bar-top">${bar}</div>
        <div class="chess-stage">
          <div class="chess-ranks"></div>
          <div class="chess-board">
            <div class="chess-squares"></div>
            <div class="chess-pieces"></div>
            <div class="chess-promo" hidden></div>
          </div>
          <div class="chess-files"></div>
        </div>
        <div class="chess-bar chess-bar-bottom">${bar}</div>
      </div>`;
    this.rootEl = host.querySelector<HTMLElement>('.chess-root')!;
    this.boardEl = host.querySelector<HTMLElement>('.chess-board')!;
    this.piecesEl = host.querySelector<HTMLElement>('.chess-pieces')!;
    this.promoEl = host.querySelector<HTMLElement>('.chess-promo')!;

    const topBar = host.querySelector<HTMLElement>('.chess-bar-top')!;
    const bottomBar = host.querySelector<HTMLElement>('.chess-bar-bottom')!;
    const barEls = (el: HTMLElement): BarEls => ({
      root: el,
      tray: el.querySelector<HTMLElement>('.chess-tray')!,
      score: el.querySelector<HTMLElement>('.chess-score')!,
    });
    this.bars = this.flipped
      ? { w: barEls(topBar), b: barEls(bottomBar) }
      : { w: barEls(bottomBar), b: barEls(topBar) };
    const name = (color: 'w' | 'b') => {
      const colorName = color === 'w' ? 'White' : 'Black';
      const seat = color === 'w' ? 0 : 1;
      return seat === ctx.humanSeat ? `You · ${colorName}` : `Bot · ${colorName}`;
    };
    for (const color of ['w', 'b'] as const) {
      this.bars[color].root.querySelector('.chess-bar-name')!.textContent = name(color);
    }

    const ranks = host.querySelector<HTMLElement>('.chess-ranks')!;
    const files = host.querySelector<HTMLElement>('.chess-files')!;
    for (let i = 0; i < 8; i++) {
      const rank = this.flipped ? i + 1 : 8 - i;
      const file = this.flipped ? 7 - i : i;
      ranks.insertAdjacentHTML('beforeend', `<span>${rank}</span>`);
      files.insertAdjacentHTML('beforeend', `<span>${'abcdefgh'[file]}</span>`);
    }

    const squares = host.querySelector<HTMLElement>('.chess-squares')!;
    this.squareEls = new Array<HTMLElement>(64);
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const file = this.flipped ? 7 - col : col;
        const rank = this.flipped ? row : 7 - row;
        const sq = rank * 8 + file;
        const el = document.createElement('div');
        el.className = `chess-sq ${(file + rank) % 2 === 1 ? 'chess-sq-light' : 'chess-sq-dark'}`;
        el.dataset.sq = String(sq);
        this.squareEls[sq] = el;
        squares.append(el);
      }
    }

    this.boardEl.addEventListener('pointerdown', (e) => this.onPointerDown(e));
    this.boardEl.addEventListener('pointermove', (e) => this.onPointerMove(e));
    this.boardEl.addEventListener('pointerup', (e) => this.onPointerUp(e));
    this.boardEl.addEventListener('pointercancel', () => this.cancelDrag(true));
    this.boardEl.addEventListener('contextmenu', (e) => {
      if (this.drag) e.preventDefault();
    });
    this.promoEl.addEventListener('click', (e) => {
      if (e.target === this.promoEl) {
        this.closePromo();
        this.select(null);
      }
    });
  }

  render(state: ViewState): void {
    const view = parseView(state.viewData);
    if (!view) return;
    this.view = view;
    this.gameOver = state.isOver;
    this.syncAll();
  }

  async animate(event: MatchEventData, after: ViewState): Promise<void> {
    const skip = this.skipSlide;
    this.skipSlide = false;
    this.disarm();
    const view = parseView(after.viewData);
    if (!view) return;
    const move = parseTransition(event.data) ?? transitionFromLabel(event.label);
    if (move) this.lastMove = { from: sqIndex(move.from), to: sqIndex(move.to) };
    this.gameOver = after.isOver;
    const scale = this.ctx.animationScale();
    if (move && scale > 0 && !skip) await this.slide(move, scale);
    this.view = view;
    this.syncAll();
    if (scale > 0 && !skip) await sleep(SETTLE_MS * scale);
  }

  promptAction(labels: string[]): void {
    if (this.ctx.humanSeat < 0) return;
    this.moves.clear();
    for (const label of labels) {
      if (!UCI_RE.test(label)) continue;
      const from = sqIndex(label.slice(0, 2));
      const to = sqIndex(label.slice(2, 4));
      let dests = this.moves.get(from);
      if (!dests) {
        dests = new Map();
        this.moves.set(from, dests);
      }
      const list = dests.get(to);
      if (list) list.push(label);
      else dests.set(to, [label]);
    }
    this.inputArmed = true;
    this.rootEl.classList.add('chess-armed');
    for (const from of this.moves.keys()) this.squareEls[from].classList.add('chess-sq-movable');
  }

  unmount(): void {
    this.host.replaceChildren();
  }

  // ---------- input ----------

  private squareAt(x: number, y: number): number | null {
    const r = this.boardEl.getBoundingClientRect();
    if (x < r.left || x >= r.right || y < r.top || y >= r.bottom) return null;
    const col = Math.floor(((x - r.left) / r.width) * 8);
    const row = Math.floor(((y - r.top) / r.height) * 8);
    const file = this.flipped ? 7 - col : col;
    const rank = this.flipped ? row : 7 - row;
    return rank * 8 + file;
  }

  private onPointerDown(e: PointerEvent): void {
    if (!this.inputArmed || !this.promoEl.hidden || this.drag) return;
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    const sq = this.squareAt(e.clientX, e.clientY);
    if (sq === null) return;
    if (this.selected !== null) {
      const labels = this.moves.get(this.selected)?.get(sq);
      if (labels) {
        if (labels.length > 1) this.openPromo(labels);
        else this.submitMove(labels[0]);
        return;
      }
    }
    if (!this.moves.has(sq)) {
      this.select(null);
      return;
    }
    const el = this.pieceEls.get(sq);
    if (!el) return;
    e.preventDefault();
    const wasSelected = this.selected === sq;
    this.select(sq);
    const ghost = el.cloneNode(true) as HTMLElement;
    ghost.classList.add('chess-piece-ghost');
    this.piecesEl.append(ghost);
    el.classList.add('chess-piece-drag');
    this.boardEl.classList.add('chess-dragging');
    this.drag = { pointerId: e.pointerId, from: sq, el, ghost, wasSelected, hover: null };
    this.boardEl.setPointerCapture(e.pointerId);
    this.moveDragTo(e.clientX, e.clientY);
  }

  private onPointerMove(e: PointerEvent): void {
    if (!this.drag || e.pointerId !== this.drag.pointerId) return;
    this.moveDragTo(e.clientX, e.clientY);
  }

  private moveDragTo(x: number, y: number): void {
    if (!this.drag) return;
    const r = this.boardEl.getBoundingClientRect();
    const cell = r.width / 8;
    const px = x - r.left - cell / 2;
    const py = y - r.top - cell / 2;
    this.drag.el.style.transform = `translate(${px}px, ${py}px) scale(1.15)`;
    const sq = this.squareAt(x, y);
    if (sq === this.drag.hover) return;
    if (this.drag.hover !== null) this.squareEls[this.drag.hover].classList.remove('chess-sq-drop');
    this.drag.hover = null;
    if (sq !== null && sq !== this.drag.from && this.moves.get(this.drag.from)?.has(sq)) {
      this.squareEls[sq].classList.add('chess-sq-drop');
      this.drag.hover = sq;
    }
  }

  private onPointerUp(e: PointerEvent): void {
    if (!this.drag || e.pointerId !== this.drag.pointerId) return;
    const d = this.drag;
    this.drag = null;
    this.endDragVisuals(d);
    const sq = this.squareAt(e.clientX, e.clientY);
    const labels = sq !== null && sq !== d.from ? this.moves.get(d.from)?.get(sq) : undefined;
    if (sq !== null && labels) {
      this.settle(d.el, sq);
      this.removeVictim(d.from, sq);
      if (labels.length > 1) {
        this.promoFromDrag = true;
        this.openPromo(labels);
      } else {
        this.submitMove(labels[0], true);
      }
      return;
    }
    this.snapBack(d.el, d.from);
    if (sq === d.from && d.wasSelected) this.select(null);
  }

  private cancelDrag(animate: boolean): void {
    if (!this.drag) return;
    const d = this.drag;
    this.drag = null;
    this.endDragVisuals(d);
    if (animate) this.snapBack(d.el, d.from);
    else {
      d.el.classList.remove('chess-piece-drag');
      this.place(d.el, d.from);
    }
  }

  private endDragVisuals(d: DragState): void {
    if (d.hover !== null) this.squareEls[d.hover].classList.remove('chess-sq-drop');
    d.ghost.remove();
    this.boardEl.classList.remove('chess-dragging');
  }

  private settle(el: HTMLElement, sq: number): void {
    el.classList.remove('chess-piece-drag');
    el.style.zIndex = '5';
    this.place(el, sq);
  }

  private snapBack(el: HTMLElement, sq: number): void {
    el.classList.remove('chess-piece-drag');
    const ms = SNAP_MS * this.ctx.animationScale();
    if (ms <= 0) {
      this.place(el, sq);
      return;
    }
    el.style.zIndex = '5';
    el.style.transitionDuration = `${ms}ms`;
    void el.offsetWidth;
    this.place(el, sq);
    window.setTimeout(() => {
      el.style.transitionDuration = '';
      el.style.zIndex = '';
    }, ms + 30);
  }

  /** Clear the captured piece (incl. en passant) the instant a drag drops. */
  private removeVictim(from: number, to: number): void {
    const direct = this.pieceEls.get(to);
    if (direct) {
      direct.remove();
      this.pieceEls.delete(to);
      return;
    }
    if (!this.view) return;
    const mover = pieceAt(this.view.board, from).toLowerCase();
    if (mover !== 'p' || from % 8 === to % 8) return;
    const epSq = Math.floor(from / 8) * 8 + (to % 8);
    const ep = this.pieceEls.get(epSq);
    if (ep) {
      ep.remove();
      this.pieceEls.delete(epSq);
    }
  }

  private select(sq: number | null): void {
    this.clearSelection();
    if (sq === null) return;
    const dests = this.moves.get(sq);
    if (!dests) return;
    this.selected = sq;
    this.squareEls[sq].classList.add('chess-sq-selected');
    for (const dest of dests.keys()) {
      this.squareEls[dest].classList.add(
        this.pieceEls.has(dest) ? 'chess-sq-capture' : 'chess-sq-target',
      );
    }
  }

  private submitMove(label: string, fromDrag = false): void {
    this.skipSlide = fromDrag;
    this.promoFromDrag = false;
    this.disarm();
    this.ctx.submit(label);
  }

  private openPromo(labels: string[]): void {
    const from = sqIndex(labels[0].slice(0, 2));
    const ch = this.view ? pieceAt(this.view.board, from) : 'P';
    const white = ch === ch.toUpperCase();
    const fromDrag = this.promoFromDrag;
    const panel = document.createElement('div');
    panel.className = 'chess-promo-panel';
    for (const suffix of PROMO_ORDER) {
      const label = labels.find((l) => l.charAt(4) === suffix);
      if (!label) continue;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'chess-promo-btn';
      btn.innerHTML = pieceSvg(suffix, white);
      btn.onclick = () => this.submitMove(label, fromDrag);
      panel.append(btn);
    }
    this.promoEl.replaceChildren(panel);
    this.promoEl.hidden = false;
  }

  private closePromo(): void {
    const restore = this.promoFromDrag;
    this.promoFromDrag = false;
    this.promoEl.hidden = true;
    this.promoEl.replaceChildren();
    if (restore && this.view) this.syncPieces(this.view);
  }

  private disarm(): void {
    this.cancelDrag(false);
    this.inputArmed = false;
    this.moves.clear();
    this.clearSelection();
    this.closePromo();
    this.rootEl.classList.remove('chess-armed');
    for (const el of this.squareEls) el.classList.remove('chess-sq-movable');
  }

  private clearSelection(): void {
    this.selected = null;
    for (const el of this.squareEls) {
      el.classList.remove('chess-sq-selected', 'chess-sq-target', 'chess-sq-capture', 'chess-sq-drop');
    }
  }

  // ---------- board sync ----------

  private syncAll(): void {
    if (!this.view) return;
    this.clearSelection();
    this.syncPieces(this.view);
    this.syncHighlights(this.view);
    this.syncBars(this.view);
  }

  private syncPieces(view: ChessView): void {
    this.pieceEls.clear();
    const frag = document.createDocumentFragment();
    for (let sq = 0; sq < 64; sq++) {
      const ch = pieceAt(view.board, sq);
      if (ch === '.') continue;
      const el = document.createElement('div');
      el.className = 'chess-piece';
      el.innerHTML = pieceSvg(ch.toLowerCase(), ch === ch.toUpperCase());
      this.place(el, sq);
      this.pieceEls.set(sq, el);
      frag.append(el);
    }
    this.piecesEl.replaceChildren(frag);
  }

  private syncHighlights(view: ChessView): void {
    for (const el of this.squareEls) {
      el.classList.remove('chess-sq-last', 'chess-sq-check', 'chess-sq-mate');
    }
    if (this.lastMove) {
      this.squareEls[this.lastMove.from].classList.add('chess-sq-last');
      this.squareEls[this.lastMove.to].classList.add('chess-sq-last');
    }
    if (view.check) {
      const king = view.turn === 'w' ? 'K' : 'k';
      for (let sq = 0; sq < 64; sq++) {
        if (pieceAt(view.board, sq) === king) {
          this.squareEls[sq].classList.add('chess-sq-check');
          if (this.gameOver) this.squareEls[sq].classList.add('chess-sq-mate');
        }
      }
    }
  }

  private syncBars(view: ChessView): void {
    const counts: Record<string, number> = {};
    for (const ch of view.board) {
      if (ch !== '.') counts[ch] = (counts[ch] ?? 0) + 1;
    }
    const lost = (color: 'w' | 'b') => {
      const pieces: string[] = [];
      let pts = 0;
      for (const p of TRAY_ORDER) {
        const have = counts[color === 'w' ? p.toUpperCase() : p] ?? 0;
        const n = Math.max(0, (START_COUNT[p] ?? 0) - have);
        pts += n * (PIECE_POINTS[p] ?? 0);
        for (let i = 0; i < n; i++) pieces.push(p);
      }
      return { pieces, pts };
    };
    const wLost = lost('w');
    const bLost = lost('b');
    for (const color of ['w', 'b'] as const) {
      const taken = color === 'w' ? bLost : wLost;
      const lead = color === 'w' ? bLost.pts - wLost.pts : wLost.pts - bLost.pts;
      const els = this.bars[color];
      els.tray.replaceChildren(
        ...taken.pieces.map((p) => {
          const s = document.createElement('span');
          s.className = 'chess-tray-piece';
          s.innerHTML = pieceSvg(p, color === 'b');
          return s;
        }),
      );
      els.score.textContent = lead > 0 ? `+${lead}` : '';
      els.root.classList.toggle('chess-bar-active', !this.gameOver && view.turn === color);
    }
  }

  private place(el: HTMLElement, sq: number): void {
    const col = this.flipped ? 7 - (sq % 8) : sq % 8;
    const row = this.flipped ? Math.floor(sq / 8) : 7 - Math.floor(sq / 8);
    el.style.transform = `translate(${col * 100}%, ${row * 100}%)`;
  }

  private async slide(t: ChessTransition, scale: number): Promise<void> {
    const ms = SLIDE_MS * scale;
    const from = sqIndex(t.from);
    const to = sqIndex(t.to);
    const mover = this.pieceEls.get(from);
    if (!mover) return;
    const victimSq =
      t.capturedSquare !== null ? sqIndex(t.capturedSquare) : this.pieceEls.has(to) ? to : null;
    if (victimSq !== null && victimSq !== from) {
      const victim = this.pieceEls.get(victimSq);
      if (victim) {
        victim.style.transition = `opacity ${ms}ms ease`;
        victim.style.opacity = '0';
      }
    }
    const glide = (el: HTMLElement, dest: number) => {
      el.style.zIndex = '3';
      el.style.transitionDuration = `${ms}ms`;
      void el.offsetWidth;
      this.place(el, dest);
    };
    glide(mover, to);
    if (t.castleRookFrom !== null && t.castleRookTo !== null) {
      const rook = this.pieceEls.get(sqIndex(t.castleRookFrom));
      if (rook) glide(rook, sqIndex(t.castleRookTo));
    }
    await sleep(ms + 30);
  }
}

export function createChessFrontend(): GameFrontend {
  return new ChessFrontend();
}

// ---------- styles ----------

const STYLE_ID = 'chess-frontend-style';

function injectStyle(): void {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = CSS_TEXT;
  document.head.append(style);
}

const CSS_TEXT = `
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
  box-shadow: 0 1px 0 rgba(244, 238, 218, 0.05), 0 14px 30px rgba(5, 8, 3, 0.45);
  user-select: none;
  -webkit-user-select: none;
  touch-action: none;
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
`;
