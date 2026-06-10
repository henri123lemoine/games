// Chess frontend: a dedicated board with click-to-move and legal-move hints,
// sliding piece animation, last-move/check highlights, captured-piece trays
// and a promotion picker. Spectator mode renders and animates only.
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

const GLYPH: Record<string, string> = { k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟' };
const TRAY_ORDER = ['q', 'r', 'b', 'n', 'p'] as const;
const START_COUNT: Record<string, number> = { q: 1, r: 2, b: 2, n: 2, p: 8 };
const PIECE_POINTS: Record<string, number> = { q: 9, r: 5, b: 3, n: 3, p: 1 };
const PROMO_ORDER = ['q', 'r', 'b', 'n'] as const;
const UCI_RE = /^[a-h][1-8][a-h][1-8][qrbn]?$/;
const SLIDE_MS = 240;
const SETTLE_MS = 120;

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
  private resizeObs: ResizeObserver | null = null;

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
        <div class="chess-board">
          <div class="chess-squares"></div>
          <div class="chess-pieces"></div>
          <div class="chess-promo" hidden></div>
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
        if (col === 0) {
          el.insertAdjacentHTML(
            'beforeend',
            `<span class="chess-coord chess-coord-rank">${rank + 1}</span>`,
          );
        }
        if (row === 7) {
          el.insertAdjacentHTML(
            'beforeend',
            `<span class="chess-coord chess-coord-file">${'abcdefgh'[file]}</span>`,
          );
        }
        this.squareEls[sq] = el;
        squares.append(el);
      }
    }
    squares.addEventListener('click', (e) => {
      const sqEl = (e.target as HTMLElement).closest<HTMLElement>('.chess-sq');
      if (sqEl?.dataset.sq) this.onSquareClick(Number(sqEl.dataset.sq));
    });
    this.promoEl.addEventListener('click', (e) => {
      if (e.target === this.promoEl) {
        this.closePromo();
        this.select(null);
      }
    });

    this.resizeObs = new ResizeObserver(() => this.updateFont());
    this.resizeObs.observe(this.boardEl);
  }

  render(state: ViewState): void {
    const view = parseView(state.viewData);
    if (!view) return;
    this.view = view;
    this.gameOver = state.isOver;
    this.syncAll();
  }

  async animate(event: MatchEventData, after: ViewState): Promise<void> {
    this.disarm();
    const view = parseView(after.viewData);
    if (!view) return;
    const move = parseTransition(event.data) ?? transitionFromLabel(event.label);
    if (move) this.lastMove = { from: sqIndex(move.from), to: sqIndex(move.to) };
    this.gameOver = after.isOver;
    const scale = this.ctx.animationScale();
    if (move && scale > 0) await this.slide(move, scale);
    this.view = view;
    this.syncAll();
    if (scale > 0) await sleep(SETTLE_MS * scale);
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
    this.resizeObs?.disconnect();
    this.resizeObs = null;
    this.host.replaceChildren();
  }

  // ---------- input ----------

  private onSquareClick(sq: number): void {
    if (!this.inputArmed || !this.promoEl.hidden) return;
    if (this.selected !== null) {
      const labels = this.moves.get(this.selected)?.get(sq);
      if (labels) {
        if (labels.length > 1) this.openPromo(labels);
        else this.submitMove(labels[0]);
        return;
      }
    }
    this.select(this.moves.has(sq) && sq !== this.selected ? sq : null);
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

  private submitMove(label: string): void {
    this.disarm();
    this.ctx.submit(label);
  }

  private openPromo(labels: string[]): void {
    const from = sqIndex(labels[0].slice(0, 2));
    const ch = this.view ? pieceAt(this.view.board, from) : 'P';
    const white = ch === ch.toUpperCase();
    const panel = document.createElement('div');
    panel.className = 'chess-promo-panel';
    for (const suffix of PROMO_ORDER) {
      const label = labels.find((l) => l.charAt(4) === suffix);
      if (!label) continue;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `chess-promo-btn ${white ? 'chess-piece-w' : 'chess-piece-b'}`;
      btn.textContent = GLYPH[suffix] ?? '';
      btn.onclick = () => this.submitMove(label);
      panel.append(btn);
    }
    this.promoEl.replaceChildren(panel);
    this.promoEl.hidden = false;
  }

  private closePromo(): void {
    this.promoEl.hidden = true;
    this.promoEl.replaceChildren();
  }

  private disarm(): void {
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
      el.classList.remove('chess-sq-selected', 'chess-sq-target', 'chess-sq-capture');
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
      const white = ch === ch.toUpperCase();
      el.className = `chess-piece ${white ? 'chess-piece-w' : 'chess-piece-b'}`;
      el.textContent = GLYPH[ch.toLowerCase()] ?? '';
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
          s.className = `chess-tray-piece ${color === 'w' ? 'chess-piece-b' : 'chess-piece-w'}`;
          s.textContent = GLYPH[p] ?? '';
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

  private updateFont(): void {
    const w = this.boardEl.clientWidth;
    if (w > 0) this.piecesEl.style.fontSize = `${(w / 8) * 0.72}px`;
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
  max-width: 540px;
  max-width: min(540px, calc(100dvh - 270px));
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
  gap: 1px;
  font-size: 1.15rem;
  line-height: 1;
  min-height: 1.15rem;
}

.chess-score {
  color: var(--good);
  font-size: 0.85rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.chess-board {
  position: relative;
  width: 100%;
  aspect-ratio: 1 / 1;
  border-radius: calc(var(--radius) - 3px);
  overflow: hidden;
  box-shadow: 0 14px 36px rgba(1, 4, 9, 0.5);
  user-select: none;
  -webkit-user-select: none;
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
  background: #b9c7da;
}

.chess-sq-dark {
  background: #50647f;
}

.chess-armed .chess-sq-movable {
  cursor: pointer;
}

.chess-armed .chess-sq-movable:hover::before {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(88, 166, 255, 0.16);
}

.chess-sq-last::before {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(88, 166, 255, 0.27);
}

.chess-sq-selected::before {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(88, 166, 255, 0.45);
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
  width: 27%;
  height: 27%;
  border-radius: 50%;
  background: rgba(7, 12, 20, 0.38);
}

.chess-sq-capture::after {
  content: '';
  position: absolute;
  inset: 7%;
  border-radius: 50%;
  border: 3px solid rgba(248, 81, 73, 0.65);
}

.chess-sq-check {
  background-image: radial-gradient(
    circle at 50% 50%,
    rgba(248, 81, 73, 0.65) 22%,
    rgba(248, 81, 73, 0.25) 50%,
    transparent 68%
  );
}

.chess-sq-mate {
  background-image: radial-gradient(
    circle at 50% 50%,
    rgba(248, 81, 73, 0.85) 26%,
    rgba(248, 81, 73, 0.35) 55%,
    transparent 75%
  );
}

.chess-coord {
  position: absolute;
  font-size: 11px;
  font-weight: 700;
  line-height: 1;
  opacity: 0.85;
  pointer-events: none;
}

.chess-coord-rank {
  top: 3px;
  left: 4px;
}

.chess-coord-file {
  bottom: 3px;
  right: 4px;
}

.chess-sq-light .chess-coord {
  color: #50647f;
}

.chess-sq-dark .chess-coord {
  color: #b9c7da;
}

.chess-pieces {
  position: absolute;
  inset: 0;
  pointer-events: none;
  font-size: 44px;
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
  line-height: 1;
  z-index: 1;
  will-change: transform;
  transition: transform 0ms cubic-bezier(0.22, 0.85, 0.3, 1);
  font-family: 'Segoe UI Symbol', 'Noto Sans Symbols 2', 'DejaVu Sans', sans-serif;
}

.chess-piece-w {
  color: #f4f1e8;
  text-shadow: 0 2px 3px rgba(1, 4, 9, 0.55), 0 0 2px rgba(1, 4, 9, 0.7);
}

.chess-piece-b {
  color: #23272e;
  text-shadow: 0 1px 2px rgba(244, 241, 232, 0.22), 0 0 2px rgba(244, 241, 232, 0.28);
}

.chess-promo {
  position: absolute;
  inset: 0;
  z-index: 6;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(1, 4, 9, 0.62);
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
  box-shadow: 0 16px 48px rgba(1, 4, 9, 0.6);
}

.chess-promo-btn {
  width: 62px;
  height: 62px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 42px;
  line-height: 1;
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) - 3px);
  cursor: pointer;
  transition: border-color 0.15s ease, transform 0.15s ease;
  font-family: 'Segoe UI Symbol', 'Noto Sans Symbols 2', 'DejaVu Sans', sans-serif;
}

.chess-promo-btn:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
}
`;
