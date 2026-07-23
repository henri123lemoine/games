// Four-player chess frontend: a fixed-orientation 14×14 cross board framed by
// the four armies at their compass edges. The board is never rotated: Yellow
// is north, Blue west, Green east, and Red south.

import type { MatchEventData, ViewState } from '../../engine/protocol';
import type { FrontendCtx, GameFrontend } from '../types';

type Color = 'r' | 'b' | 'y' | 'g';

interface FourPiece {
  square: string;
  color: Color;
  piece: string;
  dead: boolean;
  promoted: boolean;
}

interface FourView {
  size: 14;
  pieces: FourPiece[];
  turn: Color;
  active: [boolean, boolean, boolean, boolean];
  scores: [number, number, number, number];
  check: [boolean, boolean, boolean, boolean];
  end: string;
  last: string | null;
}

interface FourTransition {
  from: string;
  to: string;
  color: Color;
  scoreGain: number;
}

const COLORS: Color[] = ['r', 'b', 'y', 'g'];
const NAMES = ['Red', 'Blue', 'Yellow', 'Green'];
const GLYPHS: Record<string, string> = {
  P: '♟',
  N: '♞',
  B: '♝',
  R: '♜',
  Q: '♛',
  K: '♚',
};
const MOVE_RE = /^([a-n](?:1[0-4]|[1-9]))([a-n](?:1[0-4]|[1-9]))(?:=Q)?$/i;
const STYLE_ID = 'four-player-chess-frontend-style';

function seat(color: Color): number {
  return COLORS.indexOf(color);
}

function isColor(value: unknown): value is Color {
  return value === 'r' || value === 'b' || value === 'y' || value === 'g';
}

function isSquare(value: unknown): value is string {
  if (typeof value !== 'string') return false;
  const match = /^([a-n])(1[0-4]|[1-9])$/.exec(value);
  if (!match) return false;
  const x = match[1].charCodeAt(0) - 97;
  const y = Number(match[2]) - 1;
  return !((x < 3 || x > 10) && (y < 3 || y > 10));
}

function parseMove(label: string): { from: string; to: string } | null {
  const match = MOVE_RE.exec(label.trim());
  return match && isSquare(match[1]) && isSquare(match[2])
    ? { from: match[1].toLowerCase(), to: match[2].toLowerCase() }
    : null;
}

function parseView(data: unknown): FourView | null {
  if (!data || typeof data !== 'object') return null;
  const view = data as Partial<FourView>;
  if (
    view.size !== 14 ||
    !isColor(view.turn) ||
    !Array.isArray(view.pieces) ||
    !Array.isArray(view.active) ||
    view.active.length !== 4 ||
    !Array.isArray(view.scores) ||
    view.scores.length !== 4 ||
    !Array.isArray(view.check) ||
    view.check.length !== 4
  ) {
    return null;
  }
  return view as FourView;
}

function parseTransition(data: unknown, label: string): FourTransition | null {
  const fallback = parseMove(label);
  if (!data || typeof data !== 'object') {
    return fallback ? { ...fallback, color: 'r', scoreGain: 0 } : null;
  }
  const move = data as Partial<FourTransition>;
  const from = isSquare(move.from) ? move.from : fallback?.from;
  const to = isSquare(move.to) ? move.to : fallback?.to;
  if (!from || !to || !isColor(move.color)) return null;
  return {
    from,
    to,
    color: move.color,
    scoreGain: typeof move.scoreGain === 'number' ? move.scoreGain : 0,
  };
}

function xy(square: string): { x: number; y: number } {
  return { x: square.charCodeAt(0) - 97, y: Number(square.slice(1)) - 1 };
}

function injectStyle(): void {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.append(style);
}

class FourPlayerChessFrontend implements GameFrontend {
  private ctx!: FrontendCtx;
  private root!: HTMLElement;
  private board!: HTMLElement;
  private message!: HTMLElement;
  private fallback!: HTMLElement;
  private squares = new Map<string, HTMLElement>();
  private rails: HTMLElement[] = [];
  private selected: string | null = null;
  private actions = new Map<string, Map<string, string>>();

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    injectStyle();
    host.innerHTML = `
      <div class="fpc-root">
        <div class="fpc-kicker">FREE FOR ALL <span>\u00b7</span> FOUR ARMIES <span>\u00b7</span> ONE BOARD</div>
        <div class="fpc-stage" aria-label="Four-player chess board">
          ${this.railMarkup(2, 'north')}
          ${this.railMarkup(1, 'west')}
          <div class="fpc-board" role="grid"></div>
          ${this.railMarkup(3, 'east')}
          ${this.railMarkup(0, 'south')}
        </div>
        <div class="fpc-message" aria-live="polite"></div>
        <pre class="fpc-fallback" hidden></pre>
      </div>`;
    this.root = host.querySelector<HTMLElement>('.fpc-root')!;
    this.board = host.querySelector<HTMLElement>('.fpc-board')!;
    this.message = host.querySelector<HTMLElement>('.fpc-message')!;
    this.fallback = host.querySelector<HTMLElement>('.fpc-fallback')!;
    this.rails = COLORS.map((_, i) => host.querySelector<HTMLElement>(`.fpc-rail-${i}`)!);

    for (let row = 0; row < 14; row++) {
      const y = 13 - row;
      for (let x = 0; x < 14; x++) {
        if ((x < 3 || x > 10) && (y < 3 || y > 10)) continue;
        const name = `${String.fromCharCode(97 + x)}${y + 1}`;
        const square = document.createElement('button');
        square.type = 'button';
        square.className = `fpc-square ${(x + y) % 2 ? 'fpc-light' : 'fpc-dark'}`;
        square.dataset.square = name;
        square.setAttribute('role', 'gridcell');
        square.setAttribute('aria-label', name);
        square.style.gridColumn = String(x + 1);
        square.style.gridRow = String(row + 1);
        square.addEventListener('click', () => this.onSquare(name));
        this.squares.set(name, square);
        this.board.append(square);
      }
    }
  }

  render(state: ViewState): void {
    this.disableInput();
    const view = parseView(state.viewData);
    if (!view) {
      this.root.classList.add('fpc-text-only');
      this.fallback.hidden = false;
      this.fallback.textContent = state.view;
      return;
    }
    this.root.classList.remove('fpc-text-only');
    this.fallback.hidden = true;
    for (const [name, square] of this.squares) {
      square.replaceChildren();
      square.setAttribute('aria-label', name);
    }

    for (const piece of view.pieces) {
      const square = this.squares.get(piece.square);
      if (!square || !isColor(piece.color)) continue;
      const el = document.createElement('span');
      el.className = `fpc-piece fpc-piece-${piece.color}${piece.dead ? ' fpc-piece-dead' : ''}`;
      el.dataset.piece = piece.piece;
      el.textContent = GLYPHS[piece.piece] ?? piece.piece;
      el.setAttribute('aria-hidden', 'true');
      if (piece.promoted) {
        const crown = document.createElement('span');
        crown.className = 'fpc-promoted';
        crown.textContent = '1';
        el.append(crown);
      }
      square.append(el);
      square.setAttribute(
        'aria-label',
        `${piece.square}, ${NAMES[seat(piece.color)]} ${piece.piece}${piece.dead ? ', eliminated' : ''}`,
      );
    }

    for (let i = 0; i < 4; i++) {
      const rail = this.rails[i];
      rail.classList.toggle('fpc-turn', !state.isOver && seat(view.turn) === i);
      rail.classList.toggle('fpc-check', view.check[i]);
      rail.classList.toggle('fpc-dead', !view.active[i]);
      rail.querySelector<HTMLElement>('.fpc-score-value')!.textContent = String(view.scores[i]);
      rail.querySelector<HTMLElement>('.fpc-state')!.textContent = !view.active[i]
        ? 'OUT'
        : view.check[i]
          ? 'CHECK'
          : seat(view.turn) === i
            ? 'MOVE'
            : 'READY';
    }

    this.clearMarks();
    if (view.last) {
      const last = parseMove(view.last);
      if (last) {
        this.squares.get(last.from)?.classList.add('fpc-last');
        this.squares.get(last.to)?.classList.add('fpc-last', 'fpc-last-to');
      }
    }
    this.message.textContent = state.isOver
      ? state.result ?? this.endLabel(view.end)
      : view.check[seat(view.turn)]
        ? `${NAMES[seat(view.turn)]} is in check`
        : `${NAMES[seat(view.turn)]} to move`;
  }

  async animate(event: MatchEventData, after: ViewState): Promise<void> {
    const transition = parseTransition(event.data, event.label);
    this.render(after);
    if (!transition || this.ctx.animationScale() <= 0) return;
    const destination = this.squares.get(transition.to)?.querySelector<HTMLElement>('.fpc-piece');
    if (destination) {
      const from = xy(transition.from);
      const to = xy(transition.to);
      const scale = this.ctx.animationScale();
      const animation = destination.animate(
        [
          {
            transform: `translate(${(from.x - to.x) * 100}%, ${(to.y - from.y) * 100}%) scale(.86)`,
            filter: 'brightness(1.5)',
          },
          { transform: 'translate(0, 0) scale(1)', filter: 'brightness(1)' },
        ],
        { duration: 220 * scale, easing: 'cubic-bezier(.2,.8,.2,1)' },
      );
      await animation.finished.catch(() => undefined);
    }
    if (transition.scoreGain > 0) {
      const rail = this.rails[seat(transition.color)];
      rail.animate(
        [
          { transform: 'scale(1)', filter: 'brightness(1)' },
          { transform: 'scale(1.035)', filter: 'brightness(1.6)' },
          { transform: 'scale(1)', filter: 'brightness(1)' },
        ],
        { duration: 380 * this.ctx.animationScale(), easing: 'ease-out' },
      );
    }
  }

  promptAction(labels: string[]): void {
    this.actions.clear();
    for (const label of labels) {
      const move = parseMove(label);
      if (!move) continue;
      let targets = this.actions.get(move.from);
      if (!targets) {
        targets = new Map();
        this.actions.set(move.from, targets);
      }
      targets.set(move.to, label);
    }
    for (const from of this.actions.keys()) {
      const square = this.squares.get(from);
      square?.classList.add('fpc-origin');
      if (square) square.tabIndex = 0;
    }
  }

  unmount(): void {
    this.actions.clear();
    this.squares.clear();
  }

  private railMarkup(seatIndex: number, position: string): string {
    const you = this.ctx.humanSeat === seatIndex ? '<span class="fpc-you">YOU</span>' : '';
    return `<div class="fpc-rail fpc-rail-${seatIndex} fpc-rail-${position}">
      <span class="fpc-pressure"></span>
      <span class="fpc-army"><i></i>${NAMES[seatIndex]}${you}</span>
      <span class="seat-slot" data-seat="${seatIndex}"></span>
      <span class="fpc-state">READY</span>
      <span class="fpc-score"><b class="fpc-score-value">0</b><small>PTS</small></span>
    </div>`;
  }

  private onSquare(square: string): void {
    if (this.selected) {
      const label = this.actions.get(this.selected)?.get(square);
      if (label) {
        this.disableInput();
        this.ctx.submit(label);
        return;
      }
    }
    this.select(this.actions.has(square) ? square : null);
  }

  private select(square: string | null): void {
    this.selected = square;
    this.clearSelectionMarks();
    if (!square) return;
    this.squares.get(square)?.classList.add('fpc-selected');
    for (const to of this.actions.get(square)?.keys() ?? []) {
      const target = this.squares.get(to);
      target?.classList.add(target.childElementCount ? 'fpc-capture' : 'fpc-target');
      if (target) target.tabIndex = 0;
    }
  }

  private disableInput(): void {
    this.actions.clear();
    this.selected = null;
    for (const square of this.squares.values()) square.tabIndex = -1;
    this.clearSelectionMarks();
  }

  private clearSelectionMarks(): void {
    for (const square of this.squares.values()) {
      square.classList.remove('fpc-selected', 'fpc-target', 'fpc-capture', 'fpc-origin');
    }
  }

  private clearMarks(): void {
    for (const square of this.squares.values()) {
      square.classList.remove('fpc-last', 'fpc-last-to');
    }
  }

  private endLabel(end: string): string {
    return end === 'last-army' ? 'One army remains' : `Game ended: ${end.replaceAll('-', ' ')}`;
  }
}

export function createFourPlayerChessFrontend(): GameFrontend {
  return new FourPlayerChessFrontend();
}

const CSS = `
.fpc-root {
  --fpc-red: #ff564d;
  --fpc-blue: #4ea1ff;
  --fpc-yellow: #ffd54a;
  --fpc-green: #42d08b;
  width: min(100%, var(--board-fit));
  margin: auto;
  padding: clamp(38px, 6.5vw, 62px);
  color: var(--text);
  user-select: none;
}
.fpc-kicker {
  margin: -28px 0 11px;
  text-align: center;
  font: 700 clamp(.58rem, 1vw, .7rem)/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: .17em;
  color: var(--text-dim);
}
.fpc-kicker span { padding: 0 .45em; opacity: .45; }
.fpc-stage {
  position: relative;
  aspect-ratio: 1;
  isolation: isolate;
  filter: drop-shadow(0 18px 24px rgba(0,0,0,.22));
}
.fpc-board {
  position: absolute;
  inset: 0;
  display: grid;
  grid-template-columns: repeat(14, 1fr);
  grid-template-rows: repeat(14, 1fr);
}
.fpc-square {
  position: relative;
  min-width: 0;
  padding: 0;
  border: 0;
  color: inherit;
  display: grid;
  place-items: center;
  cursor: default;
  appearance: none;
  transition: filter .14s, box-shadow .14s;
}
.fpc-light { background: #d7cdb5; }
.fpc-dark { background: #68756e; }
.dark .fpc-light { background: #b9b199; }
.dark .fpc-dark { background: #4d5c56; }
.fpc-square::before {
  content: '';
  position: absolute;
  inset: 0;
  border: 1px solid rgba(15,22,22,.055);
  pointer-events: none;
}
.fpc-square:focus-visible {
  z-index: 5;
  outline: 3px solid #fff;
  outline-offset: -3px;
}
.fpc-piece {
  position: relative;
  z-index: 2;
  display: grid;
  place-items: center;
  width: 100%;
  height: 100%;
  font: 700 clamp(15px, 4.25vw, 42px)/1 Georgia, 'Times New Roman', serif;
  -webkit-text-stroke: clamp(.35px, .075vw, 1px) rgba(0,0,0,.65);
  text-shadow: 0 2px 2px rgba(0,0,0,.45), 0 0 1px #000;
  pointer-events: none;
  will-change: transform, filter;
}
.fpc-piece-r { color: var(--fpc-red); }
.fpc-piece-b { color: var(--fpc-blue); }
.fpc-piece-y { color: var(--fpc-yellow); }
.fpc-piece-g { color: var(--fpc-green); }
.fpc-piece-dead { opacity: .34; filter: saturate(.25); }
.fpc-promoted {
  position: absolute;
  right: 4%;
  bottom: 5%;
  display: grid;
  place-items: center;
  width: 34%;
  aspect-ratio: 1;
  border-radius: 50%;
  color: #101615;
  background: #fff3bf;
  font: 800 clamp(6px, 1vw, 10px)/1 ui-monospace, monospace;
  -webkit-text-stroke: 0;
  box-shadow: 0 1px 2px #0007;
}
.fpc-last { box-shadow: inset 0 0 0 999px rgba(255,220,74,.22); }
.fpc-last-to { box-shadow: inset 0 0 0 3px rgba(255,235,130,.82), inset 0 0 0 999px rgba(255,220,74,.18); }
.fpc-origin { cursor: pointer; }
.fpc-origin::after {
  content: '';
  position: absolute;
  inset: 8%;
  border: 2px solid rgba(255,255,255,.32);
  pointer-events: none;
}
.fpc-selected { box-shadow: inset 0 0 0 4px #fff3a8, inset 0 0 0 999px rgba(255,211,72,.3); z-index: 3; }
.fpc-target, .fpc-capture { cursor: pointer; }
.fpc-target::after {
  content: '';
  position: absolute;
  z-index: 4;
  width: 27%;
  aspect-ratio: 1;
  border-radius: 50%;
  background: rgba(20,30,28,.43);
}
.fpc-capture::after {
  content: '';
  position: absolute;
  z-index: 4;
  inset: 7%;
  border: clamp(2px, .45vw, 5px) solid rgba(255,83,75,.72);
  border-radius: 50%;
}
.fpc-rail {
  position: absolute;
  z-index: 8;
  display: flex;
  align-items: center;
  min-height: 34px;
  gap: clamp(4px, .7vw, 9px);
  padding: 5px clamp(6px, 1vw, 11px);
  border: 1px solid color-mix(in srgb, currentColor 55%, transparent);
  background: color-mix(in srgb, var(--bg-inset) 92%, currentColor 8%);
  color: var(--rail-color);
  font: 720 clamp(.58rem, 1.35vw, .78rem)/1 ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: .06em;
  text-transform: uppercase;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.04), 0 5px 14px rgba(0,0,0,.16);
  transition: opacity .2s, box-shadow .2s, filter .2s;
}
.fpc-rail-0 { --rail-color: var(--fpc-red); }
.fpc-rail-1 { --rail-color: var(--fpc-blue); }
.fpc-rail-2 { --rail-color: var(--fpc-yellow); }
.fpc-rail-3 { --rail-color: var(--fpc-green); }
.fpc-rail-north, .fpc-rail-south { left: calc(3 / 14 * 100%); width: calc(8 / 14 * 100%); justify-content: space-between; }
.fpc-rail-north { top: 0; transform: translateY(calc(-100% - 5px)); }
.fpc-rail-south { bottom: 0; transform: translateY(calc(100% + 5px)); }
.fpc-rail-west, .fpc-rail-east {
  top: calc(3 / 14 * 100%);
  width: calc(8 / 14 * 100%);
  transform-origin: top left;
  justify-content: space-between;
}
.fpc-rail-west { left: -5px; transform: rotate(-90deg) translateX(-100%); }
.fpc-rail-east { left: calc(100% + 5px); transform: rotate(90deg); }
.fpc-pressure {
  width: 5px;
  align-self: stretch;
  background: currentColor;
  box-shadow: 0 0 9px currentColor;
  opacity: .42;
}
.fpc-army { display: flex; align-items: center; gap: 5px; white-space: nowrap; }
.fpc-army i { width: 8px; height: 8px; border-radius: 2px; background: currentColor; box-shadow: 0 0 8px currentColor; }
.fpc-you { padding: 3px 4px 2px; border: 1px solid currentColor; border-radius: 3px; font-size: .72em; }
.fpc-state { color: var(--text-dim); font-size: .82em; }
.fpc-score { display: inline-flex; align-items: baseline; gap: 3px; margin-left: auto; }
.fpc-score b { color: var(--text); font-size: 1.12em; }
.fpc-score small { color: var(--text-dim); font-size: .65em; }
.fpc-rail .seat-slot { min-width: 0; }
.fpc-rail.fpc-turn {
  box-shadow: inset 0 0 0 1px currentColor, 0 0 18px color-mix(in srgb, currentColor 43%, transparent);
}
.fpc-rail.fpc-turn .fpc-pressure { opacity: 1; }
.fpc-rail.fpc-turn .fpc-state { color: currentColor; }
.fpc-rail.fpc-check { animation: fpc-alert .8s ease-in-out infinite alternate; }
.fpc-rail.fpc-dead { opacity: .42; filter: grayscale(.7); }
.fpc-message {
  min-height: 1.3em;
  margin: 12px 0 -28px;
  text-align: center;
  color: var(--text-dim);
  font: 650 clamp(.72rem, 1.6vw, .9rem)/1.3 ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: .04em;
}
.fpc-fallback { white-space: pre-wrap; color: var(--text); }
.fpc-text-only .fpc-stage, .fpc-text-only .fpc-kicker, .fpc-text-only .fpc-message { display: none; }
@keyframes fpc-alert { to { filter: brightness(1.55); box-shadow: inset 0 0 0 2px currentColor, 0 0 21px currentColor; } }
@media (max-width: 620px) {
  .fpc-root { padding: 34px; }
  .fpc-kicker { margin-top: -24px; letter-spacing: .09em; }
  .fpc-rail { min-height: 27px; padding: 3px 5px; gap: 3px; }
  .fpc-state, .fpc-score small, .fpc-rail .seat-slot { display: none; }
  .fpc-pressure { width: 3px; }
  .fpc-army i { width: 6px; height: 6px; }
  .fpc-you { padding: 2px; }
  .fpc-message { margin-bottom: -24px; }
}
@media (prefers-reduced-motion: reduce) {
  .fpc-rail.fpc-check { animation: none; box-shadow: inset 0 0 0 2px currentColor; }
  .fpc-square, .fpc-rail { transition: none; }
}
`;
