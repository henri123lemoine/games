// Canonical Battlesnake frontend: two to four snakes choose simultaneously on
// the official 11×11 board. Multiple food and royale hazard cells are rendered
// directly from the game state; no partially committed move exists.
//
// View JSON (contract with games/snake/src/battlesnake/ui.rs):
//   { side: 11, coordinateSystem: "battlesnake", simultaneous: true,
//     snakes: [ { cells: [[x,y], ... head first], dir: "n|e|s|w",
//                 alive: bool, score: n, health: 0..=100 }, ... ],
//     food: [[x,y], ...], hazards: [[x,y], ...], turn: n,
//     outcome: "ongoing" | "win0" | "win1" | "draw" }
// The wire coordinates use Battlesnake's y-up convention and are converted to
// canvas y-down once at the boundary. Arrow keys / WASD / swipe choose the
// human move; the engine collects every opponent choice from that same state.
//
// Smoothness: a single requestAnimationFrame loop draws the board every frame,
// fully decoupled from bot computation in the worker. Between two consecutive
// authoritative states every snake glides linearly for one fixed cell period.
// Input changes the look direction immediately but never fabricates positions.

import type { MatchEventData, ViewState } from '../../engine/protocol';
import { sleep, type FrontendCtx, type GameFrontend } from '../types';

type Abs = 'n' | 'e' | 's' | 'w';

interface SnakeInfo {
  cells: [number, number][];
  dir: Abs;
  alive: boolean;
  score: number;
  health: number;
}

const MAX_HEALTH = 100;

interface BattlesnakeView {
  side: number;
  snakes: SnakeInfo[];
  food: [number, number][];
  hazards: [number, number][];
  turn: number;
  outcome: string;
}

const ABS_OF_KEY: Record<string, Abs> = {
  ArrowUp: 'n',
  ArrowRight: 'e',
  ArrowDown: 's',
  ArrowLeft: 'w',
  w: 'n',
  d: 'e',
  s: 's',
  a: 'w',
  W: 'n',
  D: 'e',
  S: 's',
  A: 'w',
};

const OPPOSITE: Record<Abs, Abs> = { n: 's', s: 'n', e: 'w', w: 'e' };
const DELTA: Record<Abs, [number, number]> = {
  n: [0, -1],
  e: [1, 0],
  s: [0, 1],
  w: [-1, 0],
};

/** The canonical action label the engine offers for each absolute heading. */
const LABEL_OF: Record<Abs, string> = { n: 'up', e: 'right', s: 'down', w: 'left' };

/** Defer submission one task so the shell has installed its input resolver. */
const SUBMIT_DELAY_MS = 0;
/** One authoritative grid step. Inputs are buffered throughout this interval
 * and sampled at its next boundary; render time and AI time never change it. */
const CELL_MS = 170;
const TURN_BUFFER_CAP = 2;

interface Palette {
  body: string; // mid band
  bodyHi: string; // lit band
  bodyLo: string; // shaded band
  head: string;
  rim: string; // dark outline under the tube
  glow: string; // soft outer glow
}

/** "You" runs emerald→mint, the bot electric blue→cyan — distinct hues with a
 * full rim/highlight/glow so each snake reads as a glossy 3D tube. */
const SEAT_PALETTES: Palette[] = [
  {
    body: '#21c46a',
    bodyHi: '#5cf0a0',
    bodyLo: '#127a43',
    head: '#9bffc7',
    rim: '#063a22',
    glow: 'rgba(45, 220, 120, 0.55)',
  },
  {
    body: '#3d8bff',
    bodyHi: '#7ec0ff',
    bodyLo: '#1f4fae',
    head: '#c4e2ff',
    rim: '#061634',
    glow: 'rgba(70, 150, 255, 0.55)',
  },
  {
    body: '#ef9f27',
    bodyHi: '#ffd37a',
    bodyLo: '#9a5510',
    head: '#ffe6a8',
    rim: '#3b2105',
    glow: 'rgba(255, 174, 55, 0.55)',
  },
  {
    body: '#b16cea',
    bodyHi: '#d8a7ff',
    bodyLo: '#67329b',
    head: '#ecd2ff',
    rim: '#26103a',
    glow: 'rgba(190, 105, 255, 0.55)',
  },
];

const SEAT_COLORS = ['Green', 'Blue', 'Amber', 'Violet'];

function asView(data: unknown): BattlesnakeView | null {
  if (!data || typeof data !== 'object') return null;
  const v = data as Partial<BattlesnakeView> & {
    coordinateSystem?: string;
    simultaneous?: boolean;
  };
  if (
    typeof v.side !== 'number' ||
    v.coordinateSystem !== 'battlesnake' ||
    v.simultaneous !== true ||
    !Array.isArray(v.snakes) ||
    v.snakes.length < 2 ||
    v.snakes.length > 4 ||
    !Array.isArray(v.food) ||
    !Array.isArray(v.hazards)
  ) return null;
  for (const s of v.snakes) {
    if (!s || !Array.isArray(s.cells) || (s.alive && s.cells.length === 0)) return null;
  }
  const flip = ([x, y]: [number, number]): [number, number] => [x, v.side! - 1 - y];
  return {
    side: v.side,
    snakes: v.snakes.map((snake) => ({
      ...snake,
      // Out-of-bounds deaths use a sentinel head cell in the engine. Never
      // feed it into interpolation, where it would yank the tube off screen.
      cells: snake.cells
        .filter(([x, y]) => x >= 0 && x < v.side! && y >= 0 && y < v.side!)
        .map(flip),
    })),
    food: v.food.map(flip),
    hazards: v.hazards.map(flip),
    turn: v.turn ?? 0,
    outcome: v.outcome ?? 'ongoing',
  };
}

function headMoved(a: BattlesnakeView | null, b: BattlesnakeView): boolean {
  if (!a) return false;
  for (let i = 0; i < b.snakes.length; i++) {
    if (a.snakes[i]?.alive !== b.snakes[i].alive) return true;
    const [ax, ay] = a.snakes[i].cells[0] ?? [];
    const [bx, by] = b.snakes[i].cells[0] ?? [];
    if (ax === undefined || ay === undefined || bx === undefined || by === undefined) continue;
    if (ax !== bx || ay !== by) return true;
  }
  return false;
}

const STYLE_ID = 'snake-frontend-style';
const CSS = `
.snk-root {
  align-self: center;
  width: min(100%, 560px);
  display: flex;
  flex-direction: column;
  gap: 14px;
  user-select: none;
}
.snk-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: stretch;
  gap: 10px;
}
.snk-chip {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 8px 13px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--bg-inset);
  color: var(--text-dim);
  font-size: 0.86rem;
  white-space: nowrap;
  transition: border-color 0.25s, box-shadow 0.25s, color 0.25s, opacity 0.25s;
}
.snk-chip.snk-dead {
  opacity: 0.42;
  filter: grayscale(0.5);
}
.snk-dot {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  flex: none;
}
.snk-chip-0 .snk-dot {
  background: radial-gradient(circle at 35% 30%, ${SEAT_PALETTES[0].bodyHi}, ${SEAT_PALETTES[0].body} 70%, ${SEAT_PALETTES[0].bodyLo});
  box-shadow: 0 0 9px ${SEAT_PALETTES[0].glow};
}
.snk-chip-1 .snk-dot {
  background: radial-gradient(circle at 35% 30%, ${SEAT_PALETTES[1].bodyHi}, ${SEAT_PALETTES[1].body} 70%, ${SEAT_PALETTES[1].bodyLo});
  box-shadow: 0 0 9px ${SEAT_PALETTES[1].glow};
}
.snk-chip .snk-len {
  margin-left: auto;
  font-variant-numeric: tabular-nums;
  color: var(--text);
  font-weight: 700;
}
.snk-hp {
  position: relative;
  flex: none;
  width: 48px;
  height: 6px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.12);
  overflow: hidden;
}
.snk-hp-fill {
  position: absolute;
  inset: 0 auto 0 0;
  width: 100%;
  border-radius: 999px;
  transition: width 0.2s linear, background-color 0.3s;
}
.snk-chip-0 .snk-hp-fill { background: ${SEAT_PALETTES[0].body}; }
.snk-chip-1 .snk-hp-fill { background: ${SEAT_PALETTES[1].body}; }
.snk-hp.snk-hp-low .snk-hp-fill { background: #f85149; }
.snk-stage {
  position: relative;
  aspect-ratio: 1 / 1;
  border-radius: 16px;
  overflow: hidden;
  background:
    radial-gradient(120% 100% at 50% 0%, #14304f 0%, #0a1c30 45%, #050d18 100%);
  border: 1px solid rgba(120, 180, 255, 0.14);
  box-shadow:
    inset 0 1px 0 rgba(180, 220, 255, 0.07),
    inset 0 0 60px rgba(0, 0, 0, 0.55),
    0 10px 30px rgba(0, 0, 0, 0.35);
}
.snk-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  display: block;
}
.snk-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  background: rgba(4, 10, 20, 0.62);
  backdrop-filter: blur(2px);
  color: #eaf2ff;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.35s;
  text-align: center;
  padding: 12px;
}
.snk-overlay.snk-show {
  opacity: 1;
}
.snk-overlay.snk-start-gate {
  transition: none;
}
.snk-overlay b {
  font-size: 1.6rem;
  letter-spacing: 0.03em;
}
.snk-overlay small {
  color: rgba(200, 215, 235, 0.8);
}
.snk-hint {
  text-align: center;
  color: var(--text-dim);
  font-size: 0.8rem;
  min-height: 1.1em;
}
.snk-debug {
  position: absolute;
  top: 8px;
  left: 8px;
  padding: 6px 9px;
  border-radius: 8px;
  background: rgba(4, 10, 20, 0.62);
  border: 1px solid rgba(120, 180, 255, 0.18);
  color: #d7e6ff;
  font-family: var(--mono);
  font-size: 11px;
  line-height: 1.5;
  letter-spacing: 0.02em;
  white-space: pre;
  pointer-events: none;
  display: none;
}
.snk-debug.snk-debug-on {
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

/** The functional test seam is opt-in: `?snakeDebug` in the URL, or a sticky
 * `snakeDebug` localStorage flag, so it never clutters normal play. */
function debugEnabled(): boolean {
  try {
    if (new URLSearchParams(window.location.search).has('snakeDebug')) return true;
    return window.localStorage.getItem('snakeDebug') === '1';
  } catch {
    return false;
  }
}

/** A glide between two discrete game states, run LINEARLY over `dur` ms so the
 * snake advances at constant velocity within the cell. `dur` is locked when the
 * glide starts (from the live cadence), so a cadence drift never changes the
 * speed of a cell already in motion. */
interface Glide {
  from: BattlesnakeView;
  to: BattlesnakeView;
  /** Authoritative resting state when `to` is a death-only visual pose. */
  commitTo?: BattlesnakeView;
  start: number;
  dur: number;
}

/** A short-lived visual flourish anchored to a board cell. */
interface Flash {
  x: number;
  y: number;
  born: number;
  dur: number;
  color: string;
}

/** A fading orb spat out when a snake dies. */
interface DeathOrb {
  x: number;
  y: number;
  vx: number;
  vy: number;
  born: number;
  color: string;
}

class SnakeFrontend implements GameFrontend {
  private ctx!: FrontendCtx;
  private canvas!: HTMLCanvasElement;
  private c2d!: CanvasRenderingContext2D;
  private chips: HTMLElement[] = [];
  private lenEls: HTMLElement[] = [];
  private hpEls: HTMLElement[] = [];
  private hpFillEls: HTMLElement[] = [];
  private overlayEl!: HTMLElement;
  private overlayTitleEl!: HTMLElement;
  private overlaySubEl!: HTMLElement;
  private hintEl!: HTMLElement;

  private view: BattlesnakeView | null = null;
  private glide: Glide | null = null;
  private side = 11;
  private cssSize = 0;
  private rafId = 0;
  private resizeObs: ResizeObserver | null = null;

  private pendingLabels: string[] | null = null;
  /** Preserve quick corner sequences, but never let a stale script build up. */
  private turnBuffer: Abs[] = [];
  private tickTimer = 0;
  private awaitingStart = false;
  private acknowledgedDir: Abs | null = null;
  private mySeat = -1;
  private wrapped = false;

  private foodPops = new Map<string, number>();
  private flashes: Flash[] = [];
  private deathOrbs: DeathOrb[] = [];
  private deadSeats = new Set<number>();
  private prevScores: number[] = [];

  // Opt-in functional test overlay.
  private debugEl!: HTMLElement;
  private showDebug = false;

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    this.mySeat = ctx.humanSeat;
    this.awaitingStart = this.mySeat >= 0;
    this.wrapped = ['wrapped', 'wrapped-constrictor'].includes(String(ctx.opts.mode ?? 'standard'));
    injectStyle();
    const chips = Array.from({ length: ctx.numSeats }, (_, seat) => `
          <div class="snk-chip snk-chip-${seat}">
            <span class="snk-dot" style="background:${SEAT_PALETTES[seat].body}"></span>
            <span class="seat-slot" data-seat="${seat}"></span>
            <span class="snk-hp"><span class="snk-hp-fill" style="background:${SEAT_PALETTES[seat].body}"></span></span>
            <span class="snk-len">3</span>
          </div>`).join('');
    host.innerHTML = `
      <div class="snk-root">
        <div class="snk-bar">${chips}</div>
        <div class="snk-stage">
          <canvas class="snk-canvas"></canvas>
          <div class="snk-debug"></div>
          <div class="snk-overlay"><b></b><small></small></div>
        </div>
        <div class="snk-hint"></div>
      </div>`;
    this.canvas = host.querySelector('.snk-canvas')!;
    this.c2d = this.canvas.getContext('2d')!;
    this.chips = Array.from(host.querySelectorAll<HTMLElement>('.snk-chip'));
    this.lenEls = this.chips.map((chip) => chip.querySelector<HTMLElement>('.snk-len')!);
    this.hpEls = this.chips.map((chip) => chip.querySelector<HTMLElement>('.snk-hp')!);
    this.hpFillEls = this.chips.map((chip) => chip.querySelector<HTMLElement>('.snk-hp-fill')!);
    this.prevScores = Array(ctx.numSeats).fill(0);
    this.overlayEl = host.querySelector('.snk-overlay')!;
    this.overlayTitleEl = this.overlayEl.querySelector('b')!;
    this.overlaySubEl = this.overlayEl.querySelector('small')!;
    this.hintEl = host.querySelector('.snk-hint')!;
    this.debugEl = host.querySelector('.snk-debug')!;
    this.showDebug = debugEnabled();
    this.debugEl.classList.toggle('snk-debug-on', this.showDebug);

    const stage = host.querySelector<HTMLElement>('.snk-stage')!;
    if (this.mySeat >= 0) {
      // Capture before page scrolling or another shell handler can eat arrows.
      window.addEventListener('keydown', this.onKey, true);
      stage.addEventListener('touchstart', this.onTouchStart, { passive: true });
      stage.addEventListener('touchmove', this.onTouchMove, { passive: false });
      stage.addEventListener('touchend', this.onTouchEnd);
      this.hintEl.textContent = 'Choose a direction to start · arrow keys / WASD / swipe';
    } else {
      this.hintEl.textContent = 'Watching the bots play';
    }

    this.resizeObs = new ResizeObserver(() => this.resize(stage));
    this.resizeObs.observe(stage);
    this.resize(stage);
    this.loop(performance.now());
  }

  render(state: ViewState): void {
    const view = asView(state.viewData);
    if (!view) return;
    this.side = view.side;
    // Snap to this state only when no glide owns the display (the human's-turn /
    // final redraw path, which doesn't go through animate()).
    if (!this.glide) this.view = view;
    this.syncJuice(view);
    this.updateBar(view, state);
    this.updateOverlay(view, state);
  }

  async animate(event: MatchEventData, after: ViewState): Promise<void> {
    const next = asView(after.viewData);
    if (!next) return;
    const prev = this.latestKnown();
    this.syncJuice(next);
    this.updateBar(next, after);
    this.updateOverlay(next, after);
    const scale = this.ctx.animationScale();
    this.side = next.side;

    // Motion disabled (reduced-motion / instant speed): snap to the final state.
    if (scale <= 0) {
      this.view = next;
      this.glide = null;
      this.acknowledgedDir = this.turnBuffer.at(-1) ?? null;
      await sleep(CELL_MS);
      return;
    }

    // A chance-only food update has no motion of its own.
    if (!prev || !headMoved(prev, next)) {
      this.acknowledgedDir = this.turnBuffer.at(-1) ?? null;
      this.view = next;
      this.glide = null;
      return;
    }

    // One clock owns movement. Every snake interpolates between the same two
    // authoritative states for exactly one cell period; input may queue during
    // this glide, but it never invents another position for the renderer.
    const start = performance.now();
    this.acknowledgedDir = this.turnBuffer.at(-1) ?? null;
    let visualTo = next;
    let commitTo: BattlesnakeView | undefined;
    const moves = transitionDirections(event.data);
    for (let seat = 0; seat < next.snakes.length; seat++) {
      if (!prev.snakes[seat]?.alive || next.snakes[seat].alive) continue;
      const direction = moves[seat] ?? next.snakes[seat].dir;
      const deathPose = predictSnakeMove(prev, seat, direction, this.wrapped);
      if (visualTo === next) visualTo = cloneView(next);
      visualTo.snakes[seat] = {
        ...visualTo.snakes[seat],
        cells: deathPose.snakes[seat].cells,
        dir: direction,
      };
      commitTo = next;
      if (seat === this.mySeat) this.acknowledgedDir = null;
    }
    this.glide = {
      from: prev,
      to: visualTo,
      commitTo,
      start,
      dur: CELL_MS * scale,
    };
    await sleep(this.glide.dur);
    this.advanceGlide(performance.now());
  }

  /** The most recent discrete state the frontend knows about — the active
   * glide's target, or the committed view. New moves compare against this to
   * decide if the head actually moved. */
  private latestKnown(): BattlesnakeView | null {
    if (this.glide) return this.glide.to;
    return this.view;
  }

  promptAction(labels: string[]): void {
    this.pendingLabels = labels;
    if (this.mySeat < 0) return;
    // Turn zero waits for an intentional direction. A random opening can never
    // move before the player has found their snake and chosen where to go.
    if (this.awaitingStart) return;
    this.armTick();
  }

  private armTick(): void {
    if (this.tickTimer || !this.pendingLabels || this.awaitingStart) return;
    this.tickTimer = window.setTimeout(() => {
      this.tickTimer = 0;
      this.fireTick();
    }, SUBMIT_DELAY_MS);
  }

  unmount(): void {
    cancelAnimationFrame(this.rafId);
    if (this.tickTimer) clearTimeout(this.tickTimer);
    this.tickTimer = 0;
    window.removeEventListener('keydown', this.onKey, true);
    this.resizeObs?.disconnect();
    this.resizeObs = null;
  }

  /** Submit the next buffered turn, or continue straight. */
  private fireTick(): void {
    if (!this.pendingLabels || this.mySeat < 0) return;
    const cur = this.currentHeading();
    const pressed = this.turnBuffer.shift() ?? null;
    const want = pressed ?? cur;
    this.acknowledgedDir = this.turnBuffer.at(-1) ?? (pressed ? want : null);
    const label = LABEL_OF[want];
    const i = this.pendingLabels.indexOf(label);
    const labels = this.pendingLabels;
    this.pendingLabels = null;
    const fallback = labels.indexOf(LABEL_OF[cur]);
    this.ctx.submit(String(i >= 0 ? i : fallback >= 0 ? fallback : 0));
  }

  /** The seat's CURRENT heading — read from the latest committed state (the
   * glide's target while a glide is in flight), not the resting `this.view`,
   * which lags behind during a glide. Using the stale view here would wrongly
   * judge a just-pressed legal turn as a 180° reversal and drop it. */
  private currentHeading(): Abs {
    const v = this.glide ? this.glide.to : this.view;
    return v?.snakes[this.mySeat]?.dir ?? 'e';
  }

  private onKey = (e: KeyboardEvent): void => {
    if (this.mySeat < 0 || e.metaKey || e.ctrlKey || e.altKey) return;
    const t = e.target as HTMLElement | null;
    if (
      t &&
      (t.tagName === 'INPUT' ||
        t.tagName === 'TEXTAREA' ||
        t.tagName === 'SELECT' ||
        t.tagName === 'BUTTON' ||
        t.isContentEditable)
    ) return;
    const abs = ABS_OF_KEY[e.key];
    if (!abs) return;
    e.preventDefault();
    this.steer(abs);
  };

  private touchStart: { x: number; y: number } | null = null;

  private onTouchStart = (e: TouchEvent): void => {
    const t = e.changedTouches[0];
    this.touchStart = { x: t.clientX, y: t.clientY };
  };

  private onTouchMove = (e: TouchEvent): void => {
    if (this.touchStart) e.preventDefault();
  };

  private onTouchEnd = (e: TouchEvent): void => {
    if (!this.touchStart) return;
    const t = e.changedTouches[0];
    const dx = t.clientX - this.touchStart.x;
    const dy = t.clientY - this.touchStart.y;
    this.touchStart = null;
    if (Math.max(Math.abs(dx), Math.abs(dy)) < 22) return;
    this.steer(
      Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 'e' : 'w') : dy > 0 ? 's' : 'n',
    );
  };

  /** Buffer up to two legal turns. Validation follows the last queued turn, so
   * Right→Down pressed between ticks survives as two crisp consecutive moves. */
  private steer(abs: Abs): void {
    if (this.mySeat < 0) return;
    const previous = this.turnBuffer.at(-1) ?? this.currentHeading();
    if (abs === previous) {
      // Pressing the already-facing direction is still an intentional start.
      if (this.awaitingStart) {
        this.turnBuffer.push(abs);
        this.acknowledgeInput(abs);
        this.beginHumanPlay();
      }
      return;
    }
    // A stacked opening body has no neck yet, so all four first directions are
    // legal. After that, reject a 180° turn against the committed/queued neck.
    const snake = this.latestKnown()?.snakes[this.mySeat];
    const stacked =
      !!snake &&
      snake.cells.length > 0 &&
      snake.cells.every(([x, y]) => x === snake.cells[0][0] && y === snake.cells[0][1]);
    if (!(stacked && this.turnBuffer.length === 0) && abs === OPPOSITE[previous]) return;
    if (this.turnBuffer.length >= TURN_BUFFER_CAP) return;
    this.turnBuffer.push(abs);
    this.acknowledgeInput(abs);
    if (this.awaitingStart) this.beginHumanPlay();
    this.armTick();
  }

  private beginHumanPlay(): void {
    this.awaitingStart = false;
    this.hintEl.textContent = 'Arrow keys / WASD / swipe to steer';
    this.updateStartOverlay();
    this.armTick();
  }

  private acknowledgeInput(dir: Abs): void {
    this.acknowledgedDir = dir;
  }

  private updateBar(view: BattlesnakeView, _state: ViewState): void {
    for (let seat = 0; seat < view.snakes.length; seat++) {
      const s = view.snakes[seat];
      this.lenEls[seat].textContent = String(s.score);
      const hp = Math.max(0, Math.min(MAX_HEALTH, s.health ?? MAX_HEALTH));
      const pct = s.alive ? (hp / MAX_HEALTH) * 100 : 0;
      this.hpFillEls[seat].style.width = `${pct}%`;
      this.hpEls[seat].classList.toggle('snk-hp-low', s.alive && hp <= 25);
      this.hpEls[seat].title = `health ${hp}`;
      this.chips[seat].classList.toggle('snk-dead', !s.alive);
      // No whose-turn highlight: snake is real-time (both snakes move every
      // tick), so a flashing "your turn" chip is meaningless and distracting.
    }
  }

  private updateOverlay(view: BattlesnakeView, state: ViewState): void {
    if (!state.isOver) {
      this.updateStartOverlay();
      return;
    }
    let title = 'Draw';
    const winner = /^win(\d+)$/.exec(view.outcome)?.[1];
    if (winner !== undefined) title = `${SEAT_COLORS[Number(winner)]} wins`;
    if (this.mySeat >= 0 && view.outcome !== 'draw') {
      const won = view.outcome === `win${this.mySeat}`;
      title = won ? 'You win!' : 'You lose';
    }
    this.overlayTitleEl.textContent = title;
    this.overlaySubEl.textContent = `${view.snakes.map((snake, seat) => `${SEAT_COLORS[seat]} ${snake.score}`).join(' · ')} · turn ${view.turn}`;
    this.overlayEl.classList.add('snk-show');
  }

  private updateStartOverlay(): void {
    if (!this.awaitingStart || this.mySeat < 0) {
      this.overlayEl.classList.remove('snk-show');
      if (this.overlayEl.classList.contains('snk-start-gate')) {
        requestAnimationFrame(() => this.overlayEl.classList.remove('snk-start-gate'));
      }
      return;
    }
    this.overlayTitleEl.textContent = 'Choose your first move';
    this.overlaySubEl.textContent = 'Arrow keys, WASD, or swipe to start';
    this.overlayEl.classList.add('snk-start-gate');
    this.overlayEl.classList.add('snk-show');
  }

  /** Fire the small flourishes off a fresh state: food respawn pop, an eat
   * ring when a score ticks up, and a death burst when a snake dies. */
  private syncJuice(view: BattlesnakeView): void {
    const now = performance.now();
    const currentFood = new Set(view.food.map(([x, y]) => `${x},${y}`));
    for (const key of currentFood) {
      if (!this.foodPops.has(key)) this.foodPops.set(key, now);
    }
    for (const key of this.foodPops.keys()) {
      if (!currentFood.has(key)) this.foodPops.delete(key);
    }
    for (let seat = 0; seat < view.snakes.length; seat++) {
      const s = view.snakes[seat];
      if (s.score > this.prevScores[seat]) {
        const [hx, hy] = s.cells[0];
        this.flashes.push({
          x: hx,
          y: hy,
          born: now,
          dur: 360,
          color: SEAT_PALETTES[seat].bodyHi,
        });
      }
      this.prevScores[seat] = s.score;
      if (!s.alive && !this.deadSeats.has(seat)) {
        this.deadSeats.add(seat);
        this.spawnDeath(seat, s);
      }
      if (s.alive) this.deadSeats.delete(seat);
    }
  }

  private spawnDeath(seat: number, s: SnakeInfo): void {
    const now = performance.now();
    const pal = SEAT_PALETTES[seat];
    const step = Math.max(1, Math.floor(s.cells.length / 22));
    for (let i = 0; i < s.cells.length; i += step) {
      const [cx, cy] = s.cells[i];
      const ang = Math.random() * Math.PI * 2;
      const spd = 0.4 + Math.random() * 1.4;
      this.deathOrbs.push({
        x: cx,
        y: cy,
        vx: Math.cos(ang) * spd,
        vy: Math.sin(ang) * spd,
        born: now + i * 4,
        color: (i & 4) === 0 ? pal.bodyHi : pal.body,
      });
    }
  }

  private resize(stage: HTMLElement): void {
    const rect = stage.getBoundingClientRect();
    const size = Math.max(1, Math.round(Math.min(rect.width, rect.height)));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.cssSize = size;
    this.canvas.width = size * dpr;
    this.canvas.height = size * dpr;
    this.c2d.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  private loop = (now: number): void => {
    this.draw(now);
    if (this.showDebug) this.paintDebug();
    this.rafId = requestAnimationFrame(this.loop);
  };

  private paintDebug(): void {
    const text = `turn  ${this.view?.turn ?? 0}`;
    if (this.debugEl.textContent !== text) this.debugEl.textContent = text;
  }

  private draw(now: number): void {
    const ctx = this.c2d;
    const size = this.cssSize;
    if (size <= 0) return;
    const cell = size / this.side;
    ctx.clearRect(0, 0, size, size);
    this.drawGrid(cell);

    this.advanceGlide(now);

    const view = this.glide ? this.glide.to : this.view;
    if (view) {
      this.drawHazards(view.hazards, cell);
      for (const food of view.food) this.drawFood(food, cell, now);
    }

    if (view) {
      for (let seat = 0; seat < view.snakes.length; seat++) {
        this.drawSnake(seat, this.glideProgress(now), cell);
      }
    }

    this.drawFlashes(cell, now);
    this.drawDeathOrbs(cell, now);
  }

  /** Commit the authoritative target at the fixed cell boundary. */
  private advanceGlide(now: number): void {
    if (this.glide && now - this.glide.start >= this.glide.dur) {
      this.view = this.glide.commitTo ?? this.glide.to;
      this.glide = null;
    }
  }

  private glideProgress(now: number): number {
    if (!this.glide) return 1;
    return Math.min(1, (now - this.glide.start) / this.glide.dur);
  }

  private drawGrid(cell: number): void {
    const ctx = this.c2d;
    ctx.save();
    ctx.strokeStyle = 'rgba(130, 190, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 1; i < this.side; i++) {
      const p = Math.round(i * cell) + 0.5;
      ctx.moveTo(p, 0);
      ctx.lineTo(p, this.cssSize);
      ctx.moveTo(0, p);
      ctx.lineTo(this.cssSize, p);
    }
    ctx.stroke();
    ctx.restore();
  }

  /** Royale hazards read as a closing electric storm rather than another game
   * piece: a translucent violet field with a directional hatch. */
  private drawHazards(hazards: [number, number][], cell: number): void {
    if (hazards.length === 0) return;
    const ctx = this.c2d;
    ctx.save();
    for (const [x, y] of hazards) {
      const left = x * cell;
      const top = y * cell;
      const grad = ctx.createRadialGradient(
        left + cell * 0.5,
        top + cell * 0.5,
        0,
        left + cell * 0.5,
        top + cell * 0.5,
        cell * 0.8,
      );
      grad.addColorStop(0, 'rgba(173, 100, 255, 0.26)');
      grad.addColorStop(1, 'rgba(82, 27, 126, 0.42)');
      ctx.fillStyle = grad;
      ctx.fillRect(left, top, cell, cell);
      ctx.strokeStyle = 'rgba(225, 184, 255, 0.22)';
      ctx.lineWidth = Math.max(1, cell * 0.04);
      ctx.beginPath();
      ctx.moveTo(left, top + cell * 0.72);
      ctx.lineTo(left + cell * 0.72, top);
      ctx.moveTo(left + cell * 0.28, top + cell);
      ctx.lineTo(left + cell, top + cell * 0.28);
      ctx.stroke();
    }
    ctx.restore();
  }

  private drawFood(food: [number, number], cell: number, now: number): void {
    const ctx = this.c2d;
    const cx = (food[0] + 0.5) * cell;
    const cy = (food[1] + 0.5) * cell;
    const age = (now - (this.foodPops.get(`${food[0]},${food[1]}`) ?? now)) / 240;
    const pop = age < 1 ? 0.55 + 0.45 * easeOut(age) : 1;
    const breathe = 1 + 0.07 * Math.sin(now / 360);
    const r = cell * 0.32 * pop * breathe;

    ctx.save();
    // Soft outer halo.
    const halo = ctx.createRadialGradient(cx, cy, r * 0.4, cx, cy, r * 2.6);
    halo.addColorStop(0, 'rgba(255, 120, 100, 0.4)');
    halo.addColorStop(1, 'rgba(255, 80, 70, 0)');
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(cx, cy, r * 2.6, 0, Math.PI * 2);
    ctx.fill();

    // The orb: white-hot core fading to a warm red rim.
    const grad = ctx.createRadialGradient(cx - r * 0.32, cy - r * 0.34, r * 0.1, cx, cy, r);
    grad.addColorStop(0, '#fff2ec');
    grad.addColorStop(0.4, '#ff9d86');
    grad.addColorStop(1, '#f0463c');
    ctx.fillStyle = grad;
    ctx.shadowColor = 'rgba(248, 81, 73, 0.85)';
    ctx.shadowBlur = cell * 0.7;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();

    // Two orbiting sparkles.
    ctx.shadowBlur = 0;
    ctx.fillStyle = 'rgba(255, 240, 220, 0.9)';
    for (let k = 0; k < 2; k++) {
      const a = now / 600 + k * Math.PI;
      const sx = cx + Math.cos(a) * r * 1.5;
      const sy = cy + Math.sin(a) * r * 1.5;
      ctx.beginPath();
      ctx.arc(sx, sy, cell * 0.045, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  /** Draw one snake as a glossy capsule tube: a dark rim, the gradient body
   * banded along the spine, a glossy highlight, then the head with eyes. The
   * spine is interpolated by `t` so the body glides one cell per tick. */
  private drawSnake(seat: number, t: number, cell: number): void {
    const pal = SEAT_PALETTES[seat];
    const to = (this.glide ? this.glide.to : this.view!).snakes[seat];
    const from = this.glide ? this.glide.from.snakes[seat] : to;
    if (!to.alive && !this.glide) return;
    const motionDir = to.dir;
    const visibleDir =
      seat === this.mySeat
        ? (this.acknowledgedDir ?? motionDir)
        : to.dir;
    // LINEAR in t (no easing): constant velocity within a cell, so combined with
    // the fixed cellMs-per-cell cadence the snake glides at a uniform speed.
    const interpolated = interpBody(from.cells, to.cells, t, this.side, this.wrapped);
    const cells = this.wrapped ? unwrapBody(interpolated, this.side) : interpolated;
    if (cells.length === 0) return;

    // Opt-in (?snakeDebug) test seam: publish each snake's interpolated head
    // (board CELL coords), heading, and alive flag every frame, so the validation
    // harness can read the exact rendered heads — far cleaner than colour-
    // tracking the canvas through the juice. Seat 0 verifies input-following;
    // seat 1 verifies the bot actually plays (doesn't suicide).
    if (this.showDebug) {
      const key = seat === 0 ? '__snakeHead0' : '__snakeHead1';
      const debugWindow = window as unknown as Record<string, unknown>;
      debugWindow[key] = {
        t: performance.now(),
        x: this.wrapped ? wrapCoordinate(cells[0][0], this.side) : cells[0][0],
        y: this.wrapped ? wrapCoordinate(cells[0][1], this.side) : cells[0][1],
        maxLink: cells.slice(1).reduce((largest, cell, i) => {
          const previous = cells[i];
          return Math.max(largest, Math.hypot(cell[0] - previous[0], cell[1] - previous[1]));
        }, 0),
        dir: motionDir,
        lookDir: visibleDir,
        alive: to.alive,
        len: to.cells.length,
      };
    }

    // Center-of-cell points; the tube width tapers slightly toward the tail.
    const pts = cells.map(([x, y]) => [(x + 0.5) * cell, (y + 0.5) * cell] as [number, number]);
    const baseW = cell * 0.74;

    // A torus has no privileged seam. Draw the continuous unwrapped tube in
    // every periodic copy that intersects the canvas, so a head exits one edge
    // and enters the other without a board-wide bridge.
    const shifts = this.wrapped
      ? periodicPixelOffsets(cells, this.side, this.cssSize)
      : [[0, 0] as [number, number]];
    for (const [ox, oy] of shifts) {
      const shifted = pts.map(([x, y]) => [x + ox, y + oy] as [number, number]);
      this.drawSnakeCopy(shifted, visibleDir, cell, baseW, pal, to.alive);
    }
  }

  private drawSnakeCopy(
    pts: [number, number][],
    dir: Abs,
    cell: number,
    baseW: number,
    pal: Palette,
    alive: boolean,
  ): void {
    const ctx = this.c2d;

    ctx.save();
    ctx.globalAlpha = alive ? 1 : 0.4;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    // Outer glow under everything (skipped when dead — it's dissolving).
    if (alive) {
      ctx.save();
      ctx.shadowColor = pal.glow;
      ctx.shadowBlur = cell * 0.6;
      ctx.strokeStyle = pal.glow;
      ctx.lineWidth = baseW;
      strokeTube(ctx, pts, 1);
      ctx.restore();
    }

    // Dark rim, a touch wider than the body.
    ctx.strokeStyle = pal.rim;
    ctx.lineWidth = baseW + Math.max(1.5, cell * 0.1);
    strokeTube(ctx, pts, 1);

    // Body: a head→tail gradient between the lit and shaded tones, tapering.
    if (pts.length >= 2) {
      const grad = ctx.createLinearGradient(pts[0][0], pts[0][1], pts[pts.length - 1][0], pts[pts.length - 1][1]);
      grad.addColorStop(0, pal.bodyHi);
      grad.addColorStop(0.5, pal.body);
      grad.addColorStop(1, pal.bodyLo);
      ctx.strokeStyle = grad;
    } else {
      ctx.strokeStyle = pal.body;
    }
    ctx.lineWidth = baseW;
    strokeTube(ctx, pts, 0.82); // taper the tail

    // Glossy highlight riding the top of the tube.
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
    ctx.lineWidth = Math.max(1, baseW * 0.3);
    strokeTube(ctx, pts, 1, -baseW * 0.22);

    ctx.restore();

    this.drawHead(pts[0], dir, cell, pal, alive);
  }

  private drawHead(
    head: [number, number],
    dir: Abs,
    cell: number,
    pal: Palette,
    alive: boolean,
  ): void {
    const ctx = this.c2d;
    const [hx, hy] = head;
    const r = cell * 0.42;
    const [dx, dy] = DELTA[dir];

    ctx.save();
    ctx.globalAlpha = alive ? 1 : 0.4;
    // Rounded head cap with a glossy radial sheen toward the light.
    ctx.fillStyle = pal.rim;
    ctx.beginPath();
    ctx.arc(hx, hy, r + Math.max(1, cell * 0.05), 0, Math.PI * 2);
    ctx.fill();
    const grad = ctx.createRadialGradient(hx - r * 0.34, hy - r * 0.4, r * 0.1, hx, hy, r);
    grad.addColorStop(0, pal.head);
    grad.addColorStop(0.55, pal.body);
    grad.addColorStop(1, pal.bodyLo);
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(hx, hy, r, 0, Math.PI * 2);
    ctx.fill();

    // Eyes: set on the head's forward-sides, pupils looking the travel way.
    const fwd = cell * 0.14;
    const side = cell * 0.18;
    const eyeR = cell * 0.13;
    const pupilR = cell * 0.07;
    for (const s of [-1, 1]) {
      const ex = hx + dx * fwd + dy * side * s;
      const ey = hy + dy * fwd + dx * side * s;
      ctx.fillStyle = '#f4f9ff';
      ctx.beginPath();
      ctx.arc(ex, ey, eyeR, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#0a1424';
      ctx.beginPath();
      ctx.arc(ex + dx * eyeR * 0.4, ey + dy * eyeR * 0.4, pupilR, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  private drawFlashes(cell: number, now: number): void {
    if (this.flashes.length === 0) return;
    const ctx = this.c2d;
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    this.flashes = this.flashes.filter((f) => {
      const age = (now - f.born) / f.dur;
      if (age >= 1) return false;
      const cx = (f.x + 0.5) * cell;
      const cy = (f.y + 0.5) * cell;
      const r = cell * (0.3 + 1.1 * easeOut(age));
      ctx.globalAlpha = (1 - age) * 0.7;
      ctx.strokeStyle = f.color;
      ctx.lineWidth = cell * 0.12 * (1 - age);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
      return true;
    });
    ctx.restore();
  }

  private drawDeathOrbs(cell: number, now: number): void {
    if (this.deathOrbs.length === 0) return;
    const ctx = this.c2d;
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    this.deathOrbs = this.deathOrbs.filter((o) => {
      const age = now - o.born;
      if (age < 0) return true;
      const life = age / 720;
      if (life >= 1) return false;
      const cx = (o.x + 0.5 + o.vx * life) * cell;
      const cy = (o.y + 0.5 + o.vy * life) * cell;
      const r = cell * 0.3 * (1 - life * 0.5);
      ctx.globalAlpha = (1 - life) * 0.85;
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 2);
      g.addColorStop(0, '#ffffff');
      g.addColorStop(0.4, o.color);
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 2, 0, Math.PI * 2);
      ctx.fill();
      return true;
    });
    ctx.restore();
  }
}

function cloneView(view: BattlesnakeView): BattlesnakeView {
  return {
    ...view,
    snakes: view.snakes.map((snake) => ({
      ...snake,
      cells: snake.cells.map(([x, y]) => [x, y]),
    })),
    food: view.food.map(([x, y]) => [x, y]),
    hazards: view.hazards.map(([x, y]) => [x, y]),
  };
}

/** Read the resolved joint action so a dead snake can finish its final real
 * step locally instead of interpolating toward the engine's off-board sentinel. */
function transitionDirections(data: unknown): (Abs | undefined)[] {
  if (!data || typeof data !== 'object') return [];
  const moves = (data as { moves?: unknown }).moves;
  if (!Array.isArray(moves)) return [];
  const directions: Record<string, Abs> = {
    up: 'n',
    right: 'e',
    down: 's',
    left: 'w',
  };
  return moves.map((move) => (typeof move === 'string' ? directions[move] : undefined));
}

/** Reconstruct one already-resolved displacement for the death animation.
 * Collision, health, hazards, food, and alive state remain engine-owned. */
function predictSnakeMove(
  view: BattlesnakeView,
  seat: number,
  dir: Abs,
  wrapped: boolean,
): BattlesnakeView {
  const out = cloneView(view);
  const snake = out.snakes[seat];
  const [dx, dy] = DELTA[dir];
  let x = snake.cells[0][0] + dx;
  let y = snake.cells[0][1] + dy;
  if (wrapped) {
    x = wrapCoordinate(x, view.side);
    y = wrapCoordinate(y, view.side);
  }
  const cells: [number, number][] = [[x, y], ...snake.cells.slice(0, -1)];
  const ate =
    x >= 0 &&
    x < view.side &&
    y >= 0 &&
    y < view.side &&
    view.food.some(([fx, fy]) => fx === x && fy === y);
  if (ate && cells.length > 0) {
    const tail = cells[cells.length - 1];
    cells.push([tail[0], tail[1]]);
  }
  out.snakes[seat] = {
    ...snake,
    cells,
    dir,
    score: cells.length,
  };
  return out;
}

/** Interpolate a body's segments from their previous to current positions.
 * On a non-eating tick the body lengths match and segment i tweens from
 * from[i] to to[i]; on an eating tick the body grew by one, so the new head
 * tweens out of the old head's cell and the rest shift down by one. */
function interpBody(
  from: [number, number][],
  to: [number, number][],
  t: number,
  side: number,
  wrapped: boolean,
): [number, number][] {
  const out: [number, number][] = [];
  const grew = to.length > from.length;
  for (let i = 0; i < to.length; i++) {
    const a = grew ? from[Math.max(0, i - 1)] : from[Math.min(i, from.length - 1)];
    const b = to[i];
    const bx = wrapped ? nearestPeriodic(b[0], a[0], side) : b[0];
    const by = wrapped ? nearestPeriodic(b[1], a[1], side) : b[1];
    out.push([lerp(a[0], bx, t), lerp(a[1], by, t)]);
  }
  return out;
}

/** Put every segment in the periodic image nearest its predecessor. This turns
 * a body stored as `[0, …, 10]` across an 11-wide seam into one continuous
 * local path such as `[0, …, -1]`, never a line spanning the board. */
function unwrapBody(cells: [number, number][], side: number): [number, number][] {
  if (cells.length === 0) return [];
  const out: [number, number][] = [[cells[0][0], cells[0][1]]];
  for (let i = 1; i < cells.length; i++) {
    const previous = out[i - 1];
    out.push([
      nearestPeriodic(cells[i][0], previous[0], side),
      nearestPeriodic(cells[i][1], previous[1], side),
    ]);
  }
  return out;
}

function nearestPeriodic(value: number, anchor: number, side: number): number {
  return value + Math.round((anchor - value) / side) * side;
}

function wrapCoordinate(value: number, side: number): number {
  return ((value % side) + side) % side;
}

/** Pixel translations for every periodic copy of an unwrapped body that can
 * touch the visible board. Usually this is one copy, or two at a seam. */
function periodicPixelOffsets(
  cells: [number, number][],
  side: number,
  size: number,
): [number, number][] {
  const xs = cells.map(([x]) => x);
  const ys = cells.map(([, y]) => y);
  const axis = (min: number, max: number): number[] => {
    const out: number[] = [];
    const lo = Math.floor((-max - 1) / side);
    const hi = Math.ceil((side - min + 1) / side);
    for (let k = lo; k <= hi; k++) {
      if (max + k * side >= -1 && min + k * side <= side) out.push(k * size);
    }
    return out;
  };
  const xOffsets = axis(Math.min(...xs), Math.max(...xs));
  const yOffsets = axis(Math.min(...ys), Math.max(...ys));
  const out: [number, number][] = [];
  for (const x of xOffsets) for (const y of yOffsets) out.push([x, y]);
  return out;
}

/** Stroke a tube along `pts` (head→tail). `taper` scales the line width down
 * to the tail; `perp` offsets the path perpendicular to local heading (for the
 * gloss highlight). The current `ctx.lineWidth` is the head-end width. */
function strokeTube(
  ctx: CanvasRenderingContext2D,
  pts: [number, number][],
  taper: number,
  perp = 0,
): void {
  if (pts.length === 0) return;
  if (pts.length === 1) {
    ctx.beginPath();
    ctx.arc(pts[0][0], pts[0][1], ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = ctx.strokeStyle as string;
    ctx.fill();
    return;
  }
  if (perp === 0 && taper >= 1) {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
    return;
  }
  // Tapering / offset: draw as short round-capped segments whose width shrinks
  // toward the tail so the join blends; cheap and reads as one smooth tube.
  const headW = ctx.lineWidth;
  for (let i = 0; i < pts.length - 1; i++) {
    const f = i / (pts.length - 1);
    const w = headW * (1 - (1 - taper) * f);
    let [ax, ay] = pts[i];
    let [bx, by] = pts[i + 1];
    if (perp !== 0) {
      const dx = bx - ax;
      const dy = by - ay;
      const len = Math.hypot(dx, dy) || 1;
      const nx = (-dy / len) * perp;
      const ny = (dx / len) * perp;
      ax += nx;
      ay += ny;
      bx += nx;
      by += ny;
    }
    ctx.lineWidth = w;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function easeOut(t: number): number {
  return 1 - (1 - t) * (1 - t);
}

export function createSnakeFrontend(): GameFrontend {
  return new SnakeFrontend();
}
