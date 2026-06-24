// Snake frontend: two snakes race on a 20×20 board, rendered to read like a
// polished arcade game — rounded capsule bodies with a gradient sheen and eyes,
// a board with a soft vignette and inner glow, glowing food orbs, and eat/turn/
// death flourishes.
//
// View JSON (contract with games/snake/src/duel_ui.rs):
//   { side: 20,
//     snakes: [ { cells: [[x,y], ... head first], dir: "n|e|s|w",
//                 alive: bool, score: n, health: 0..=100 }, { ... } ],
//     food: [x,y] | null,
//     step: n, cap: n,
//     outcome: "ongoing" | "win0" | "win1" | "draw" }
// `x` grows rightward, `y` downward. snakes[0] is Snake A (seat 0), snakes[1]
// Snake B (seat 1). `health` drains by one each tick and refills to 100 on a
// meal; a snake that hits 0 starves, so the bar doubles as a "find food now"
// pressure gauge.
//
// The Duel game is turn-based under the hood (seat 0 commits, then seat 1
// commits seeing it, then both advance), but play is REAL-TIME here: on the
// human's turn this frontend auto-submits the snake's current heading on a
// fixed clock, so the snake never stalls. Arrow keys / WASD / swipe queue a
// turn (180° reversals are dropped, as in classic snake); the queued turn is
// consumed on the next tick. Watch mode just animates the bots' moves.
//
// Smoothness: a single requestAnimationFrame loop draws the board every frame,
// fully decoupled from the bot's move computation (which runs in the worker /
// on the GPU). Between two discrete game states the snakes GLIDE — each segment
// eases from its previous cell to its next over the real wall-clock interval
// between ticks, so even when the bot's think runs long the snakes keep sliding
// toward their target rather than freezing then snapping.

import type { MatchEventData, ViewState } from '../../engine/protocol';
import { sleep, type FrontendCtx, type GameFrontend } from '../types';
import { lastMove, resetTelemetry } from './telemetry';

type Abs = 'n' | 'e' | 's' | 'w';

interface SnakeInfo {
  cells: [number, number][];
  dir: Abs;
  alive: boolean;
  score: number;
  health: number;
}

const MAX_HEALTH = 100;

interface DuelView {
  side: number;
  snakes: [SnakeInfo, SnakeInfo];
  food: [number, number] | null;
  step: number;
  cap: number;
  outcome: 'ongoing' | 'win0' | 'win1' | 'draw';
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

/** The action label the engine offers for each absolute heading
 * (games/snake/src/duel_ui.rs::action_label). */
const LABEL_OF: Record<Abs, string> = { n: 'up', e: 'right', s: 'down', w: 'left' };

/** Delay before the human's move auto-commits when it's their turn. Near a
 * frame, NOT a "beat to steer": the held/latest direction is sampled at the
 * commit and persists across ticks, so the player never loses an input by
 * committing fast — and committing fast keeps the human's near-zero think out
 * of the cell's supply interval, so the cadence stays steady on the bot's
 * search alone (a long human-tick gap was injecting jitter and stalls). */
const TICK_MS = 16;
/** Each cell glides LINEARLY over the time until the NEXT move actually arrives
 * — the measured interval between consecutive head-moving states — so the glide
 * always spans the real supply gap and the snake never reaches a cell early and
 * freezes. The bot time-budgets its search to a near-constant, so consecutive
 * intervals cluster tightly → a single near-constant velocity, smooth from the
 * very first move (the first uses CELL_MS_DEFAULT, then it tracks the true
 * supply — no stepped warmup, no lock). Clamped to a sane band. */
const CELL_MS_DEFAULT = 200;
const CELL_MS_MIN = 110;
const CELL_MS_MAX = 480;
/** How long `animate` blocks the shell before returning — short, so the bot's
 * search for the NEXT move overlaps this glide (its think is hidden under the
 * motion) WITHOUT the display ever running ahead: the next glide starts only
 * when this one ends, keeping exactly ONE move in flight (responsive input). */
const PACE_MS = 30;

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
const SEAT_PALETTES: [Palette, Palette] = [
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
];

const SEAT_COLORS = ['Green', 'Blue'];

function asView(data: unknown): DuelView | null {
  if (!data || typeof data !== 'object') return null;
  const v = data as Partial<DuelView>;
  if (typeof v.side !== 'number' || !Array.isArray(v.snakes) || v.snakes.length !== 2) return null;
  for (const s of v.snakes) {
    if (!s || !Array.isArray(s.cells) || s.cells.length === 0) return null;
  }
  return v as DuelView;
}

function headMoved(a: DuelView | null, b: DuelView): boolean {
  if (!a) return false;
  for (let i = 0; i < 2; i++) {
    const [ax, ay] = a.snakes[i].cells[0];
    const [bx, by] = b.snakes[i].cells[0];
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

/** The perf overlay is opt-in: `?snakeDebug` in the URL, or a sticky
 * `snakeDebug` localStorage flag, so it never clutters normal play but the
 * team can flip it on to read backend/ms/sims/trips/FPS in-browser. */
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
  from: DuelView;
  to: DuelView;
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

  private view: DuelView | null = null;
  private glide: Glide | null = null;
  /** Wall-clock time the current (or last-scheduled) glide ends. The next
   * head-move glide is scheduled to start here, so exactly one move is ever in
   * flight and the display never runs ahead of the engine. */
  private glideEndsAt = 0;
  /** When the last head-moving glide was scheduled to start, so the next glide's
   * duration can match the measured supply interval. */
  private lastGlideStart = 0;
  private side = 20;
  private cssSize = 0;
  private rafId = 0;
  private resizeObs: ResizeObserver | null = null;

  private pendingLabels: string[] | null = null;
  /** The human's most recently pressed direction, sampled at the next commit so
   * the snake turns immediately. `null` means keep going straight. Not a queue —
   * a queue would let an old keypress land late; the LATEST press wins. */
  private desired: Abs | null = null;
  private tickTimer = 0;
  private mySeat = -1;

  private foodPop = { at: 0, x: -1, y: -1 };
  private flashes: Flash[] = [];
  private deathOrbs: DeathOrb[] = [];
  private deadSeats = new Set<number>();
  private prevScores = [0, 0];

  // FPS instrumentation, surfaced to the console for the perf check.
  private frameTimes: number[] = [];
  private fpsLogAt = 0;

  // Opt-in perf overlay: backend / ms-per-move / sims / GPU round-trips / FPS.
  private debugEl!: HTMLElement;
  private showDebug = false;

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    this.mySeat = ctx.humanSeat;
    injectStyle();
    host.innerHTML = `
      <div class="snk-root">
        <div class="snk-bar">
          <div class="snk-chip snk-chip-0">
            <span class="snk-dot"></span>
            <span class="seat-slot" data-seat="0"></span>
            <span class="snk-hp"><span class="snk-hp-fill"></span></span>
            <span class="snk-len">3</span>
          </div>
          <div class="snk-chip snk-chip-1">
            <span class="snk-dot"></span>
            <span class="seat-slot" data-seat="1"></span>
            <span class="snk-hp"><span class="snk-hp-fill"></span></span>
            <span class="snk-len">3</span>
          </div>
        </div>
        <div class="snk-stage">
          <canvas class="snk-canvas"></canvas>
          <div class="snk-debug"></div>
          <div class="snk-overlay"><b></b><small></small></div>
        </div>
        <div class="snk-hint"></div>
      </div>`;
    this.canvas = host.querySelector('.snk-canvas')!;
    this.c2d = this.canvas.getContext('2d')!;
    this.chips = [host.querySelector('.snk-chip-0')!, host.querySelector('.snk-chip-1')!];
    this.lenEls = [
      this.chips[0].querySelector('.snk-len')!,
      this.chips[1].querySelector('.snk-len')!,
    ];
    this.hpEls = [this.chips[0].querySelector('.snk-hp')!, this.chips[1].querySelector('.snk-hp')!];
    this.hpFillEls = [
      this.chips[0].querySelector('.snk-hp-fill')!,
      this.chips[1].querySelector('.snk-hp-fill')!,
    ];
    this.overlayEl = host.querySelector('.snk-overlay')!;
    this.overlayTitleEl = this.overlayEl.querySelector('b')!;
    this.overlaySubEl = this.overlayEl.querySelector('small')!;
    this.hintEl = host.querySelector('.snk-hint')!;
    this.debugEl = host.querySelector('.snk-debug')!;
    this.showDebug = debugEnabled();
    this.debugEl.classList.toggle('snk-debug-on', this.showDebug);

    const stage = host.querySelector<HTMLElement>('.snk-stage')!;
    if (this.mySeat >= 0) {
      window.addEventListener('keydown', this.onKey);
      stage.addEventListener('touchstart', this.onTouchStart, { passive: true });
      stage.addEventListener('touchmove', this.onTouchMove, { passive: false });
      stage.addEventListener('touchend', this.onTouchEnd);
      this.hintEl.textContent = 'Arrow keys / WASD / swipe to steer';
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

  // ----- real-time driver interface (snake play) -----
  // The real-time driver (shell/snake-realtime.ts) owns a FIXED game clock and
  // drives the board through these two fire-and-forget methods, so the player's
  // snake advances on the clock and is NEVER gated by the bot's search.

  /** The player's heading for THIS tick: their latest pressed direction (legal
   * vs the current heading), else straight on. Sampled instantly — no await, no
   * bot — so the driver can apply it the moment the clock ticks. Clears the
   * consumed press so a held key keeps turning only once. */
  pollHeading(): string {
    const cur = this.currentHeading();
    const pressed = this.desired;
    this.desired = null;
    const want = pressed && pressed !== OPPOSITE[cur] ? pressed : cur;
    return LABEL_OF[want];
  }

  /** Glide to a freshly-applied state over `durMs` (the fixed clock period).
   * Fire-and-forget: starts the glide and returns immediately, so the driver's
   * clock is never blocked. A non-head-moving update (pending commit / food)
   * just merges in. */
  pushState(state: ViewState, durMs: number): void {
    const next = asView(state.viewData);
    if (!next) return;
    const prev = this.latestKnown();
    this.syncJuice(next);
    this.updateBar(next, state);
    this.updateOverlay(next, state);
    this.side = next.side;
    const scale = this.ctx.animationScale();
    if (scale <= 0) {
      this.view = next;
      this.glide = null;
      return;
    }
    if (!prev || !headMoved(prev, next)) {
      if (!this.glide) this.view = next;
      return;
    }
    const now = performance.now();
    const from = this.glide ? this.glide.to : (this.view ?? prev);
    this.glide = { from, to: next, start: now, dur: durMs * scale };
    this.glideEndsAt = now + durMs * scale;
  }

  async animate(_event: MatchEventData, after: ViewState): Promise<void> {
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
      return;
    }

    // No head movement (a seat-0 pending commit, or a food spawn): no motion of
    // its own. A live glide owns the display; otherwise snap this state in.
    if (!prev || !headMoved(prev, next)) {
      if (!this.glide) this.view = next;
      return;
    }

    // Glide for the time until the NEXT move is expected: the measured interval
    // between consecutive head-moving arrivals (this is metronomic because the
    // bot time-budgets its search). Using the real supply interval means the
    // glide spans the actual gap, so the snake never reaches a cell early and
    // freezes — and it tracks the gap whether one search (human play) or two
    // (watch) feed each cell, with no stepped warmup. The first move has no prior
    // interval, so it uses a sensible default.
    const arrived = performance.now();
    const dur =
      (this.lastGlideStart > 0
        ? clamp(arrived - this.lastGlideStart, CELL_MS_MIN, CELL_MS_MAX)
        : CELL_MS_DEFAULT) * scale;

    // Let the in-flight glide finish before starting the next, so EXACTLY ONE
    // move is ever in flight and the display never runs ahead of the engine.
    // That single-move-in-flight rule is what keeps input responsive: there is
    // no buffer of future moves for a keypress to queue behind, so the steering
    // sampled at the next commit always lands on the next visible move. While we
    // wait here the bot's search for the move AFTER this one is already running
    // in the shell loop, overlapping the glide — its think is hidden under the
    // motion, not stacked ahead of the player.
    const waitForPrev = this.glideEndsAt - performance.now();
    if (this.glide && waitForPrev > 0) await sleep(waitForPrev);

    const start = performance.now();
    const from = this.glide ? this.glide.to : (this.view ?? prev);
    this.glide = { from, to: next, start, dur };
    this.glideEndsAt = start + dur;
    this.lastGlideStart = arrived;
    // Return after only a short pace (the glide finishes in the rAF loop) so the
    // shell loops on to compute the next move while this one is still gliding.
    await sleep(PACE_MS * scale);
  }

  /** The most recent discrete state the frontend knows about — the active
   * glide's target, or the committed view. New moves compare against this to
   * decide if the head actually moved. */
  private latestKnown(): DuelView | null {
    if (this.glide) return this.glide.to;
    return this.view;
  }

  promptAction(labels: string[]): void {
    this.pendingLabels = labels;
    if (this.mySeat < 0) return;
    // Real-time: submit on the clock even with no input (the snake glides on).
    // The tick fires after one TICK_MS so the human always has a beat to steer.
    if (this.tickTimer) return;
    const scale = this.ctx.animationScale();
    const wait = TICK_MS * Math.max(scale, 0.001);
    this.tickTimer = window.setTimeout(() => {
      this.tickTimer = 0;
      this.fireTick();
    }, wait);
  }

  unmount(): void {
    cancelAnimationFrame(this.rafId);
    if (this.tickTimer) clearTimeout(this.tickTimer);
    this.tickTimer = 0;
    window.removeEventListener('keydown', this.onKey);
    this.resizeObs?.disconnect();
    this.resizeObs = null;
    resetTelemetry();
  }

  /** Submit the human's heading for this commit: the LATEST pressed direction
   * (sampled now, so a turn the player just pressed lands on THIS move), else
   * the snake's current heading (straight on). A 180° reversal of the current
   * heading is dropped as illegal and the snake continues straight. */
  private fireTick(): void {
    if (!this.pendingLabels || this.mySeat < 0) return;
    const cur = this.currentHeading();
    const pressed = this.desired;
    this.desired = null;
    const want = pressed && pressed !== OPPOSITE[cur] ? pressed : cur;
    const label = LABEL_OF[want];
    const i = this.pendingLabels.indexOf(label);
    const labels = this.pendingLabels;
    this.pendingLabels = null;
    this.ctx.submit(String(i >= 0 ? i : labels.indexOf(LABEL_OF[cur])));
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
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
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

  /** Record the human's intended heading. The LATEST press wins (overwrites any
   * earlier un-committed one), so the snake turns the way the player most
   * recently pressed on the very next commit — no stale queued turn to lag
   * behind. A 180° reversal of the current heading is dropped (it would fold the
   * snake onto itself); the rest is sampled at `fireTick`. */
  private steer(abs: Abs): void {
    if (this.mySeat < 0) return;
    if (abs === OPPOSITE[this.currentHeading()]) return;
    this.desired = abs;
  }

  private updateBar(view: DuelView, _state: ViewState): void {
    for (let seat = 0; seat < 2; seat++) {
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

  private updateOverlay(view: DuelView, state: ViewState): void {
    if (!state.isOver) {
      this.overlayEl.classList.remove('snk-show');
      return;
    }
    let title = 'Draw';
    if (view.outcome === 'win0') title = `${SEAT_COLORS[0]} wins`;
    else if (view.outcome === 'win1') title = `${SEAT_COLORS[1]} wins`;
    if (this.mySeat >= 0 && view.outcome !== 'draw') {
      const won = view.outcome === `win${this.mySeat}`;
      title = won ? 'You win!' : 'You lose';
    }
    this.overlayTitleEl.textContent = title;
    this.overlaySubEl.textContent = `${SEAT_COLORS[0]} ${view.snakes[0].score} · ${SEAT_COLORS[1]} ${view.snakes[1].score} · ${view.step} ticks`;
    this.overlayEl.classList.add('snk-show');
  }

  /** Fire the small flourishes off a fresh state: food respawn pop, an eat
   * ring when a score ticks up, and a death burst when a snake dies. */
  private syncJuice(view: DuelView): void {
    const now = performance.now();
    const f = view.food;
    if (f && (f[0] !== this.foodPop.x || f[1] !== this.foodPop.y)) {
      this.foodPop = { at: now, x: f[0], y: f[1] };
    }
    for (let seat = 0; seat < 2; seat++) {
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
    this.recordFps(now);
    if (this.showDebug) this.paintDebug(now);
    this.rafId = requestAnimationFrame(this.loop);
  };

  private recordFps(now: number): void {
    this.frameTimes.push(now);
    while (this.frameTimes.length && now - this.frameTimes[0] > 1000) this.frameTimes.shift();
    if (now - this.fpsLogAt > 3000) {
      this.fpsLogAt = now;
      // eslint-disable-next-line no-console
      console.info(`[snake] ${this.frameTimes.length} FPS`);
    }
  }

  /** Paint the opt-in perf HUD from the live FPS window and the bot driver's
   * latest per-move telemetry. `frameTimes` is a 1 s sliding window, so its
   * length is the current FPS. */
  private paintDebug(now: number): void {
    const fps = this.frameTimes.length;
    const m = lastMove();
    const lines = [`fps   ${fps}`];
    if (m) {
      lines.push(
        `bot   ${m.backend.toUpperCase()}`,
        `move  ${m.ms.toFixed(0)} ms`,
        `sims  ${m.sims}`,
      );
      if (m.backend === 'gpu') lines.push(`trips ${m.trips}`);
      const ageMs = now - m.at;
      if (ageMs > 2000) lines.push(`(idle ${(ageMs / 1000).toFixed(0)}s)`);
    } else {
      lines.push('bot   —');
    }
    const text = lines.join('\n');
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
    if (view?.food) this.drawFood(view.food, cell, now);

    const t = this.glideProgress(now);
    if (view) {
      for (let seat = 0; seat < 2; seat++) this.drawSnake(seat, t, cell);
    }

    this.drawFlashes(cell, now);
    this.drawDeathOrbs(cell, now);
  }

  /** Retire a finished glide: commit its target as the resting view so the next
   * `animate` glides out of the right place (and `render`, the human's-turn
   * redraw, can advance the view again). The NEXT glide is scheduled by
   * `animate` to begin at this one's end, so the cadence stays a metronome with
   * exactly one move in flight — no queue to drain here. */
  private advanceGlide(now: number): void {
    if (this.glide && now - this.glide.start >= this.glide.dur) {
      this.view = this.glide.to;
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

  private drawFood(food: [number, number], cell: number, now: number): void {
    const ctx = this.c2d;
    const cx = (food[0] + 0.5) * cell;
    const cy = (food[1] + 0.5) * cell;
    const age = (now - this.foodPop.at) / 240;
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
    // LINEAR in t (no easing): constant velocity within a cell, so combined with
    // the fixed cellMs-per-cell cadence the snake glides at a uniform speed.
    const cells = interpBody(from.cells, to.cells, t);
    if (cells.length === 0) return;

    // Opt-in (?snakeDebug) test seam: publish each snake's interpolated head
    // (board CELL coords), heading, and alive flag every frame, so the validation
    // harness can read the exact rendered heads — far cleaner than colour-
    // tracking the canvas through the juice. Seat 0 verifies input-following;
    // seat 1 verifies the bot actually plays (doesn't suicide).
    if (this.showDebug) {
      const key = seat === 0 ? '__snakeHead0' : '__snakeHead1';
      (window as unknown as Record<string, unknown>)[key] = {
        t: performance.now(),
        x: cells[0][0],
        y: cells[0][1],
        dir: to.dir,
        alive: to.alive,
        len: to.cells.length,
      };
    }

    const ctx = this.c2d;
    // Center-of-cell points; the tube width tapers slightly toward the tail.
    const pts = cells.map(([x, y]) => [(x + 0.5) * cell, (y + 0.5) * cell] as [number, number]);
    const baseW = cell * 0.74;

    ctx.save();
    ctx.globalAlpha = to.alive ? 1 : 0.4;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    // Outer glow under everything (skipped when dead — it's dissolving).
    if (to.alive) {
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

    this.drawHead(pts[0], to.dir, cell, pal, to.alive);
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

/** Interpolate a body's segments from their previous to current positions.
 * On a non-eating tick the body lengths match and segment i tweens from
 * from[i] to to[i]; on an eating tick the body grew by one, so the new head
 * tweens out of the old head's cell and the rest shift down by one. */
function interpBody(
  from: [number, number][],
  to: [number, number][],
  t: number,
): [number, number][] {
  const out: [number, number][] = [];
  const grew = to.length > from.length;
  for (let i = 0; i < to.length; i++) {
    const a = grew ? from[Math.max(0, i - 1)] : from[Math.min(i, from.length - 1)];
    const b = to[i];
    out.push([lerp(a[0], b[0], t), lerp(a[1], b[1], t)]);
  }
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

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

function easeOut(t: number): number {
  return 1 - (1 - t) * (1 - t);
}

export function createSnakeFrontend(): GameFrontend {
  return new SnakeFrontend();
}
