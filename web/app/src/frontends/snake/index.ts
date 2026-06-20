// Snake frontend: two neon snakes racing on a 20x20 canvas grid.
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

import type { MatchEventData, ViewState } from '../../engine/protocol';
import type { FrontendCtx, GameFrontend } from '../types';

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

const TICK_MS = 150;
const QUEUE_MAX = 2;
const SEAT_COLORS = [
  { body: '#3fb950', head: '#8aff9f', glow: 'rgba(63, 185, 80, 0.55)' },
  { body: '#58a6ff', head: '#bcd9ff', glow: 'rgba(88, 166, 255, 0.55)' },
];
const SEAT_NAMES = ['Snake A', 'Snake B'];

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
  padding: 7px 12px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--bg-inset);
  color: var(--text-dim);
  font-size: 0.86rem;
  white-space: nowrap;
  transition: border-color 0.25s, box-shadow 0.25s, color 0.25s, opacity 0.25s;
}
.snk-chip.snk-dead {
  opacity: 0.45;
}
.snk-chip.snk-turn {
  border-color: var(--accent);
  color: var(--text);
  box-shadow: 0 0 12px rgba(88, 166, 255, 0.28);
}
.snk-dot {
  width: 13px;
  height: 13px;
  border-radius: 50%;
  flex: none;
}
.snk-chip-0 .snk-dot { background: ${SEAT_COLORS[0].body}; box-shadow: 0 0 8px ${SEAT_COLORS[0].glow}; }
.snk-chip-1 .snk-dot { background: ${SEAT_COLORS[1].body}; box-shadow: 0 0 8px ${SEAT_COLORS[1].glow}; }
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
.snk-chip-0 .snk-hp-fill { background: ${SEAT_COLORS[0].body}; }
.snk-chip-1 .snk-hp-fill { background: ${SEAT_COLORS[1].body}; }
.snk-hp.snk-hp-low .snk-hp-fill { background: #f85149; }
.snk-stage {
  position: relative;
  aspect-ratio: 1 / 1;
  border-radius: var(--radius);
  overflow: hidden;
  background: radial-gradient(circle at 50% 42%, #0c1f15, #05100b 92%);
  border: 1px solid var(--border);
  box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.5);
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
  background: rgba(4, 12, 8, 0.66);
  color: var(--text);
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
  font-size: 1.5rem;
  letter-spacing: 0.04em;
}
.snk-overlay small {
  color: var(--text-dim);
}
.snk-hint {
  text-align: center;
  color: var(--text-dim);
  font-size: 0.8rem;
  min-height: 1.1em;
}
`;

function injectStyle(): void {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.append(style);
}

interface Tween {
  from: DuelView;
  to: DuelView;
  start: number;
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
  private tween: Tween | null = null;
  private side = 20;
  private cssSize = 0;
  private rafId = 0;
  private resizeObs: ResizeObserver | null = null;

  private pendingLabels: string[] | null = null;
  private queue: Abs[] = [];
  private tickTimer = 0;
  private mySeat = -1;
  private foodPop = { at: 0, x: -1, y: -1 };

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    this.mySeat = ctx.humanSeat;
    injectStyle();
    host.innerHTML = `
      <div class="snk-root">
        <div class="snk-bar">
          <div class="snk-chip snk-chip-0">
            <span class="snk-dot"></span><span class="snk-name">${SEAT_NAMES[0]}</span>
            <span class="seat-slot" data-seat="0"></span>
            <span class="snk-hp"><span class="snk-hp-fill"></span></span>
            <span class="snk-len">3</span>
          </div>
          <div class="snk-chip snk-chip-1">
            <span class="snk-dot"></span><span class="snk-name">${SEAT_NAMES[1]}</span>
            <span class="seat-slot" data-seat="1"></span>
            <span class="snk-hp"><span class="snk-hp-fill"></span></span>
            <span class="snk-len">3</span>
          </div>
        </div>
        <div class="snk-stage">
          <canvas class="snk-canvas"></canvas>
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
    this.loop();
  }

  render(state: ViewState): void {
    const view = asView(state.viewData);
    if (!view) return;
    this.side = view.side;
    // A render with no head movement (e.g. seat 0's pending commit, or a food
    // spawn) snaps; a render that advanced both heads is tweened by animate().
    if (!this.tween) this.view = view;
    this.updateBar(view, state);
    this.maybeFoodPop(view);
    this.updateOverlay(view, state);
  }

  async animate(_event: MatchEventData, after: ViewState): Promise<void> {
    const next = asView(after.viewData);
    if (!next) return;
    const prev = this.view;
    this.maybeFoodPop(next);
    this.updateBar(next, after);
    this.updateOverlay(next, after);
    const scale = this.ctx.animationScale();
    if (!prev || !headMoved(prev, next) || scale <= 0) {
      this.view = next;
      this.tween = null;
      this.side = next.side;
      return;
    }
    this.side = next.side;
    const dur = TICK_MS * Math.max(scale, 0.001);
    this.tween = { from: prev, to: next, start: performance.now() };
    await new Promise<void>((resolve) => {
      const done = () => {
        if (!this.tween || this.tween.to !== next) return resolve();
        if (performance.now() - this.tween.start >= dur) {
          this.view = next;
          this.tween = null;
          resolve();
        } else {
          requestAnimationFrame(done);
        }
      };
      requestAnimationFrame(done);
    });
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
  }

  /** Submit the human's heading for this tick: a queued turn if any, else the
   * snake's current heading (straight on). */
  private fireTick(): void {
    if (!this.pendingLabels || this.mySeat < 0) return;
    const cur = this.view?.snakes[this.mySeat].dir ?? 'e';
    const want = this.queue.shift() ?? cur;
    const label = LABEL_OF[want];
    const i = this.pendingLabels.indexOf(label);
    const labels = this.pendingLabels;
    this.pendingLabels = null;
    this.ctx.submit(String(i >= 0 ? i : labels.indexOf(LABEL_OF[cur])));
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

  /** Queue a turn, ignoring a 180° reversal of the heading already committed
   * for the next tick (the snake cannot fold back on itself). */
  private steer(abs: Abs): void {
    if (this.mySeat < 0) return;
    const last = this.queue.length ? this.queue[this.queue.length - 1] : this.view?.snakes[this.mySeat].dir;
    if (last && abs === OPPOSITE[last]) return;
    if (last && abs === last) return;
    if (this.queue.length < QUEUE_MAX) this.queue.push(abs);
  }

  private updateBar(view: DuelView, state: ViewState): void {
    for (let seat = 0; seat < 2; seat++) {
      const s = view.snakes[seat];
      this.lenEls[seat].textContent = String(s.score);
      const hp = Math.max(0, Math.min(MAX_HEALTH, s.health ?? MAX_HEALTH));
      const pct = s.alive ? (hp / MAX_HEALTH) * 100 : 0;
      this.hpFillEls[seat].style.width = `${pct}%`;
      this.hpEls[seat].classList.toggle('snk-hp-low', s.alive && hp <= 25);
      this.hpEls[seat].title = `health ${hp}`;
      this.chips[seat].classList.toggle('snk-dead', !s.alive);
      this.chips[seat].classList.toggle(
        'snk-turn',
        !state.isOver && s.alive && state.toAct === seat,
      );
    }
  }

  private updateOverlay(view: DuelView, state: ViewState): void {
    if (!state.isOver) {
      this.overlayEl.classList.remove('snk-show');
      return;
    }
    let title = 'Draw';
    if (view.outcome === 'win0') title = `${SEAT_NAMES[0]} wins`;
    else if (view.outcome === 'win1') title = `${SEAT_NAMES[1]} wins`;
    if (this.mySeat >= 0 && view.outcome !== 'draw') {
      const won = view.outcome === `win${this.mySeat}`;
      title = won ? 'You win!' : 'You lose';
    }
    this.overlayTitleEl.textContent = title;
    this.overlaySubEl.textContent = `${SEAT_NAMES[0]} ${view.snakes[0].score} · ${SEAT_NAMES[1]} ${view.snakes[1].score} · ${view.step} ticks`;
    this.overlayEl.classList.add('snk-show');
  }

  private maybeFoodPop(view: DuelView): void {
    const f = view.food;
    if (!f) return;
    if (f[0] !== this.foodPop.x || f[1] !== this.foodPop.y) {
      this.foodPop = { at: performance.now(), x: f[0], y: f[1] };
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

  private loop = (): void => {
    this.draw();
    this.rafId = requestAnimationFrame(this.loop);
  };

  private draw(): void {
    const ctx = this.c2d;
    const size = this.cssSize;
    if (size <= 0) return;
    const cell = size / this.side;
    ctx.clearRect(0, 0, size, size);
    this.drawGrid(cell);

    const view = this.tween ? this.tween.to : this.view;
    if (!view) return;

    if (view.food) this.drawFood(view.food, cell);

    const t = this.tweenProgress();
    for (let seat = 0; seat < 2; seat++) {
      this.drawSnake(seat, t, cell);
    }
  }

  private tweenProgress(): number {
    if (!this.tween) return 1;
    const scale = this.ctx.animationScale();
    const dur = TICK_MS * Math.max(scale, 0.001);
    return Math.min(1, (performance.now() - this.tween.start) / dur);
  }

  private drawGrid(cell: number): void {
    const ctx = this.c2d;
    ctx.save();
    ctx.strokeStyle = 'rgba(120, 220, 150, 0.06)';
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

  private drawFood(food: [number, number], cell: number): void {
    const ctx = this.c2d;
    const cx = (food[0] + 0.5) * cell;
    const cy = (food[1] + 0.5) * cell;
    const age = (performance.now() - this.foodPop.at) / 220;
    const pop = age < 1 ? 0.6 + 0.4 * easeOut(age) : 1;
    const r = cell * 0.34 * pop;
    ctx.save();
    ctx.shadowColor = 'rgba(248, 81, 73, 0.9)';
    ctx.shadowBlur = cell * 0.6;
    const grad = ctx.createRadialGradient(cx - r * 0.3, cy - r * 0.3, r * 0.1, cx, cy, r);
    grad.addColorStop(0, '#ff8d7e');
    grad.addColorStop(1, '#f85149');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  /** Draw one snake, interpolating each segment from its `from` position to
   * its `to` position by `t` so the body glides one cell per tick. */
  private drawSnake(seat: number, t: number, cell: number): void {
    const color = SEAT_COLORS[seat];
    const to = (this.tween ? this.tween.to : this.view!).snakes[seat];
    const from = this.tween ? this.tween.from.snakes[seat] : to;
    const cells = interpBody(from.cells, to.cells, t);
    if (cells.length === 0) return;

    const ctx = this.c2d;
    const pad = cell * 0.12;
    const seg = cell - pad * 2;
    const radius = seg * 0.32;
    ctx.save();
    ctx.shadowColor = color.glow;
    ctx.shadowBlur = to.alive ? cell * 0.45 : 0;
    ctx.globalAlpha = to.alive ? 1 : 0.4;
    for (let i = cells.length - 1; i >= 0; i--) {
      const [x, y] = cells[i];
      const head = i === 0;
      ctx.fillStyle = head ? color.head : color.body;
      roundRect(ctx, x * cell + pad, y * cell + pad, seg, seg, radius);
      ctx.fill();
      if (head) this.drawEyes(x, y, to.dir, cell);
    }
    ctx.restore();
  }

  private drawEyes(x: number, y: number, dir: Abs, cell: number): void {
    const ctx = this.c2d;
    const [dx, dy] = DELTA[dir];
    const cx = (x + 0.5) * cell;
    const cy = (y + 0.5) * cell;
    const fwd = cell * 0.16;
    const side = cell * 0.16;
    const r = cell * 0.07;
    ctx.save();
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#06120a';
    for (const s of [-1, 1]) {
      const ex = cx + dx * fwd + dy * side * s;
      const ey = cy + dy * fwd + dx * side * s;
      ctx.beginPath();
      ctx.arc(ex, ey, r, 0, Math.PI * 2);
      ctx.fill();
    }
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

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function easeOut(t: number): number {
  return 1 - (1 - t) * (1 - t);
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

export function createSnakeFrontend(): GameFrontend {
  return new SnakeFrontend();
}
