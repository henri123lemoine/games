// Snake frontend: neon canvas snake on the dark arcade shell.
//
// View JSON (contract with games/snake/src/ui.rs):
//   {"width":w,"height":h,"snake":[[x,y],... head first],"food":[x,y]|null,
//    "dir":"n|e|s|w","score":len,"status":"alive|crashed|starved|won"}
// `x` grows rightward, `y` downward; `food` is null while a spawn is
// pending; `score` is the snake length.
//
// The game's actions are RELATIVE (left / straight / right); absolute arrow
// keys are translated using the current heading from the view.
//
// Play mode is REAL-TIME: the engine stays turn-based, but once the player
// makes their first move this frontend auto-submits `straight` on a fixed
// clock (quickening as the snake eats), so stalling is impossible. Steering
// inputs queue two deep — classic snake — and are consumed one per tick;
// the clock starts on the first input and pauses while the tab is hidden.
// Watch mode is untouched: bot moves pace themselves through `animate`.

import type { MatchEventData, ViewState } from '../../engine/protocol';
import type { FrontendCtx, GameFrontend } from '../types';
import { sleep } from '../types';

type Abs = 'n' | 'e' | 's' | 'w';
type Rel = 'left' | 'straight' | 'right';

interface SnakeView {
  width: number;
  height: number;
  snake: [number, number][];
  food: [number, number] | null;
  dir: Abs;
  score: number;
  status: string;
}

function asView(data: unknown): SnakeView | null {
  if (!data || typeof data !== 'object') return null;
  const v = data as Partial<SnakeView>;
  if (typeof v.width !== 'number' || typeof v.height !== 'number') return null;
  if (!Array.isArray(v.snake) || v.snake.length === 0) return null;
  if (v.dir !== 'n' && v.dir !== 'e' && v.dir !== 's' && v.dir !== 'w') return null;
  return v as SnakeView;
}

const ABS_KEYS: Record<string, Abs> = {
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

const LEFT_OF: Record<Abs, Abs> = { n: 'w', w: 's', s: 'e', e: 'n' };
const RIGHT_OF: Record<Abs, Abs> = { n: 'e', e: 's', s: 'w', w: 'n' };

const DELTA: Record<Abs, [number, number]> = {
  n: [0, -1],
  e: [1, 0],
  s: [0, 1],
  w: [-1, 0],
};

/** Maps an absolute heading the player wants onto the game's relative
 * action; null when it is the 180° reverse (ignored, as in classic snake). */
function relativeOf(cur: Abs, want: Abs): Rel | null {
  if (want === cur) return 'straight';
  if (LEFT_OF[cur] === want) return 'left';
  if (LEFT_OF[want] === cur) return 'right';
  return null;
}

const PAD: { rel: Rel; glyph: string }[] = [
  { rel: 'left', glyph: '↶' },
  { rel: 'straight', glyph: '↑' },
  { rel: 'right', glyph: '↷' },
];

const MOVE_MS = 120;
const FLASH_MS = 560;
const TICK_BASE_MS = 180;
const TICK_FLOOR_MS = 90;
const TICK_RAMP_MS = 3; // shaved off the tick per food eaten
const QUEUE_MAX = 2;

const STYLE = `
.snake {
  margin: auto;
  width: min(100%, 520px);
  display: flex;
  flex-direction: column;
  gap: 14px;
  user-select: none;
}
.snake-top {
  display: flex;
  align-items: stretch;
  gap: 10px;
}
.snake-logo {
  margin-right: auto;
  align-self: center;
  font-size: 1.5rem;
  font-weight: 850;
  letter-spacing: 0.18em;
  background: linear-gradient(135deg, var(--good), var(--accent));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.snake-stat {
  min-width: 76px;
  padding: 6px 14px;
  text-align: center;
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) - 2px);
}
.snake-stat small {
  display: block;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.62rem;
  color: var(--text-dim);
}
.snake-stat b {
  font-size: 1.15rem;
  font-variant-numeric: tabular-nums;
}
.snake-frame {
  display: flex;
  justify-content: center;
  padding: 10px;
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
.snake-canvas {
  display: block;
  border-radius: calc(var(--radius) - 4px);
}
.snake-pad {
  display: flex;
  justify-content: center;
  gap: 8px;
}
.snake-pad.snake-hidden { display: none; }
.snake-btn {
  min-width: 96px;
  padding: 9px 0;
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) - 2px);
  color: var(--text);
  font-size: 0.92rem;
  transition: border-color 0.12s, color 0.12s;
}
.snake-btn:not(:disabled):hover { border-color: var(--good); color: var(--good); }
.snake-btn:disabled { opacity: 0.32; cursor: default; }
@media (max-width: 480px) {
  .snake-btn { min-width: 0; flex: 1; }
}
`;

function injectStyle(): void {
  if (document.getElementById('snake-frontend-style')) return;
  const style = document.createElement('style');
  style.id = 'snake-frontend-style';
  style.textContent = STYLE;
  document.head.append(style);
}

interface Tween {
  from: SnakeView;
  start: number;
  dur: number;
}

class SnakeFrontend implements GameFrontend {
  private ctx!: FrontendCtx;
  private view: SnakeView | null = null;
  private tween: Tween | null = null;
  private flash: { start: number; dur: number } | null = null;
  private overState: 'dead' | 'win' | null = null;
  private pending: string[] | null = null;
  private live = false;
  private armed = false;
  private queue: Rel[] = [];
  private tickTimer = 0;
  private nextTickAt = 0;
  private canvas!: HTMLCanvasElement;
  private c2d!: CanvasRenderingContext2D;
  private frameEl!: HTMLElement;
  private scoreEl!: HTMLElement;
  private lenEl!: HTMLElement;
  private padBtns = new Map<Rel, HTMLButtonElement>();
  private cssW = 0;
  private cssH = 0;
  private rafId = 0;
  private resizeObs: ResizeObserver | null = null;
  private colors = {
    bg: '#010409',
    snake: '#3fb950',
    headGlow: '#8aff9f',
    food: '#f85149',
    win: '#edc22e',
  };

  mount(host: HTMLElement, ctx: FrontendCtx): void {
    this.ctx = ctx;
    this.live = ctx.humanSeat >= 0;
    injectStyle();
    host.innerHTML = `
      <div class="snake">
        <div class="snake-top">
          <span class="snake-logo">SNAKE</span>
          <div class="snake-stat"><small>score</small><b class="snake-score">0</b></div>
          <div class="snake-stat"><small>length</small><b class="snake-len">0</b></div>
        </div>
        <div class="snake-frame"><canvas class="snake-canvas"></canvas></div>
        <div class="snake-pad"></div>
      </div>`;
    this.frameEl = host.querySelector('.snake-frame')!;
    this.canvas = host.querySelector('.snake-canvas')!;
    this.c2d = this.canvas.getContext('2d')!;
    this.scoreEl = host.querySelector('.snake-score')!;
    this.lenEl = host.querySelector('.snake-len')!;

    const cs = getComputedStyle(host);
    const cssVar = (name: string, fallback: string) =>
      cs.getPropertyValue(name).trim() || fallback;
    this.colors.bg = cssVar('--bg-inset', this.colors.bg);
    this.colors.snake = cssVar('--good', this.colors.snake);
    this.colors.food = cssVar('--bad', this.colors.food);

    const pad = host.querySelector<HTMLElement>('.snake-pad')!;
    if (ctx.humanSeat < 0) {
      pad.classList.add('snake-hidden');
    } else {
      for (const { rel, glyph } of PAD) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'snake-btn';
        b.textContent = `${glyph} ${rel}`;
        b.disabled = true;
        b.onclick = () => this.onInput(rel);
        pad.append(b);
        this.padBtns.set(rel, b);
      }
    }

    this.resizeObs = new ResizeObserver(() => this.layout());
    this.resizeObs.observe(this.frameEl);
    window.addEventListener('keydown', this.onKey);
    document.addEventListener('visibilitychange', this.onVisibility);
    this.rafId = requestAnimationFrame(this.loop);
  }

  render(state: ViewState): void {
    const v = asView(state.viewData);
    if (!v) return;
    this.view = v;
    this.tween = null;
    this.layout();
    this.updateStats(v);
    if (state.isOver) {
      this.overState = v.status === 'won' ? 'win' : 'dead';
      clearTimeout(this.tickTimer);
    }
    if (state.toAct !== state.humanSeat) this.setPending(null);
  }

  async animate(event: MatchEventData, after: ViewState): Promise<void> {
    void event;
    const v = asView(after.viewData);
    if (!v) return;
    const prev = this.view;
    this.view = v;
    this.layout();
    this.updateStats(v);
    const scale = this.ctx.animationScale();
    const dur = this.live ? Math.min(MOVE_MS * scale, this.tickMs() * 0.8) : MOVE_MS * scale;
    if (dur > 0 && prev) {
      this.tween = { from: prev, start: performance.now(), dur };
      await sleep(dur);
    }
    if (after.isOver) {
      this.overState = v.status === 'won' ? 'win' : 'dead';
      if (scale > 0) {
        this.flash = { start: performance.now(), dur: FLASH_MS * scale };
        await sleep(FLASH_MS * scale);
      }
    }
  }

  promptAction(labels: string[]): void {
    this.setPending(labels);
    if (this.live && this.armed) this.scheduleTick();
  }

  unmount(): void {
    cancelAnimationFrame(this.rafId);
    clearTimeout(this.tickTimer);
    window.removeEventListener('keydown', this.onKey);
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.resizeObs?.disconnect();
    this.resizeObs = null;
  }

  private onKey = (e: KeyboardEvent): void => {
    if (this.ctx.humanSeat < 0 || e.metaKey || e.ctrlKey || e.altKey) return;
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    const abs = ABS_KEYS[e.key];
    if (!abs) return;
    e.preventDefault();
    if (!this.view) return;
    const rel = relativeOf(this.predictedDir(), abs);
    if (rel) this.onInput(rel);
  };

  /** Heading the snake will have once the queued turns are consumed —
   * arrow keys pressed mid-queue must resolve against it, not the view. */
  private predictedDir(): Abs {
    let d = this.view!.dir;
    for (const r of this.queue) d = r === 'left' ? LEFT_OF[d] : r === 'right' ? RIGHT_OF[d] : d;
    return d;
  }

  private onInput(rel: Rel): void {
    if (!this.live || this.overState) return;
    if (!this.armed) {
      if (!this.pending) return;
      this.armed = true;
      this.nextTickAt = performance.now() + this.tickMs();
      this.submitRel(rel);
      return;
    }
    if (rel !== 'straight' && this.queue.length < QUEUE_MAX) this.queue.push(rel);
  }

  private scheduleTick(): void {
    clearTimeout(this.tickTimer);
    const delay = Math.max(0, this.nextTickAt - performance.now());
    this.tickTimer = window.setTimeout(() => this.tick(), delay);
  }

  private tick(): void {
    if (!this.pending || document.hidden) return;
    this.nextTickAt = performance.now() + this.tickMs();
    this.submitRel(this.queue.shift() ?? 'straight');
  }

  private tickMs(): number {
    const eaten = Math.max(0, (this.view?.score ?? 3) - 3);
    return Math.max(TICK_FLOOR_MS, TICK_BASE_MS - TICK_RAMP_MS * eaten);
  }

  private onVisibility = (): void => {
    if (document.hidden || !this.live || !this.armed || !this.pending) return;
    this.nextTickAt = performance.now() + this.tickMs();
    this.scheduleTick();
  };

  private submitRel(rel: Rel): void {
    if (!this.pending) return;
    const i = this.pending.indexOf(rel);
    if (i < 0) return;
    this.setPending(null);
    this.ctx.submit(String(i));
  }

  private setPending(labels: string[] | null): void {
    this.pending = labels;
    const running = this.armed && !this.overState;
    for (const [rel, btn] of this.padBtns)
      btn.disabled = running ? false : !labels || !labels.includes(rel);
  }

  private updateStats(v: SnakeView): void {
    this.scoreEl.textContent = String(Math.max(0, v.score - 3));
    this.lenEl.textContent = `${v.score}/${v.width * v.height}`;
  }

  private layout(): void {
    const v = this.view;
    if (!v) return;
    const avail = Math.max(120, this.frameEl.clientWidth - 22);
    const cell = Math.min(avail / v.width, 440 / v.height);
    const cssW = Math.round(cell * v.width);
    const cssH = Math.round(cell * v.height);
    if (cssW === this.cssW && cssH === this.cssH) return;
    this.cssW = cssW;
    this.cssH = cssH;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.style.width = `${cssW}px`;
    this.canvas.style.height = `${cssH}px`;
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
  }

  private loop = (): void => {
    this.drawFrame(performance.now());
    this.rafId = requestAnimationFrame(this.loop);
  };

  private drawFrame(now: number): void {
    const v = this.view;
    if (!v || this.canvas.width === 0) return;
    const g = this.c2d;
    const W = this.canvas.width;
    const H = this.canvas.height;
    const cell = W / v.width;
    const scale = this.ctx.animationScale();

    let t = 1;
    let pts: [number, number][] = v.snake;
    const tween = this.tween;
    if (tween) {
      t = Math.min(1, (now - tween.start) / Math.max(1, tween.dur));
      if (t >= 1) this.tween = null;
      const from = tween.from.snake;
      pts = v.snake.map(([bx, by], i) => {
        const [ax, ay] = from[Math.min(i, from.length - 1)];
        return [ax + (bx - ax) * t, ay + (by - ay) * t];
      });
    }
    if (v.status === 'crashed' && pts.length > 0) {
      const bump = (tween ? Math.sin(Math.min(1, t) * Math.PI) : 0) * 0.4;
      const [dx, dy] = DELTA[v.dir];
      pts = pts.slice();
      pts[0] = [pts[0][0] + dx * bump, pts[0][1] + dy * bump];
    }

    g.clearRect(0, 0, W, H);
    g.fillStyle = this.colors.bg;
    g.fillRect(0, 0, W, H);

    g.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    g.lineWidth = 1;
    g.beginPath();
    for (let x = 1; x < v.width; x++) {
      g.moveTo(x * cell, 0);
      g.lineTo(x * cell, H);
    }
    for (let y = 1; y < v.height; y++) {
      g.moveTo(0, y * cell);
      g.lineTo(W, y * cell);
    }
    g.stroke();

    if (v.food) {
      const appeared = tween && !tween.from.food ? t : 1;
      const pulse = scale > 0 ? 1 + 0.08 * Math.sin(now / 260) : 1;
      const r = cell * 0.3 * pulse * appeared;
      const [fx, fy] = v.food;
      const cx = (fx + 0.5) * cell;
      const cy = (fy + 0.5) * cell;
      const halo = g.createRadialGradient(cx, cy, 0, cx, cy, r * 3);
      halo.addColorStop(0, 'rgba(248, 81, 73, 0.35)');
      halo.addColorStop(1, 'rgba(248, 81, 73, 0)');
      g.fillStyle = halo;
      g.fillRect(cx - r * 3, cy - r * 3, r * 6, r * 6);
      g.save();
      g.shadowColor = this.colors.food;
      g.shadowBlur = cell * 0.5;
      g.fillStyle = this.colors.food;
      g.beginPath();
      g.arc(cx, cy, r, 0, Math.PI * 2);
      g.fill();
      g.restore();
    }

    const dead = this.overState === 'dead';
    const won = this.overState === 'win';
    const body = dead ? this.colors.food : won ? this.colors.win : this.colors.snake;
    g.save();
    if (dead && !this.flash) g.globalAlpha = 0.65;
    g.shadowColor = body;
    g.shadowBlur = cell * 0.65;
    g.strokeStyle = body;
    g.lineWidth = cell * 0.64;
    g.lineJoin = 'round';
    g.lineCap = 'round';
    g.beginPath();
    g.moveTo((pts[0][0] + 0.5) * cell, (pts[0][1] + 0.5) * cell);
    for (let i = 1; i < pts.length; i++) {
      g.lineTo((pts[i][0] + 0.5) * cell, (pts[i][1] + 0.5) * cell);
    }
    g.stroke();

    const hx = (pts[0][0] + 0.5) * cell;
    const hy = (pts[0][1] + 0.5) * cell;
    g.fillStyle = dead ? this.colors.food : this.colors.headGlow;
    g.beginPath();
    g.arc(hx, hy, cell * 0.36, 0, Math.PI * 2);
    g.fill();
    g.shadowBlur = 0;
    const [dx, dy] = DELTA[v.dir];
    g.fillStyle = '#04140a';
    for (const side of [-1, 1]) {
      g.beginPath();
      g.arc(
        hx + dx * cell * 0.14 - dy * side * cell * 0.16,
        hy + dy * cell * 0.14 + dx * side * cell * 0.16,
        cell * 0.07,
        0,
        Math.PI * 2,
      );
      g.fill();
    }
    g.restore();

    if (this.live && !this.armed && !this.overState) {
      g.save();
      g.fillStyle = 'rgba(1, 4, 9, 0.55)';
      g.fillRect(0, 0, W, H);
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      g.fillStyle = 'rgba(230, 237, 243, 0.92)';
      g.font = `600 ${Math.round(cell * 0.5)}px system-ui, sans-serif`;
      g.fillText('press an arrow to start', W / 2, H / 2 - cell * 1.6);
      g.fillStyle = 'rgba(230, 237, 243, 0.55)';
      g.font = `${Math.round(cell * 0.36)}px system-ui, sans-serif`;
      g.fillText("it won't wait for you", W / 2, H / 2 - cell * 0.85);
      g.restore();
    }

    const flash = this.flash;
    if (flash) {
      const p = (now - flash.start) / Math.max(1, flash.dur);
      if (p >= 1) {
        this.flash = null;
      } else {
        const alpha = 0.38 * Math.abs(Math.sin(p * Math.PI * 3));
        g.fillStyle = won ? `rgba(63, 185, 80, ${alpha})` : `rgba(248, 81, 73, ${alpha})`;
        g.fillRect(0, 0, W, H);
      }
    }
  }
}

export function createSnakeFrontend(): GameFrontend {
  return new SnakeFrontend();
}
