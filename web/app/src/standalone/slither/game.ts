// Slither: a faithful slither.io clone rendered to a single canvas. You steer a
// worm toward the cursor, eat glowing pellets to grow, and boost (hold) to
// surge at the cost of length. A worm dies when its head crosses another worm's
// body; the survivor keeps going and the loser bursts into pellets. Several
// heuristic AI worms keep the arena alive. Everything here is self-contained —
// no engine, no network — driven by one requestAnimationFrame loop.

import type { StandaloneCtx, StandaloneGame } from '../types';

interface Vec {
  x: number;
  y: number;
}

interface Pellet {
  x: number;
  y: number;
  r: number;
  hue: number;
  value: number;
}

const WORLD = 4200;
const PELLET_TARGET = 900;
const AI_COUNT = 12;
const START_LENGTH = 22;
const SEG_SPACING = 4.4;
const BASE_SPEED = 168;
const BOOST_SPEED = 320;
const TURN_RATE = 4.2;
const BOOST_DRAIN_PER_SEC = 9;
const MIN_BOOST_LENGTH = START_LENGTH + 8;
const FOOD_PER_PELLET = 1;
const EAT_PADDING = 10;

const NAMES = [
  'Viper',
  'Coil',
  'Slinky',
  'Noodle',
  'Mamba',
  'Wriggle',
  'Twist',
  'Python',
  'Sidewinder',
  'Boa',
  'Ribbon',
  'Glide',
  'Serpent',
  'Hiss',
  'Loop',
  'Zigzag',
];

const SKINS = [196, 28, 142, 320, 50, 268, 0, 174, 96, 240, 14, 300];

function rand(lo: number, hi: number): number {
  return lo + Math.random() * (hi - lo);
}

function hueColor(hue: number, light = 55, sat = 80): string {
  return `hsl(${hue} ${sat}% ${light}%)`;
}

function clampToWorld(v: Vec): void {
  v.x = Math.max(0, Math.min(WORLD, v.x));
  v.y = Math.max(0, Math.min(WORLD, v.y));
}

class Worm {
  segments: Vec[] = [];
  angle: number;
  hue: number;
  name: string;
  isPlayer: boolean;
  food = 0;
  length: number;
  speed = BASE_SPEED;
  boosting = false;
  dead = false;
  aim = 0;
  aiRetarget = 0;

  constructor(x: number, y: number, hue: number, name: string, isPlayer: boolean) {
    this.hue = hue;
    this.name = name;
    this.isPlayer = isPlayer;
    this.angle = rand(0, Math.PI * 2);
    this.aim = this.angle;
    this.length = START_LENGTH;
    for (let i = 0; i < START_LENGTH; i++) {
      this.segments.push({ x: x - Math.cos(this.angle) * i * SEG_SPACING, y: y - Math.sin(this.angle) * i * SEG_SPACING });
    }
  }

  head(): Vec {
    return this.segments[0];
  }

  radius(): number {
    return 6 + Math.min(20, this.length / 16);
  }

  score(): number {
    return Math.max(0, Math.round((this.length - START_LENGTH) * 5 + this.food));
  }

  grow(amount: number): void {
    this.food += amount;
    while (this.food >= FOOD_PER_PELLET) {
      this.food -= FOOD_PER_PELLET;
      this.length += 1;
    }
  }

  /** Advance the head by `dist` toward `this.angle`, then drag the body along a
   * fixed-spacing chain so the worm reads as one smooth ribbon. */
  step(dist: number): void {
    const head = this.head();
    const nx = head.x + Math.cos(this.angle) * dist;
    const ny = head.y + Math.sin(this.angle) * dist;
    this.segments.unshift({ x: nx, y: ny });
    clampToWorld(this.segments[0]);

    const want = Math.max(START_LENGTH, Math.round(this.length));
    const path = this.segments;
    const out: Vec[] = [path[0]];
    let prev = path[0];
    let acc = 0;
    let i = 1;
    while (out.length < want && i < path.length) {
      const cur = path[i];
      const segLen = Math.hypot(cur.x - prev.x, cur.y - prev.y);
      if (segLen <= 1e-6) {
        i++;
        continue;
      }
      acc += segLen;
      while (acc >= SEG_SPACING && out.length < want) {
        acc -= SEG_SPACING;
        const t = 1 - acc / segLen;
        out.push({ x: prev.x + (cur.x - prev.x) * t, y: prev.y + (cur.y - prev.y) * t });
      }
      prev = cur;
      i++;
    }
    while (out.length < want) {
      const tail = out[out.length - 1];
      out.push({ x: tail.x, y: tail.y });
    }
    this.segments = out;
  }
}

export class SlitherGame implements StandaloneGame {
  private host!: HTMLElement;
  private canvas!: HTMLCanvasElement;
  private c2d!: CanvasRenderingContext2D;
  private overlay!: HTMLElement;
  private hud!: HTMLElement;

  private worms: Worm[] = [];
  private player!: Worm;
  private pellets: Pellet[] = [];
  private glow = true;
  private cam: Vec = { x: WORLD / 2, y: WORLD / 2 };
  private pointer: Vec = { x: 0, y: 0 };
  private boostHeld = false;
  private running = false;
  private over = false;
  private raf = 0;
  private last = 0;
  private best = 0;
  private dpr = 1;

  mount(host: HTMLElement, ctx: StandaloneCtx): void {
    this.host = host;
    this.glow = !ctx.reducedMotion;
    host.classList.add('slither-host');
    host.innerHTML = `
      <div class="slither-wrap">
        <canvas class="slither-canvas"></canvas>
        <div class="slither-hud">
          <div class="slither-stat"><small>length</small><b class="slither-len">0</b></div>
          <div class="slither-stat"><small>rank</small><b class="slither-rank">—</b></div>
          <div class="slither-stat"><small>best</small><b class="slither-best">0</b></div>
        </div>
        <canvas class="slither-mini"></canvas>
        <div class="slither-board">
          <small>leaderboard</small>
          <ol class="slither-board-list"></ol>
        </div>
        <div class="slither-hint">Move the mouse to steer · hold click / space to boost</div>
        <div class="slither-overlay"></div>
      </div>`;
    this.canvas = host.querySelector('.slither-canvas')!;
    this.c2d = this.canvas.getContext('2d')!;
    this.overlay = host.querySelector('.slither-overlay')!;
    this.hud = host.querySelector('.slither-hud')!;

    this.resize();
    window.addEventListener('resize', this.onResize);
    document.addEventListener('visibilitychange', this.onVisibility);
    this.canvas.addEventListener('pointermove', this.onPointerMove);
    this.canvas.addEventListener('pointerdown', this.onPointerDown);
    window.addEventListener('pointerup', this.onPointerUp);
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);

    this.reset();
  }

  unmount(): void {
    this.running = false;
    cancelAnimationFrame(this.raf);
    window.removeEventListener('resize', this.onResize);
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.canvas.removeEventListener('pointermove', this.onPointerMove);
    this.canvas.removeEventListener('pointerdown', this.onPointerDown);
    window.removeEventListener('pointerup', this.onPointerUp);
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    this.host.classList.remove('slither-host');
  }

  private reset(): void {
    this.over = false;
    this.worms = [];
    this.pellets = [];
    this.overlay.classList.remove('slither-show');
    this.overlay.innerHTML = '';

    this.player = new Worm(WORLD / 2, WORLD / 2, 140, 'You', true);
    this.worms.push(this.player);
    for (let i = 0; i < AI_COUNT; i++) this.spawnAi();
    for (let i = 0; i < PELLET_TARGET; i++) this.pellets.push(this.randomPellet());

    this.cam = { x: this.player.head().x, y: this.player.head().y };
    this.pointer = { x: this.canvas.clientWidth / 2, y: this.canvas.clientHeight / 2 };
    this.last = performance.now();
    this.running = true;
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(this.frame);
  }

  private spawnAi(): void {
    const idx = this.worms.length;
    const hue = SKINS[idx % SKINS.length] + rand(-12, 12);
    const name = NAMES[idx % NAMES.length];
    const margin = 300;
    const x = rand(margin, WORLD - margin);
    const y = rand(margin, WORLD - margin);
    const w = new Worm(x, y, hue, name, false);
    w.length = START_LENGTH + rand(0, 80);
    this.worms.push(w);
  }

  private randomPellet(): Pellet {
    const hue = rand(0, 360);
    return { x: rand(20, WORLD - 20), y: rand(20, WORLD - 20), r: rand(3, 5), hue, value: 1 };
  }

  // ---------- input ----------

  private onResize = (): void => this.resize();

  private onVisibility = (): void => {
    if (document.hidden) {
      this.running = false;
    } else if (!this.over) {
      this.running = true;
      this.last = performance.now();
      this.raf = requestAnimationFrame(this.frame);
    }
  };

  private onPointerMove = (e: PointerEvent): void => {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  private onPointerDown = (e: PointerEvent): void => {
    if (this.over) {
      this.reset();
      return;
    }
    if (e.button === 0) this.boostHeld = true;
  };

  private onPointerUp = (): void => {
    this.boostHeld = false;
  };

  private onKeyDown = (e: KeyboardEvent): void => {
    if (e.code === 'Space') {
      e.preventDefault();
      if (this.over) this.reset();
      else this.boostHeld = true;
    }
    if (e.key === 'Enter' && this.over) this.reset();
  };

  private onKeyUp = (e: KeyboardEvent): void => {
    if (e.code === 'Space') this.boostHeld = false;
  };

  private resize(): void {
    const wrap = this.host.querySelector<HTMLElement>('.slither-wrap')!;
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    this.canvas.width = Math.floor(w * this.dpr);
    this.canvas.height = Math.floor(h * this.dpr);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    const mini = this.host.querySelector<HTMLCanvasElement>('.slither-mini')!;
    const ms = Math.floor(140 * this.dpr);
    mini.width = ms;
    mini.height = ms;
  }

  // ---------- loop ----------

  private frame = (now: number): void => {
    if (!this.running) return;
    let dt = (now - this.last) / 1000;
    this.last = now;
    dt = Math.min(dt, 0.05);
    this.update(dt);
    this.render();
    this.raf = requestAnimationFrame(this.frame);
  };

  private update(dt: number): void {
    this.steerPlayer(dt);
    for (const w of this.worms) if (!w.isPlayer && !w.dead) this.steerAi(w, dt);

    for (const w of this.worms) {
      if (w.dead) continue;
      const wantBoost = w.isPlayer ? this.boostHeld : w.boosting;
      const canBoost = wantBoost && w.length > MIN_BOOST_LENGTH;
      w.speed += ((canBoost ? BOOST_SPEED : BASE_SPEED) - w.speed) * Math.min(1, dt * 8);
      if (canBoost) {
        w.length = Math.max(MIN_BOOST_LENGTH, w.length - BOOST_DRAIN_PER_SEC * dt);
        if (Math.random() < dt * 14) this.dropPellet(w.segments[w.segments.length - 1], w.hue, 1);
      }
      w.step(w.speed * dt);
    }

    this.eatPellets();
    this.resolveCollisions();
    this.refillPellets();
    this.followCamera(dt);

    if (this.player.dead && !this.over) this.endGame();
  }

  private steerPlayer(dt: number): void {
    if (this.player.dead) return;
    const head = this.player.head();
    const screenX = (head.x - this.cam.x) * this.zoom() + this.canvas.clientWidth / 2;
    const screenY = (head.y - this.cam.y) * this.zoom() + this.canvas.clientHeight / 2;
    const want = Math.atan2(this.pointer.y - screenY, this.pointer.x - screenX);
    this.player.angle = turnToward(this.player.angle, want, TURN_RATE * dt);
  }

  private steerAi(w: Worm, dt: number): void {
    w.aiRetarget -= dt;
    const head = w.head();
    let want = w.aim;

    const avoid = this.nearestThreat(w);
    if (avoid) {
      want = Math.atan2(head.y - avoid.y, head.x - avoid.x);
      w.boosting = false;
    } else {
      if (w.aiRetarget <= 0) {
        w.aiRetarget = rand(0.6, 1.8);
        const food = this.nearestPellet(head, 520);
        if (food) want = Math.atan2(food.y - head.y, food.x - head.x);
        else want = w.angle + rand(-1.1, 1.1);
        w.boosting = w.length > MIN_BOOST_LENGTH + 30 && Math.random() < 0.12;
      } else {
        want = w.aim;
      }
    }

    const wall = 220;
    if (head.x < wall) want = 0;
    else if (head.x > WORLD - wall) want = Math.PI;
    if (head.y < wall) want = Math.PI / 2;
    else if (head.y > WORLD - wall) want = -Math.PI / 2;

    w.aim = want;
    w.angle = turnToward(w.angle, want, TURN_RATE * dt);
  }

  /** A foe segment sitting just ahead of the worm's head, or null — the AI
   * steers directly away from it to avoid crashing into another body. */
  private nearestThreat(w: Worm): Vec | null {
    const head = w.head();
    const look = w.radius() + 60 + w.speed * 0.18;
    const fx = head.x + Math.cos(w.angle) * look;
    const fy = head.y + Math.sin(w.angle) * look;
    let best: Vec | null = null;
    let bestD = Infinity;
    for (const other of this.worms) {
      if (other === w || other.dead) continue;
      const step = 3;
      for (let i = 0; i < other.segments.length; i += step) {
        const s = other.segments[i];
        const d = Math.hypot(s.x - fx, s.y - fy);
        const hit = other.radius() + w.radius() + 18;
        if (d < hit && d < bestD) {
          bestD = d;
          best = s;
        }
      }
    }
    return best;
  }

  private nearestPellet(p: Vec, maxDist: number): Pellet | null {
    let best: Pellet | null = null;
    let bestD = maxDist * maxDist;
    for (const pl of this.pellets) {
      const dx = pl.x - p.x;
      const dy = pl.y - p.y;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = pl;
      }
    }
    return best;
  }

  private eatPellets(): void {
    for (const w of this.worms) {
      if (w.dead) continue;
      const head = w.head();
      const reach = w.radius() + EAT_PADDING;
      const reach2 = reach * reach;
      for (let i = this.pellets.length - 1; i >= 0; i--) {
        const pl = this.pellets[i];
        const dx = pl.x - head.x;
        const dy = pl.y - head.y;
        if (dx * dx + dy * dy <= reach2) {
          w.grow(pl.value);
          this.pellets[i] = this.pellets[this.pellets.length - 1];
          this.pellets.pop();
        }
      }
    }
  }

  /** A worm dies when its head enters another living worm's body. The head's
   * owner dies; the body's owner survives — so ramming a bigger worm is fatal
   * to the rammer, exactly as in slither.io. */
  private resolveCollisions(): void {
    const dying: Worm[] = [];
    for (const w of this.worms) {
      if (w.dead) continue;
      for (const other of this.worms) {
        if (other === w || other.dead) continue;
        if (headHitsBody(w.head(), w.radius() + other.radius(), other.segments)) {
          dying.push(w);
          break;
        }
      }
    }
    for (const w of dying) this.killWorm(w);
  }

  private killWorm(w: Worm): void {
    if (w.dead) return;
    w.dead = true;
    const drop = Math.max(8, Math.floor(w.segments.length / 2));
    for (let i = 0; i < w.segments.length; i += Math.max(1, Math.floor(w.segments.length / drop))) {
      const s = w.segments[i];
      this.dropPellet(s, w.hue, 2, true);
    }
    if (!w.isPlayer) {
      window.setTimeout(() => {
        const idx = this.worms.indexOf(w);
        if (idx >= 0) this.worms.splice(idx, 1);
        if (!this.over) this.spawnAi();
      }, 400);
    }
  }

  private dropPellet(p: Vec, hue: number, value: number, big = false): void {
    this.pellets.push({
      x: p.x + rand(-6, 6),
      y: p.y + rand(-6, 6),
      r: big ? rand(5, 7) : rand(3, 5),
      hue,
      value,
    });
  }

  private refillPellets(): void {
    while (this.pellets.length < PELLET_TARGET) this.pellets.push(this.randomPellet());
    if (this.pellets.length > PELLET_TARGET * 1.6) {
      this.pellets.splice(0, this.pellets.length - Math.floor(PELLET_TARGET * 1.4));
    }
  }

  private followCamera(dt: number): void {
    const head = this.player.dead ? this.cam : this.player.head();
    const k = Math.min(1, dt * 6);
    this.cam.x += (head.x - this.cam.x) * k;
    this.cam.y += (head.y - this.cam.y) * k;
  }

  private zoom(): number {
    const r = this.player.radius();
    return Math.max(0.62, Math.min(1.0, 12 / r));
  }

  private endGame(): void {
    this.over = true;
    this.best = Math.max(this.best, this.player.score());
    const rank = this.rankOf(this.player);
    this.overlay.innerHTML = `
      <div class="slither-over">
        <h2>You died</h2>
        <p>length <b>${Math.round(this.player.length)}</b> · rank <b>${rank}</b> of ${this.aliveCount()}</p>
        <button type="button" class="slither-replay">Play again</button>
        <small>click anywhere, or press space</small>
      </div>`;
    this.overlay.classList.add('slither-show');
    this.overlay.querySelector<HTMLButtonElement>('.slither-replay')!.onclick = () => this.reset();
  }

  private aliveCount(): number {
    return this.worms.filter((w) => !w.dead).length + (this.player.dead ? 1 : 0);
  }

  private rankOf(target: Worm): number {
    const ranked = [...this.worms].sort((a, b) => b.length - a.length);
    return ranked.indexOf(target) + 1;
  }

  // ---------- render ----------

  private render(): void {
    const c = this.c2d;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, w, h);

    const dark = document.documentElement.classList.contains('dark');
    c.fillStyle = dark ? '#0c0f14' : '#141821';
    c.fillRect(0, 0, w, h);

    const z = this.zoom();
    const ox = w / 2 - this.cam.x * z;
    const oy = h / 2 - this.cam.y * z;

    this.drawGrid(c, w, h, z, ox, oy);
    this.drawBorder(c, z, ox, oy);
    this.drawPellets(c, w, h, z, ox, oy);

    const order = [...this.worms].filter((wm) => !wm.dead).sort((a, b) => a.length - b.length);
    for (const wm of order) this.drawWorm(c, wm, z, ox, oy);

    this.drawMinimap();
    this.updateHud();
  }

  private drawGrid(c: CanvasRenderingContext2D, w: number, h: number, z: number, ox: number, oy: number): void {
    const grid = 110 * z;
    c.strokeStyle = 'rgba(255,255,255,0.045)';
    c.lineWidth = 1;
    c.beginPath();
    const startX = ox % grid;
    const startY = oy % grid;
    for (let x = startX; x < w; x += grid) {
      c.moveTo(x, 0);
      c.lineTo(x, h);
    }
    for (let y = startY; y < h; y += grid) {
      c.moveTo(0, y);
      c.lineTo(w, y);
    }
    c.stroke();
  }

  private drawBorder(c: CanvasRenderingContext2D, z: number, ox: number, oy: number): void {
    c.save();
    c.strokeStyle = 'rgba(231, 76, 90, 0.7)';
    c.lineWidth = 6;
    c.shadowColor = 'rgba(231, 76, 90, 0.6)';
    c.shadowBlur = 18;
    c.strokeRect(ox, oy, WORLD * z, WORLD * z);
    c.restore();
  }

  private drawPellets(c: CanvasRenderingContext2D, w: number, h: number, z: number, ox: number, oy: number): void {
    for (const p of this.pellets) {
      const sx = p.x * z + ox;
      const sy = p.y * z + oy;
      if (sx < -10 || sy < -10 || sx > w + 10 || sy > h + 10) continue;
      const r = p.r * z;
      c.beginPath();
      c.fillStyle = hueColor(p.hue, 60, 90);
      if (this.glow) {
        c.shadowColor = hueColor(p.hue, 60, 90);
        c.shadowBlur = 10 * z;
      }
      c.arc(sx, sy, r, 0, Math.PI * 2);
      c.fill();
    }
    c.shadowBlur = 0;
  }

  private drawWorm(c: CanvasRenderingContext2D, wm: Worm, z: number, ox: number, oy: number): void {
    const r = wm.radius() * z;
    const segs = wm.segments;
    c.lineCap = 'round';
    c.lineJoin = 'round';

    c.beginPath();
    c.moveTo(segs[0].x * z + ox, segs[0].y * z + oy);
    for (let i = 1; i < segs.length; i++) c.lineTo(segs[i].x * z + ox, segs[i].y * z + oy);

    c.strokeStyle = hueColor(wm.hue, 30, 70);
    c.lineWidth = r * 2 + 4;
    c.stroke();

    c.strokeStyle = hueColor(wm.hue, wm.isPlayer ? 60 : 52, 85);
    c.lineWidth = r * 2;
    if (this.glow && wm.speed > BASE_SPEED + 40) {
      c.shadowColor = hueColor(wm.hue, 65, 90);
      c.shadowBlur = 16;
    }
    c.stroke();
    c.shadowBlur = 0;

    c.strokeStyle = 'rgba(255,255,255,0.18)';
    c.lineWidth = r * 0.7;
    c.stroke();

    this.drawHead(c, wm, z, ox, oy, r);
    this.drawLabel(c, wm, z, ox, oy);
  }

  private drawHead(c: CanvasRenderingContext2D, wm: Worm, z: number, ox: number, oy: number, r: number): void {
    const head = wm.head();
    const hx = head.x * z + ox;
    const hy = head.y * z + oy;
    const eyeR = Math.max(2, r * 0.34);
    const eyeOff = r * 0.55;
    const perp = wm.angle + Math.PI / 2;
    const fwd = r * 0.45;
    for (const sign of [-1, 1]) {
      const ex = hx + Math.cos(perp) * eyeOff * sign + Math.cos(wm.angle) * fwd;
      const ey = hy + Math.sin(perp) * eyeOff * sign + Math.sin(wm.angle) * fwd;
      c.beginPath();
      c.fillStyle = '#fff';
      c.arc(ex, ey, eyeR, 0, Math.PI * 2);
      c.fill();
      c.beginPath();
      c.fillStyle = '#10131a';
      c.arc(ex + Math.cos(wm.angle) * eyeR * 0.4, ey + Math.sin(wm.angle) * eyeR * 0.4, eyeR * 0.5, 0, Math.PI * 2);
      c.fill();
    }
  }

  private drawLabel(c: CanvasRenderingContext2D, wm: Worm, z: number, ox: number, oy: number): void {
    const head = wm.head();
    const hx = head.x * z + ox;
    const hy = head.y * z + oy;
    c.font = `600 ${Math.max(10, 13 * z)}px ui-sans-serif, system-ui, sans-serif`;
    c.textAlign = 'center';
    c.fillStyle = wm.isPlayer ? '#fff' : 'rgba(255,255,255,0.7)';
    c.fillText(wm.name, hx, hy - wm.radius() * z - 8);
  }

  private drawMinimap(): void {
    const mini = this.host.querySelector<HTMLCanvasElement>('.slither-mini')!;
    const mc = mini.getContext('2d')!;
    const size = mini.width;
    mc.setTransform(1, 0, 0, 1, 0, 0);
    mc.clearRect(0, 0, size, size);
    mc.fillStyle = 'rgba(10,13,18,0.82)';
    mc.fillRect(0, 0, size, size);
    mc.strokeStyle = 'rgba(231,76,90,0.5)';
    mc.lineWidth = 1.5;
    mc.strokeRect(1, 1, size - 2, size - 2);
    const s = (size - 4) / WORLD;
    for (const wm of this.worms) {
      if (wm.dead) continue;
      const head = wm.head();
      mc.beginPath();
      mc.fillStyle = wm.isPlayer ? '#7CFFB0' : hueColor(wm.hue, 60, 80);
      mc.arc(2 + head.x * s, 2 + head.y * s, wm.isPlayer ? 3.2 : 2, 0, Math.PI * 2);
      mc.fill();
    }
  }

  private updateHud(): void {
    this.hud.querySelector('.slither-len')!.textContent = String(Math.round(this.player.length));
    this.hud.querySelector('.slither-rank')!.textContent = this.player.dead
      ? '—'
      : `${this.rankOf(this.player)}/${this.aliveCount()}`;
    this.hud.querySelector('.slither-best')!.textContent = String(Math.max(this.best, this.player.score()));

    const list = this.host.querySelector<HTMLElement>('.slither-board-list')!;
    const top = [...this.worms]
      .filter((w) => !w.dead)
      .sort((a, b) => b.length - a.length)
      .slice(0, 5);
    list.innerHTML = top
      .map((w) => {
        const cls = w.isPlayer ? ' class="slither-me"' : '';
        const dot = `<i style="background:${hueColor(w.hue, 60, 80)}"></i>`;
        return `<li${cls}>${dot}<span>${escapeHtml(w.name)}</span><b>${Math.round(w.length)}</b></li>`;
      })
      .join('');
  }
}

/** Whether `head` lies within `hitDist` of any of `body`'s segments past the
 * neck (index ≥ 2 so a worm never collides with its own head's wake). */
export function headHitsBody(head: Vec, hitDist: number, body: Vec[]): boolean {
  const hit2 = hitDist * hitDist;
  for (let i = 2; i < body.length; i++) {
    const dx = body[i].x - head.x;
    const dy = body[i].y - head.y;
    if (dx * dx + dy * dy <= hit2) return true;
  }
  return false;
}

function turnToward(current: number, target: number, maxStep: number): number {
  let diff = target - current;
  while (diff > Math.PI) diff -= Math.PI * 2;
  while (diff < -Math.PI) diff += Math.PI * 2;
  if (diff > maxStep) diff = maxStep;
  else if (diff < -maxStep) diff = -maxStep;
  return current + diff;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}
