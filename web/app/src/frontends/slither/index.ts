// Slither: a real-time canvas game played against the trained encircle bot.
//
// Unlike the engine games (turn-based, driven by the wasm worker), slither is
// a continuous real-time sim, so it runs standalone on its own screen — the
// shell mounts it like DOOM. The dynamics and the bot both live in the
// `slither-engine` wasm package: `slither-rl`'s world steps every frame, and
// every non-human worm is driven by the PPO-trained net through
// `slitherinfer`'s torch-free forward over that worm's own egocentric,
// viewport-clipped view (the same partial observation it trained on). The
// human steers worm 0 by aiming at the cursor and holds to boost.

import init, { SlitherGame } from 'slither-engine';
import wasmUrl from 'slither-engine/slither_engine_bg.wasm?url';

const WEIGHTS_URL = `${import.meta.env.BASE_URL}slither/slither.weights`;

/** The world advances on a fixed 30 Hz clock (`slither_rl::world::DT`), so play
 * speed is identical regardless of the display's refresh rate. */
const TICK_HZ = 30;
const TICK_MS = 1000 / TICK_HZ;
/** Cap the catch-up so a backgrounded tab doesn't fast-forward on return. */
const MAX_TICKS_PER_FRAME = 5;

const WORMS = 8;
const PELLETS = 700;

const HUMAN = { body: '#58a6ff', head: '#bcd9ff', glow: 'rgba(88, 166, 255, 0.6)' };
const BOT = { body: '#f0883e', head: '#ffc08a', glow: 'rgba(240, 136, 62, 0.5)' };
const FOOD = '#7ee787';

let wasmReady: Promise<void> | null = null;
function ensureWasm(): Promise<void> {
  wasmReady ??= init({ module_or_path: wasmUrl }).then(() => undefined);
  wasmReady.catch(() => {
    wasmReady = null;
  });
  return wasmReady;
}

let weightsOnce: Promise<Uint8Array> | null = null;
function getWeights(): Promise<Uint8Array> {
  weightsOnce ??= (async () => {
    const resp = await fetch(WEIGHTS_URL);
    if (!resp.ok) throw new Error(`weights ${WEIGHTS_URL} missing (HTTP ${resp.status})`);
    return new Uint8Array(await resp.arrayBuffer());
  })();
  weightsOnce.catch(() => {
    weightsOnce = null;
  });
  return weightsOnce;
}

interface Worm {
  isHuman: boolean;
  dead: boolean;
  radius: number;
  length: number;
  segs: [number, number][];
}

function readWorms(blob: Float32Array): Worm[] {
  const worms: Worm[] = [];
  let i = 0;
  while (i + 5 <= blob.length) {
    const isHuman = blob[i++] === 1;
    const dead = blob[i++] === 1;
    const radius = blob[i++];
    const length = blob[i++];
    const segCount = blob[i++];
    const segs: [number, number][] = [];
    for (let s = 0; s < segCount && i + 2 <= blob.length; s++) {
      segs.push([blob[i++], blob[i++]]);
    }
    worms.push({ isHuman, dead, radius, length, segs });
  }
  return worms;
}

/** The standalone slither screen. Owns its RAF loop, input, and canvas; the
 * shell just mounts it into a host element and calls `destroy()` on teardown. */
export class SlitherScreen {
  private game: SlitherGame | null = null;
  private canvas!: HTMLCanvasElement;
  private c2d!: CanvasRenderingContext2D;
  private scoreEl!: HTMLElement;
  private aliveEl!: HTMLElement;
  private overlayEl!: HTMLElement;
  private overlayTitleEl!: HTMLElement;
  private overlaySubEl!: HTMLElement;
  private restartBtn!: HTMLButtonElement;

  private cssW = 0;
  private cssH = 0;
  private rafId = 0;
  private acc = 0;
  private last = 0;
  private resizeObs: ResizeObserver | null = null;
  private destroyed = false;

  // Input: cursor in CSS pixels relative to the canvas, and whether boost is held.
  private pointer = { x: 0, y: 0, has: false };
  private boost = false;
  private aim = 0;
  // Camera state, eased toward the human head so the view doesn't jitter.
  private cam = { x: 0, y: 0, scale: 1, ready: false };

  async mount(host: HTMLElement): Promise<void> {
    host.innerHTML = `
      <div class="slr-root">
        <div class="slr-hud">
          <span class="slr-stat">length <b class="slr-score">0</b></span>
          <span class="slr-stat">snakes <b class="slr-alive">0</b></span>
          <span class="slr-hint">move to steer · hold mouse / space to boost</span>
        </div>
        <div class="slr-stage">
          <canvas class="slr-canvas"></canvas>
          <div class="slr-overlay">
            <b class="slr-over-title"></b>
            <small class="slr-over-sub"></small>
            <button type="button" class="primary slr-restart">Play again</button>
          </div>
          <div class="slr-boot">Loading the trained bot…</div>
        </div>
      </div>`;
    this.canvas = host.querySelector('.slr-canvas')!;
    this.c2d = this.canvas.getContext('2d')!;
    this.scoreEl = host.querySelector('.slr-score')!;
    this.aliveEl = host.querySelector('.slr-alive')!;
    this.overlayEl = host.querySelector('.slr-overlay')!;
    this.overlayTitleEl = host.querySelector('.slr-over-title')!;
    this.overlaySubEl = host.querySelector('.slr-over-sub')!;
    this.restartBtn = host.querySelector('.slr-restart')!;
    const bootEl = host.querySelector<HTMLElement>('.slr-boot')!;

    const stage = host.querySelector<HTMLElement>('.slr-stage')!;
    this.resizeObs = new ResizeObserver(() => this.resize(stage));
    this.resizeObs.observe(stage);
    this.resize(stage);

    this.canvas.addEventListener('pointermove', this.onPointerMove);
    this.canvas.addEventListener('pointerdown', this.onPointerDown);
    window.addEventListener('pointerup', this.onPointerUp);
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    this.restartBtn.addEventListener('click', this.onRestart);

    try {
      const [, weights] = await Promise.all([ensureWasm(), getWeights()]);
      if (this.destroyed) return;
      this.start(weights);
      bootEl.remove();
    } catch (e) {
      bootEl.textContent = `Could not load slither: ${e instanceof Error ? e.message : e}`;
    }
  }

  private start(weights: Uint8Array): void {
    this.game?.free();
    this.game = new SlitherGame(weights, WORMS, PELLETS, randomSeed());
    this.cam.ready = false;
    this.acc = 0;
    this.last = performance.now();
    this.overlayEl.classList.remove('slr-show');
    this.aim = this.game.human_angle();
    if (!this.rafId) this.rafId = requestAnimationFrame(this.loop);
  }

  private onRestart = (): void => {
    if (!this.game) return;
    this.game.reset(randomSeed());
    this.cam.ready = false;
    this.acc = 0;
    this.last = performance.now();
    this.overlayEl.classList.remove('slr-show');
  };

  destroy(): void {
    this.destroyed = true;
    cancelAnimationFrame(this.rafId);
    this.rafId = 0;
    this.resizeObs?.disconnect();
    this.resizeObs = null;
    this.canvas.removeEventListener('pointermove', this.onPointerMove);
    this.canvas.removeEventListener('pointerdown', this.onPointerDown);
    window.removeEventListener('pointerup', this.onPointerUp);
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    this.game?.free();
    this.game = null;
  }

  private onPointerMove = (e: PointerEvent): void => {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer = { x: e.clientX - rect.left, y: e.clientY - rect.top, has: true };
  };

  private onPointerDown = (e: PointerEvent): void => {
    this.onPointerMove(e);
    this.boost = true;
  };

  private onPointerUp = (): void => {
    this.boost = false;
  };

  private onKeyDown = (e: KeyboardEvent): void => {
    if (e.code === 'Space') {
      e.preventDefault();
      this.boost = true;
    }
  };

  private onKeyUp = (e: KeyboardEvent): void => {
    if (e.code === 'Space') this.boost = false;
  };

  private resize(stage: HTMLElement): void {
    const rect = stage.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.cssW = w;
    this.cssH = h;
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.c2d.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  private loop = (now: number): void => {
    this.rafId = requestAnimationFrame(this.loop);
    const game = this.game;
    if (!game) return;

    // Fixed-timestep accumulator: catch up missed ticks, but cap so a long
    // stall (backgrounded tab) doesn't burst.
    this.acc += now - this.last;
    this.last = now;
    let ticks = 0;
    while (this.acc >= TICK_MS && ticks < MAX_TICKS_PER_FRAME) {
      this.acc -= TICK_MS;
      ticks++;
      if (!game.human_dead()) {
        this.updateAim(game);
        game.tick(this.aim, this.boost);
        if (game.human_dead()) this.showGameOver(game);
      }
    }
    if (ticks === MAX_TICKS_PER_FRAME) this.acc = 0;

    this.draw(game);
  };

  /** Aim the human head at the cursor (screen→world), holding the last aim when
   * the pointer hasn't moved onto the canvas yet. */
  private updateAim(game: SlitherGame): void {
    if (!this.pointer.has || !this.cam.ready) {
      this.aim = game.human_angle();
      return;
    }
    const head = game.human_head();
    const wx = this.cam.x + (this.pointer.x - this.cssW / 2) / this.cam.scale;
    const wy = this.cam.y + (this.pointer.y - this.cssH / 2) / this.cam.scale;
    const dx = wx - head[0];
    const dy = wy - head[1];
    if (dx * dx + dy * dy > 1) this.aim = Math.atan2(dy, dx);
  }

  private showGameOver(game: SlitherGame): void {
    this.overlayTitleEl.textContent = 'You died';
    this.overlaySubEl.textContent = `length ${Math.round(game.human_length())} · ${game.alive_count()} snakes survived you`;
    this.overlayEl.classList.add('slr-show');
  }

  private draw(game: SlitherGame): void {
    const ctx = this.c2d;
    const w = this.cssW;
    const h = this.cssH;
    if (w <= 0 || h <= 0) return;

    // Camera: center on the human head, scale so its egocentric view spans the
    // smaller screen dimension. Ease toward the target to avoid jitter.
    const head = game.human_head();
    const view = Math.max(120, game.human_view_radius());
    const targetScale = Math.min(w, h) / (2 * view);
    if (!this.cam.ready) {
      this.cam = { x: head[0], y: head[1], scale: targetScale, ready: true };
    } else {
      const k = 0.12;
      this.cam.x += (head[0] - this.cam.x) * k;
      this.cam.y += (head[1] - this.cam.y) * k;
      this.cam.scale += (targetScale - this.cam.scale) * k;
    }
    const s = this.cam.scale;
    const ox = w / 2 - this.cam.x * s;
    const oy = h / 2 - this.cam.y * s;
    const toX = (x: number) => x * s + ox;
    const toY = (y: number) => y * s + oy;

    ctx.clearRect(0, 0, w, h);
    this.drawBackground(game, s, ox, oy);

    // Pellets (cull to the viewport with a margin).
    const pellets = game.pellets_blob();
    ctx.fillStyle = FOOD;
    ctx.shadowColor = 'rgba(126, 231, 135, 0.7)';
    ctx.shadowBlur = 6;
    const pr = Math.max(1.5, 3.2 * s);
    for (let i = 0; i + 2 <= pellets.length; i += 2) {
      const px = toX(pellets[i]);
      const py = toY(pellets[i + 1]);
      if (px < -10 || px > w + 10 || py < -10 || py > h + 10) continue;
      ctx.beginPath();
      ctx.arc(px, py, pr, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.shadowBlur = 0;

    // Worms: bots first, human last (drawn on top).
    const worms = readWorms(game.worms_blob());
    for (const worm of worms) if (!worm.isHuman) this.drawWorm(worm, toX, toY, s);
    for (const worm of worms) if (worm.isHuman) this.drawWorm(worm, toX, toY, s);

    this.scoreEl.textContent = String(Math.round(game.human_length()));
    this.aliveEl.textContent = String(game.alive_count());
  }

  private drawBackground(game: SlitherGame, s: number, ox: number, oy: number): void {
    const ctx = this.c2d;
    const w = this.cssW;
    const h = this.cssH;
    ctx.fillStyle = '#070b13';
    ctx.fillRect(0, 0, w, h);
    // The arena floor (inside the wall) and a bright boundary, so the edge of
    // the world reads as a hard wall — running into it kills you.
    const size = game.world_size();
    const x0 = ox;
    const y0 = oy;
    const x1 = size * s + ox;
    const y1 = size * s + oy;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, y0, x1 - x0, y1 - y0);
    ctx.fillStyle = '#0b1220';
    ctx.fill();
    ctx.strokeStyle = 'rgba(240, 90, 90, 0.7)';
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.restore();
  }

  private drawWorm(
    worm: Worm,
    toX: (x: number) => number,
    toY: (y: number) => number,
    s: number,
  ): void {
    if (worm.dead || worm.segs.length === 0) return;
    const ctx = this.c2d;
    const color = worm.isHuman ? HUMAN : BOT;
    const r = Math.max(2, worm.radius * s);

    // The body as a single round-capped stroke through the segment chain — one
    // smooth ribbon rather than stacked discs.
    ctx.save();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.shadowColor = color.glow;
    ctx.shadowBlur = r * 0.9;
    ctx.strokeStyle = color.body;
    ctx.lineWidth = r * 2;
    ctx.beginPath();
    ctx.moveTo(toX(worm.segs[0][0]), toY(worm.segs[0][1]));
    for (let i = 1; i < worm.segs.length; i++) {
      ctx.lineTo(toX(worm.segs[i][0]), toY(worm.segs[i][1]));
    }
    ctx.stroke();
    ctx.restore();

    // Head cap + eyes, oriented along the neck.
    const [hx, hy] = worm.segs[0];
    const cx = toX(hx);
    const cy = toY(hy);
    ctx.save();
    ctx.shadowColor = color.glow;
    ctx.shadowBlur = r;
    ctx.fillStyle = color.head;
    ctx.beginPath();
    ctx.arc(cx, cy, r * 1.05, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    const next = worm.segs[1] ?? worm.segs[0];
    const ang = Math.atan2(hy - next[1], hx - next[0]);
    const ex = Math.cos(ang);
    const ey = Math.sin(ang);
    // Perpendicular for the two eyes.
    const px = -ey;
    const py = ex;
    const eyeOff = r * 0.5;
    const eyeFwd = r * 0.45;
    const eyeR = Math.max(1, r * 0.32);
    ctx.fillStyle = '#0a0f17';
    for (const side of [-1, 1]) {
      const eXx = cx + ex * eyeFwd + px * eyeOff * side;
      const eYy = cy + ey * eyeFwd + py * eyeOff * side;
      ctx.beginPath();
      ctx.arc(eXx, eYy, eyeR, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function randomSeed(): number {
  return (Math.floor(Math.random() * 0x7fff_ffff) | 1) >>> 0;
}
