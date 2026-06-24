// Slither: a real-time canvas game played against the trained encircle bot,
// rendered to read like slither.io.
//
// The dynamics and the bot both live in the `slither-engine` wasm package:
// `slither-rl`'s world steps on a fixed 30 Hz clock, and every non-human worm is
// driven by the PPO-trained net through `slitherinfer`'s torch-free forward over
// that worm's own egocentric, viewport-clipped view. The human steers worm 0 by
// aiming at the cursor and holds to boost.
//
// Performance: the sim is decoupled from rendering. A fixed-timestep accumulator
// advances `game.tick()` at exactly 30 Hz (the engine throttles bot forwards
// round-robin so a tick is cheap); rendering runs every animation frame and
// interpolates worm positions between the two most recent sim snapshots, so the
// picture is smooth at the display's refresh rate even though the sim is 30 Hz.
// Pellet orbs are pre-rendered to offscreen sprites, so the per-frame draw is
// mostly `drawImage` blits.

import init, { SlitherGame } from "slither-engine";
import wasmUrl from "slither-engine/slither_engine_bg.wasm?url";

const WEIGHTS_URL = `${import.meta.env.BASE_URL}slither/slither.weights`;

/** The world advances on a fixed 30 Hz clock (`slither_rl::world::DT`). */
const TICK_HZ = 30;
const TICK_MS = 1000 / TICK_HZ;
/** Cap catch-up so a backgrounded tab doesn't fast-forward a burst on return. */
const MAX_TICKS_PER_FRAME = 4;

const WORMS = 6;
const PELLETS = 5000;
const LEADERBOARD_ROWS = 10;

// --- slither.io palette (saturated wheel hues) -----------------------------
// The human is always the cyan-blue snake so it reads as "you"; bots get the
// rest of the wheel, assigned by seat so a worm keeps its colors across frames.
const PALETTE: [string, string][] = [
  ["#ff8c1a", "#ffd23f"], // orange / gold
  ["#22c32a", "#7cff4d"], // green / lime
  ["#9b3bff", "#d49bff"], // purple / lilac
  ["#ff36c0", "#ff9ae6"], // magenta / pink
  ["#ffe000", "#fff79e"], // yellow
  ["#ff2b2b", "#ff8a8a"], // red
  ["#1ee5d6", "#9bfff6"], // teal
  ["#7a5cff", "#bcb0ff"], // indigo
];
const HUMAN_SKIN: [string, string] = ["#2b8bff", "#9fd0ff"];

interface Skin {
  a: string; // primary band
  b: string; // secondary band
  rim: string; // dark outline
  glow: string; // additive boost halo
}

function buildSkin([a, b]: [string, string]): Skin {
  return { a, b, rim: mix(a, "#05070c", 0.55), glow: a };
}

const HUMAN_SKIN_BUILT = buildSkin(HUMAN_SKIN);
const BOT_SKINS = PALETTE.map(buildSkin);

function skinForSeat(seat: number, isHuman: boolean): Skin {
  if (isHuman) return HUMAN_SKIN_BUILT;
  return BOT_SKINS[seat % BOT_SKINS.length];
}

// --- color helpers ----------------------------------------------------------
function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}
function mix(a: string, b: string, t: number): string {
  const [ar, ag, ab] = hexToRgb(a);
  const [br, bg, bb] = hexToRgb(b);
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const bl = Math.round(ab + (bb - ab) * t);
  return `rgb(${r},${g},${bl})`;
}

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
    if (!resp.ok)
      throw new Error(`weights ${WEIGHTS_URL} missing (HTTP ${resp.status})`);
    return new Uint8Array(await resp.arrayBuffer());
  })();
  weightsOnce.catch(() => {
    weightsOnce = null;
  });
  return weightsOnce;
}

// --- decoded snapshots ------------------------------------------------------
interface WormSnap {
  seat: number;
  isHuman: boolean;
  dead: boolean;
  boosting: boolean;
  radius: number;
  length: number;
  angle: number;
  // Flat [x0,y0, x1,y1, …] segment chain, head first.
  segs: Float32Array;
  segCount: number;
}

interface Snapshot {
  worms: WormSnap[];
  pellets: Float32Array; // [x,y,value] triples
  pelletCount: number;
}

// Worm header stride in the blob: seat, isHuman, dead, boosting, r, len, angle, n.
const WORM_HEADER = 8;

function readSnapshot(
  wormBlob: Float32Array,
  pelletBlob: Float32Array,
): Snapshot {
  const worms: WormSnap[] = [];
  let i = 0;
  while (i + WORM_HEADER <= wormBlob.length) {
    const seat = wormBlob[i++];
    const isHuman = wormBlob[i++] === 1;
    const dead = wormBlob[i++] === 1;
    const boosting = wormBlob[i++] === 1;
    const radius = wormBlob[i++];
    const length = wormBlob[i++];
    const angle = wormBlob[i++];
    const segCount = wormBlob[i++];
    const n = Math.min(segCount, Math.floor((wormBlob.length - i) / 2));
    const segs = wormBlob.subarray(i, i + n * 2);
    i += n * 2;
    worms.push({
      seat,
      isHuman,
      dead,
      boosting,
      radius,
      length,
      angle,
      segs,
      segCount: n,
    });
  }
  return {
    worms,
    pellets: pelletBlob,
    pelletCount: Math.floor(pelletBlob.length / 3),
  };
}

/** Lerp angle the short way around the circle. */
function lerpAngle(a: number, b: number, t: number): number {
  let d = b - a;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return a + d * t;
}

// --- a single visual effect: the death burst --------------------------------
interface BurstOrb {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  color: string;
  born: number;
  delay: number;
}

/** The standalone slither screen. Owns its RAF loop, input, and canvas; the
 * shell just mounts it into a host element and calls `destroy()` on teardown. */
export class SlitherScreen {
  private game: SlitherGame | null = null;
  private canvas!: HTMLCanvasElement;
  private c2d!: CanvasRenderingContext2D;
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

  // Camera: position tracks the interpolated human head; zoom eases with growth.
  private cam = { x: 0, y: 0, scale: 1, ready: false };

  // Double-buffered sim snapshots for render interpolation.
  private prevSnap: Snapshot | null = null;
  private currSnap: Snapshot | null = null;

  // Death burst (the dead human's body shattering into orbs).
  private bursts: BurstOrb[] = [];
  private deathLogged = false;

  // FPS instrumentation (logged to console for the perf check).
  private frameTimes: number[] = [];
  private fpsLogAt = 0;

  // Offscreen sprites, built once up front (they're DPR-independent — scaled at
  // blit time). Pellets use a small spectrum of hues; death orbs are warm-gold.
  private hexTile = 60;
  private pelletSprites: HTMLCanvasElement[] = [];
  private deathSprite = document.createElement("canvas");
  private vignette: CanvasGradient | null = null;

  async mount(host: HTMLElement): Promise<void> {
    host.innerHTML = `
      <div class="slr-root">
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
    this.canvas = host.querySelector(".slr-canvas")!;
    this.c2d = this.canvas.getContext("2d", { alpha: false })!;
    this.overlayEl = host.querySelector(".slr-overlay")!;
    this.overlayTitleEl = host.querySelector(".slr-over-title")!;
    this.overlaySubEl = host.querySelector(".slr-over-sub")!;
    this.restartBtn = host.querySelector(".slr-restart")!;
    const bootEl = host.querySelector<HTMLElement>(".slr-boot")!;

    const stage = host.querySelector<HTMLElement>(".slr-stage")!;
    this.resizeObs = new ResizeObserver(() => this.resize(stage));
    this.resizeObs.observe(stage);
    this.resize(stage);
    this.buildSprites();

    this.canvas.addEventListener("pointermove", this.onPointerMove);
    this.canvas.addEventListener("pointerdown", this.onPointerDown);
    window.addEventListener("pointerup", this.onPointerUp);
    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);
    this.restartBtn.addEventListener("click", this.onRestart);

    try {
      const [, weights] = await Promise.all([ensureWasm(), getWeights()]);
      if (this.destroyed) return;
      this.start(weights);
      bootEl.remove();
    } catch (e) {
      bootEl.textContent = `Could not load Coil: ${e instanceof Error ? e.message : e}`;
    }
  }

  private start(weights: Uint8Array): void {
    this.game?.free();
    this.game = new SlitherGame(weights, WORMS, PELLETS, randomSeed());
    this.resetRunState();
    this.aim = this.game.human_angle();
    this.captureSnapshot();
    if (!this.rafId) this.rafId = requestAnimationFrame(this.loop);
  }

  private onRestart = (): void => {
    if (!this.game) return;
    this.game.reset(randomSeed());
    this.resetRunState();
    this.captureSnapshot();
    this.overlayEl.classList.remove("slr-show");
  };

  private resetRunState(): void {
    this.cam.ready = false;
    this.acc = 0;
    this.last = performance.now();
    this.prevSnap = null;
    this.currSnap = null;
    this.bursts = [];
    this.deathLogged = false;
  }

  destroy(): void {
    this.destroyed = true;
    cancelAnimationFrame(this.rafId);
    this.rafId = 0;
    this.resizeObs?.disconnect();
    this.resizeObs = null;
    this.canvas.removeEventListener("pointermove", this.onPointerMove);
    this.canvas.removeEventListener("pointerdown", this.onPointerDown);
    window.removeEventListener("pointerup", this.onPointerUp);
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("keyup", this.onKeyUp);
    this.game?.free();
    this.game = null;
  }

  private onPointerMove = (e: PointerEvent): void => {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      has: true,
    };
  };
  private onPointerDown = (e: PointerEvent): void => {
    this.onPointerMove(e);
    this.boost = true;
  };
  private onPointerUp = (): void => {
    this.boost = false;
  };
  private onKeyDown = (e: KeyboardEvent): void => {
    if (e.code === "Space") {
      e.preventDefault();
      this.boost = true;
    }
  };
  private onKeyUp = (e: KeyboardEvent): void => {
    if (e.code === "Space") this.boost = false;
  };

  private resize(stage: HTMLElement): void {
    const rect = stage.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    // Cap DPR at 1.5: the scene is glow/gradient heavy (additive pellet bloom,
    // full-screen scrim + vignette), so it's fill-rate bound. 1.5 keeps text and
    // edges crisp while roughly halving the pixel count vs a 2.0 retina buffer —
    // the difference between a locked 60 and a struggle on integrated GPUs.
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    this.cssW = w;
    this.cssH = h;
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.c2d.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.vignette = this.buildVignette();
  }

  // --- sim/render loop ------------------------------------------------------

  private loop = (now: number): void => {
    this.rafId = requestAnimationFrame(this.loop);
    const game = this.game;
    if (!game) return;

    this.acc += now - this.last;
    this.last = now;

    let ticks = 0;
    while (this.acc >= TICK_MS && ticks < MAX_TICKS_PER_FRAME) {
      this.acc -= TICK_MS;
      ticks++;
      if (!game.human_dead()) {
        this.updateAim(game);
        game.tick(this.aim, this.boost);
        this.captureSnapshot();
        if (game.human_dead()) this.onHumanDied(game);
      }
    }
    if (ticks === MAX_TICKS_PER_FRAME) this.acc = 0;

    const alpha = game.human_dead() ? 1 : Math.min(1, this.acc / TICK_MS);
    this.draw(game, alpha, now);
    this.recordFps(now);
  };

  /** Push the latest world into the snapshot ring (prev ← curr ← new). */
  private captureSnapshot(): void {
    const game = this.game;
    if (!game) return;
    this.prevSnap = this.currSnap;
    this.currSnap = readSnapshot(game.worms_blob(), game.pellets_blob());
  }

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

  private onHumanDied(game: SlitherGame): void {
    if (this.deathLogged) return;
    this.deathLogged = true;
    this.spawnDeathBurst();
    this.overlayTitleEl.textContent = "You died";
    const rank = this.humanRank();
    this.overlaySubEl.textContent =
      `length ${Math.round(game.human_length())} · finished #${rank} of ${WORMS}` +
      ` · ${game.alive_count()} snakes outlived you`;
    this.overlayEl.classList.add("slr-show");
  }

  private humanRank(): number {
    const game = this.game;
    if (!game) return WORMS;
    const lb = game.leaderboard_blob(); // sorted desc, [seat, isHuman, dead, len]
    for (let i = 0, r = 1; i + 4 <= lb.length; i += 4, r++) {
      if (lb[i + 1] === 1) return r;
    }
    return WORMS;
  }

  private recordFps(now: number): void {
    this.frameTimes.push(now);
    while (this.frameTimes.length > 0 && now - this.frameTimes[0] > 1000) {
      this.frameTimes.shift();
    }
    if (now - this.fpsLogAt > 2000) {
      this.fpsLogAt = now;
      // eslint-disable-next-line no-console
      console.info(`[slither] ${this.frameTimes.length} FPS`);
    }
  }

  // --- drawing --------------------------------------------------------------

  private draw(game: SlitherGame, alpha: number, now: number): void {
    const w = this.cssW;
    const h = this.cssH;
    if (w <= 0 || h <= 0 || !this.currSnap) return;

    // Camera follows the human head, which is already interpolated between sim
    // snapshots. Easing only the zoom avoids smearing the world-anchored floor.
    const head = this.interpHumanHead(alpha);
    const view = Math.max(120, game.human_view_radius());
    const targetScale = Math.min(w, h) / (2 * view);
    if (!this.cam.ready) {
      this.cam = { x: head[0], y: head[1], scale: targetScale, ready: true };
    } else {
      this.cam.x = head[0];
      this.cam.y = head[1];
      this.cam.scale += (targetScale - this.cam.scale) * 0.06;
    }
    const s = this.cam.scale;
    const ox = w / 2 - this.cam.x * s;
    const oy = h / 2 - this.cam.y * s;

    this.drawBackground(game, s, ox, oy);
    this.drawArenaBorder(game, s, ox, oy, now);
    this.drawPellets(s, ox, oy, w, h, now);

    const worms = this.interpWorms(alpha);
    // Shadows first (under everything), then bodies bots→human, then heads.
    for (const wm of worms) this.drawWormShadow(wm, s, ox, oy);
    for (const wm of worms)
      if (!wm.isHuman) this.drawWormBody(wm, s, ox, oy, now);
    for (const wm of worms)
      if (wm.isHuman) this.drawWormBody(wm, s, ox, oy, now);
    for (const wm of worms)
      if (!wm.isHuman) this.drawWormHead(wm, s, ox, oy, w, h);
    for (const wm of worms)
      if (wm.isHuman) this.drawWormHead(wm, s, ox, oy, w, h);

    this.drawBursts(s, ox, oy, now);
    this.drawHud(game);
  }

  /** The human head position, interpolated between the two latest snapshots. */
  private interpHumanHead(alpha: number): [number, number] {
    const curr = this.currSnap!.worms.find((wm) => wm.isHuman);
    if (!curr || curr.segCount === 0) {
      const h = this.game!.human_head();
      return [h[0], h[1]];
    }
    const cx = curr.segs[0];
    const cy = curr.segs[1];
    const prev = this.prevSnap?.worms.find((wm) => wm.isHuman);
    if (!prev || prev.segCount === 0) return [cx, cy];
    return [
      prev.segs[0] + (cx - prev.segs[0]) * alpha,
      prev.segs[1] + (cy - prev.segs[1]) * alpha,
    ];
  }

  /** Build per-frame interpolated worm geometry, matching prev↔curr by seat. */
  private interpWorms(alpha: number): InterpWorm[] {
    const curr = this.currSnap!;
    const prev = this.prevSnap;
    const out: InterpWorm[] = [];
    for (const c of curr.worms) {
      if (c.dead || c.segCount === 0) continue;
      const p = prev?.worms.find(
        (w) => w.seat === c.seat && !w.dead && w.segCount > 0,
      );
      const n = c.segCount;
      const xs = new Float32Array(n);
      const ys = new Float32Array(n);
      if (p) {
        const m = Math.min(n, p.segCount);
        for (let i = 0; i < m; i++) {
          const px = p.segs[i * 2];
          const py = p.segs[i * 2 + 1];
          xs[i] = px + (c.segs[i * 2] - px) * alpha;
          ys[i] = py + (c.segs[i * 2 + 1] - py) * alpha;
        }
        for (let i = m; i < n; i++) {
          xs[i] = c.segs[i * 2];
          ys[i] = c.segs[i * 2 + 1];
        }
      } else {
        for (let i = 0; i < n; i++) {
          xs[i] = c.segs[i * 2];
          ys[i] = c.segs[i * 2 + 1];
        }
      }
      const angle = p ? lerpAngle(p.angle, c.angle, alpha) : c.angle;
      out.push({
        seat: c.seat,
        isHuman: c.isHuman,
        boosting: c.boosting,
        radius: c.radius,
        angle,
        xs,
        ys,
        n,
        skin: skinForSeat(c.seat, c.isHuman),
      });
    }
    return out;
  }

  // --- background -----------------------------------------------------------

  private drawBackground(
    game: SlitherGame,
    s: number,
    ox: number,
    oy: number,
  ): void {
    const ctx = this.c2d;
    const w = this.cssW;
    const h = this.cssH;

    ctx.fillStyle = "#181b22";
    ctx.fillRect(0, 0, w, h);

    this.drawHexFloor(s, ox, oy);

    // Beyond the arena, darken the floor so the bounded play area reads as a
    // pocket of light. Punch the circular arena out of a full-screen scrim.
    const size = game.world_size();
    const radius = game.world_radius();
    const cx = (size / 2) * s + ox;
    const cy = (size / 2) * s + oy;
    const cr = radius * s;
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, w, h);
    ctx.arc(cx, cy, cr, 0, Math.PI * 2, true); // reverse-wound circle -> even-odd hole
    ctx.fillStyle = "rgba(2,4,9,0.62)";
    ctx.fill("evenodd");
    ctx.restore();

    if (this.vignette) {
      ctx.fillStyle = this.vignette;
      ctx.fillRect(0, 0, w, h);
    }
  }

  private buildVignette(): CanvasGradient {
    const ctx = this.c2d;
    const w = this.cssW;
    const h = this.cssH;
    const cx = w / 2;
    const cy = h / 2;
    const r = Math.hypot(w, h) / 2;
    const g = ctx.createRadialGradient(cx, cy, r * 0.55, cx, cy, r);
    g.addColorStop(0, "rgba(0,0,0,0)");
    g.addColorStop(1, "rgba(0,0,0,0.5)");
    return g;
  }

  private drawHexFloor(s: number, ox: number, oy: number): void {
    const ctx = this.c2d;
    const pitch = this.hexTile;
    const minCol = Math.floor((-ox / s) / pitch) - 2;
    const maxCol = Math.ceil(((this.cssW - ox) / s) / pitch) + 2;
    const minRow = Math.floor((-oy / s) / pitch) - 2;
    const maxRow = Math.ceil(((this.cssH - oy) / s) / pitch) + 2;
    const r = (pitch * 0.47) * s;

    ctx.save();
    ctx.lineWidth = Math.max(1, r * 0.16);
    ctx.strokeStyle = "#0a0f15";
    for (let row = minRow; row <= maxRow; row++) {
      const y = row * pitch * s + oy;
      const rowOffset = row & 1 ? pitch / 2 : 0;
      for (let col = minCol; col <= maxCol; col++) {
        const x = (col * pitch + rowOffset) * s + ox;
        ctx.beginPath();
        for (let k = 0; k < 6; k++) {
          const a = (Math.PI / 3) * k - Math.PI / 6; // flat-top
          const px = x + Math.cos(a) * r;
          const py = y + Math.sin(a) * r;
          if (k === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fillStyle = ((row + col) & 1) ? "#1b2127" : "#20262d";
        ctx.fill();
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  private drawArenaBorder(
    game: SlitherGame,
    s: number,
    ox: number,
    oy: number,
    now: number,
  ): void {
    const ctx = this.c2d;
    const size = game.world_size();
    const radius = game.world_radius();
    const cx = (size / 2) * s + ox;
    const cy = (size / 2) * s + oy;
    const cr = radius * s;

    // Glowing danger ring on the circular arena boundary; pulses faintly.
    const pulse = 0.5 + 0.5 * Math.sin(now / 380);
    ctx.save();
    // Outer soft glow.
    ctx.strokeStyle = `rgba(255, 90, 20, ${0.28 + 0.12 * pulse})`;
    ctx.lineWidth = Math.max(6, 16 * s);
    ctx.shadowColor = "rgba(255, 70, 20, 0.8)";
    ctx.shadowBlur = 24 * Math.min(1.5, s + 0.3);
    ctx.beginPath();
    ctx.arc(cx, cy, cr, 0, Math.PI * 2);
    ctx.stroke();
    // Inner bright edge.
    ctx.shadowBlur = 0;
    ctx.strokeStyle = `rgba(255, 59, 31, ${0.85})`;
    ctx.lineWidth = Math.max(2, 4 * s);
    ctx.beginPath();
    ctx.arc(cx, cy, cr, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // --- pellets --------------------------------------------------------------

  private buildSprites(): void {
    // A spectrum of saturated pellet hues (slither.io's full-wheel prey), each a
    // white-hot-core → hue → transparent orb. Picked per pellet by a stable hash
    // so the field shimmers in many colors without per-pellet gradient rebuilds.
    const hues = [
      "#7ee0ff",
      "#7cff8a",
      "#ffe066",
      "#ff8ad1",
      "#b69bff",
      "#ff9d5c",
      "#5cffd0",
    ];
    this.pelletSprites = hues.map((h) => {
      const cv = document.createElement("canvas");
      paintOrb(cv, "#ffffff", h, 48);
      return cv;
    });
    paintOrb(this.deathSprite, "#ffffff", "#ffd070", 64);
  }

  private drawPellets(
    s: number,
    ox: number,
    oy: number,
    w: number,
    h: number,
    now: number,
  ): void {
    const snap = this.currSnap;
    if (!snap) return;
    const ctx = this.c2d;
    const p = snap.pellets;
    const count = snap.pelletCount;

    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    const margin = 24;
    const nSprites = this.pelletSprites.length;
    for (let i = 0; i < count; i++) {
      const wx = p[i * 3];
      const wy = p[i * 3 + 1];
      const px = wx * s + ox;
      const py = wy * s + oy;
      if (px < -margin || px > w + margin || py < -margin || py > h + margin)
        continue;
      const value = p[i * 3 + 2];
      const death = value > 1.5;
      // Stable per-pellet hash from world position: fixes both the hue and the
      // shimmer phase so a pellet keeps its identity as the camera moves.
      const hash = (wx * 73.13 + wy * 19.37) | 0;
      const pulse = 0.85 + 0.15 * Math.sin(now / 420 + (hash & 63));
      const core = (death ? 6.5 : 4) * Math.max(0.6, s) * pulse;
      const d = core * 2.2;
      if (death) {
        ctx.globalAlpha = 0.95;
        ctx.drawImage(this.deathSprite, px - d, py - d, d * 2, d * 2);
      } else {
        ctx.globalAlpha = 0.78;
        const sprite =
          this.pelletSprites[((hash % nSprites) + nSprites) % nSprites];
        ctx.drawImage(sprite, px - d, py - d, d * 2, d * 2);
      }
    }
    ctx.restore();
  }

  // --- worms ----------------------------------------------------------------

  private drawWormShadow(
    wm: InterpWorm,
    s: number,
    ox: number,
    oy: number,
  ): void {
    const ctx = this.c2d;
    const r = Math.max(2, wm.radius * s);
    ctx.save();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(0,0,0,0.28)";
    ctx.lineWidth = r * 2;
    this.tracePath(ctx, wm, s, ox, oy, r * 0.35, r * 0.5);
    ctx.stroke();
    ctx.restore();
  }

  private drawWormBody(
    wm: InterpWorm,
    s: number,
    ox: number,
    oy: number,
    now: number,
  ): void {
    const ctx = this.c2d;
    const r = Math.max(2, wm.radius * s);

    // Boost halo: an additive, pulsing under-stroke wider than the body.
    if (wm.boosting) {
      const pulse = 0.5 + 0.5 * Math.sin(now / 90);
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.strokeStyle = wm.skin.glow;
      ctx.globalAlpha = 0.22 + 0.18 * pulse;
      ctx.lineWidth = r * 2 + 10 + 6 * pulse;
      ctx.beginPath();
      this.tracePath(ctx, wm, s, ox, oy, 0, 0);
      ctx.stroke();
      ctx.restore();
    }

    // Dark rim: a slightly wider stroke underneath, in the skin's shadow tone.
    ctx.save();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.strokeStyle = wm.skin.rim;
    ctx.lineWidth = r * 2 + Math.max(1.5, r * 0.5);
    ctx.beginPath();
    this.tracePath(ctx, wm, s, ox, oy, 0, 0);
    ctx.stroke();

    // Striped skin: alternate bands of the two hues along the spine. Drawn as a
    // run of round-capped sub-strokes so the bands read as on the tube.
    const band = 5;
    let i = 0;
    while (i < wm.n - 1) {
      const end = Math.min(wm.n - 1, i + band);
      ctx.beginPath();
      ctx.moveTo(wm.xs[i] * s + ox, wm.ys[i] * s + oy);
      for (let j = i + 1; j <= end; j++) {
        ctx.lineTo(wm.xs[j] * s + ox, wm.ys[j] * s + oy);
      }
      ctx.strokeStyle =
        (Math.floor(i / band) & 1) === 0 ? wm.skin.a : wm.skin.b;
      ctx.lineWidth = r * 2;
      ctx.stroke();
      i = end;
    }
    ctx.restore();

    // Glossy top highlight: a thin bright line riding the upper side of the tube.
    ctx.save();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(255,255,255,0.16)";
    ctx.lineWidth = Math.max(1, r * 0.5);
    ctx.beginPath();
    this.tracePath(ctx, wm, s, ox, oy, -r * 0.35, -r * 0.45);
    ctx.stroke();
    ctx.restore();
  }

  /** Trace the worm spine into the current path, optionally offset perpendicular
   * to its local heading (for the shadow/highlight passes). */
  private tracePath(
    ctx: CanvasRenderingContext2D,
    wm: InterpWorm,
    s: number,
    ox: number,
    oy: number,
    offX: number,
    offY: number,
  ): void {
    if (offX === 0 && offY === 0) {
      ctx.moveTo(wm.xs[0] * s + ox, wm.ys[0] * s + oy);
      for (let j = 1; j < wm.n; j++)
        ctx.lineTo(wm.xs[j] * s + ox, wm.ys[j] * s + oy);
      return;
    }
    // Offset along a fixed screen vector derived from the head heading — cheap
    // and good enough for soft shadow/highlight bands.
    const dx = Math.cos(wm.angle);
    const dy = Math.sin(wm.angle);
    const nx = -dy;
    const ny = dx;
    const sx = dx * offX + nx * offY;
    const sy = dy * offX + ny * offY;
    ctx.moveTo(wm.xs[0] * s + ox + sx, wm.ys[0] * s + oy + sy);
    for (let j = 1; j < wm.n; j++)
      ctx.lineTo(wm.xs[j] * s + ox + sx, wm.ys[j] * s + oy + sy);
  }

  private drawWormHead(
    wm: InterpWorm,
    s: number,
    ox: number,
    oy: number,
    w: number,
    h: number,
  ): void {
    const ctx = this.c2d;
    const r = Math.max(2, wm.radius * s);
    const hx = wm.xs[0] * s + ox;
    const hy = wm.ys[0] * s + oy;

    // Head cap with a glossy radial highlight.
    const grad = ctx.createRadialGradient(
      hx - r * 0.3,
      hy - r * 0.3,
      r * 0.1,
      hx,
      hy,
      r * 1.05,
    );
    grad.addColorStop(0, mix(wm.skin.a, "#ffffff", 0.5));
    grad.addColorStop(1, wm.skin.a);
    ctx.fillStyle = wm.skin.rim;
    ctx.beginPath();
    ctx.arc(hx, hy, r * 1.08 + 1, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(hx, hy, r * 1.05, 0, Math.PI * 2);
    ctx.fill();

    // Eyes: sclera on the front-left/right of the head, pupil looking forward.
    const ex = Math.cos(wm.angle);
    const ey = Math.sin(wm.angle);
    const px = -ey;
    const py = ex;
    const eyeOff = r * 0.52;
    const eyeFwd = r * 0.42;
    const scleraR = Math.max(1.5, r * 0.46);
    const pupilR = Math.max(0.8, r * 0.24);
    for (const side of [-1, 1]) {
      const cx = hx + ex * eyeFwd + px * eyeOff * side;
      const cy = hy + ey * eyeFwd + py * eyeOff * side;
      ctx.fillStyle = "#0a0f17";
      ctx.beginPath();
      ctx.arc(cx, cy, scleraR + 1, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#f5f5f5";
      ctx.beginPath();
      ctx.arc(cx, cy, scleraR, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#0a0f17";
      ctx.beginPath();
      ctx.arc(
        cx + ex * scleraR * 0.34,
        cy + ey * scleraR * 0.34,
        pupilR,
        0,
        Math.PI * 2,
      );
      ctx.fill();
    }

    // Name tag floats above the head (kept legible, scaled gently with zoom).
    if (r > 4) {
      const label = wm.isHuman ? "you" : `bot ${wm.seat}`;
      ctx.save();
      ctx.font = `${Math.round(Math.min(16, Math.max(10, r * 0.9)))}px ui-sans-serif, system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      const ty = hy - r * 1.7;
      if (hx > -40 && hx < w + 40 && ty > 0 && ty < h) {
        ctx.fillStyle = "rgba(0,0,0,0.55)";
        ctx.fillText(label, hx, ty + 1);
        ctx.fillStyle = wm.isHuman ? "#dbeeff" : "rgba(255,255,255,0.82)";
        ctx.fillText(label, hx, ty);
      }
      ctx.restore();
    }
  }

  // --- death burst ----------------------------------------------------------

  private spawnDeathBurst(): void {
    const human = this.currSnap?.worms.find((wm) => wm.isHuman);
    if (!human) return;
    const skin = HUMAN_SKIN_BUILT;
    const now = performance.now();
    const step = Math.max(1, Math.floor(human.segCount / 60));
    let order = 0;
    for (let i = 0; i < human.segCount; i += step) {
      const ang = Math.random() * Math.PI * 2;
      const spd = 8 + Math.random() * 26;
      this.bursts.push({
        x: human.segs[i * 2],
        y: human.segs[i * 2 + 1],
        vx: Math.cos(ang) * spd,
        vy: Math.sin(ang) * spd,
        r: 5 + Math.random() * 4,
        color: (i & 8) === 0 ? skin.a : "#ffffff",
        born: now,
        delay: order * 6,
      });
      order++;
    }
  }

  private drawBursts(s: number, ox: number, oy: number, now: number): void {
    if (this.bursts.length === 0) return;
    const ctx = this.c2d;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    let alive = false;
    for (const o of this.bursts) {
      const age = now - o.born - o.delay;
      if (age < 0) {
        alive = true;
        continue;
      }
      const life = age / 900;
      if (life >= 1) continue;
      alive = true;
      const t = age / 1000;
      const x = (o.x + o.vx * t) * s + ox;
      const y = (o.y + o.vy * t) * s + oy;
      const pop = life < 0.12 ? life / 0.12 : 1;
      const r = o.r * s * (1 + life * 0.6) * pop;
      ctx.globalAlpha = (1 - life) * 0.9;
      const g = ctx.createRadialGradient(x, y, 0, x, y, r * 2.4);
      g.addColorStop(0, "#ffffff");
      g.addColorStop(0.4, o.color);
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, r * 2.4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
    if (!alive) this.bursts = [];
  }

  // --- HUD ------------------------------------------------------------------

  private drawHud(game: SlitherGame): void {
    const ctx = this.c2d;
    const w = this.cssW;
    const h = this.cssH;
    const lb = game.leaderboard_blob(); // [seat,isHuman,dead,len] sorted desc
    const rows = Math.floor(lb.length / 4);
    const humanRank = this.humanRank();

    // Leaderboard (top-right).
    const pad = 12;
    const rowH = 20;
    const shown = Math.min(LEADERBOARD_ROWS, rows);
    const panelW = 182;
    const panelH = 30 + shown * rowH;
    const panelX = w - panelW - pad;
    const panelY = pad;
    roundRect(ctx, panelX, panelY, panelW, panelH, 8);
    ctx.fillStyle = "rgba(8,11,18,0.5)";
    ctx.fill();
    ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(220,230,245,0.9)";
    ctx.fillText("Leaderboard", panelX + 12, panelY + 15);
    for (let r = 0; r < shown; r++) {
      const seat = lb[r * 4];
      const isHuman = lb[r * 4 + 1] === 1;
      const dead = lb[r * 4 + 2] === 1;
      const len = Math.round(lb[r * 4 + 3]);
      const skin = skinForSeat(seat, isHuman);
      const ry = panelY + 30 + r * rowH + rowH / 2;
      ctx.fillStyle = skin.a;
      ctx.beginPath();
      ctx.arc(panelX + 16, ry, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = isHuman
        ? "#bfe0ff"
        : dead
          ? "rgba(180,190,205,0.5)"
          : "rgba(225,232,245,0.9)";
      ctx.font = isHuman
        ? "700 12px ui-sans-serif, system-ui, sans-serif"
        : "500 12px ui-sans-serif, system-ui, sans-serif";
      const name = isHuman ? "you" : `bot ${seat}`;
      ctx.fillText(`${r + 1}. ${name}`, panelX + 28, ry);
      ctx.textAlign = "right";
      ctx.fillText(String(len), panelX + panelW - 12, ry);
      ctx.textAlign = "left";
    }

    // Length / rank readout (bottom-center).
    const len = Math.round(game.human_length());
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.font = "800 26px ui-sans-serif, system-ui, sans-serif";
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillText(`length ${len}`, w / 2 + 1, h - 13);
    ctx.fillStyle = "#eaf2ff";
    ctx.fillText(`length ${len}`, w / 2, h - 14);
    ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
    ctx.fillStyle = "rgba(190,205,225,0.8)";
    ctx.fillText(`rank #${humanRank} of ${WORMS}`, w / 2, h - 40);

    // Minimap (bottom-right): round arena, your dot bright, others faint.
    const mm = 132;
    const mmX = w - mm - pad;
    const mmY = h - mm - pad;
    roundRect(ctx, mmX, mmY, mm, mm, 8);
    ctx.fillStyle = "rgba(8,11,18,0.5)";
    ctx.fill();
    const cx = mmX + mm / 2;
    const cy = mmY + mm / 2;
    const rad = mm / 2 - 12;
    ctx.beginPath();
    ctx.arc(cx, cy, rad, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,90,40,0.5)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    const size = game.world_size();
    const worms = this.currSnap?.worms ?? [];
    for (const wm of worms) {
      if (wm.dead || wm.segCount === 0) continue;
      const hx = wm.segs[0] / size;
      const hy = wm.segs[1] / size;
      const dx = (hx - 0.5) * 2 * rad;
      const dy = (hy - 0.5) * 2 * rad;
      if (dx * dx + dy * dy > rad * rad) continue;
      ctx.fillStyle = wm.isHuman ? "#7cc4ff" : skinForSeat(wm.seat, false).a;
      ctx.beginPath();
      ctx.arc(cx + dx, cy + dy, wm.isHuman ? 3.4 : 2.2, 0, Math.PI * 2);
      ctx.fill();
    }

    // Steering hint (top-left), fades implicitly by being small + dim.
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.font = "500 12px ui-sans-serif, system-ui, sans-serif";
    ctx.fillStyle = "rgba(170,185,205,0.6)";
    ctx.fillText("move to steer · hold mouse or space to boost", pad, pad);
  }
}

interface InterpWorm {
  seat: number;
  isHuman: boolean;
  boosting: boolean;
  radius: number;
  angle: number;
  xs: Float32Array;
  ys: Float32Array;
  n: number;
  skin: Skin;
}

/** Paint a soft additive orb sprite (white-hot core → hue → transparent rim)
 * into `cv`, sized `px` square. Used for pellets and the death burst. */
function paintOrb(
  cv: HTMLCanvasElement,
  core: string,
  hue: string,
  px: number,
): void {
  cv.width = px;
  cv.height = px;
  const ctx = cv.getContext("2d")!;
  const c = px / 2;
  const g = ctx.createRadialGradient(c, c, 0, c, c, c);
  g.addColorStop(0, core);
  g.addColorStop(0.32, hue);
  g.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, px, px);
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function randomSeed(): number {
  return (Math.floor(Math.random() * 0x7fff_ffff) | 1) >>> 0;
}
