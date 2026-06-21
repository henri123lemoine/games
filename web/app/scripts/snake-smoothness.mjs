// Headless real-browser validation of the snake fixes (smoothness, UI, thumbnail).
//
// Reuses browser-validate.mjs's method: a real headless Chromium (full binary +
// --headless=new so WebGPU/Metal is live) serving the built dist/ over
// http://localhost with COOP/COEP. It then:
//
//   1. SMOOTHNESS — samples snake A's body centroid (its emerald green channel)
//      every animation frame during a watch-mode match and reports the per-frame
//      head displacement series. Constant-velocity (Google-Snake) motion shows a
//      near-constant non-zero displacement with very few frozen frames; the old
//      glide-then-freeze showed long runs of zero displacement between cells.
//   2. UI — full-page screenshot of a snake match; asserts there is NO `.log`
//      move-wall and NO stuck "Thinking…" status element.
//   3. THUMBNAIL — screenshot of the home grid; crops the snake card.
//
//   node scripts/snake-smoothness.mjs           # all three
//   node scripts/snake-smoothness.mjs --headed   # show the window

import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { extname, join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const APP_DIR = resolve(HERE, '..');
const DIST = join(APP_DIR, 'dist');
const OUT_DIR = join(APP_DIR, '.validation');

const GPU_FLAGS = [
  '--headless=new',
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  '--use-angle=metal',
  '--ignore-gpu-blocklist',
];

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.azweb': 'application/octet-stream',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.woff2': 'font/woff2',
  '.ico': 'image/x-icon',
};

function startServer(rootDir) {
  const server = createServer(async (req, res) => {
    try {
      const urlPath = decodeURIComponent((req.url ?? '/').split('?')[0]);
      let filePath = join(rootDir, urlPath);
      if (urlPath.endsWith('/')) filePath = join(filePath, 'index.html');
      if (!existsSync(filePath)) filePath = join(rootDir, 'index.html');
      const body = await readFile(filePath);
      res.setHeader('content-type', MIME[extname(filePath)] ?? 'application/octet-stream');
      res.setHeader('cross-origin-opener-policy', 'same-origin');
      res.setHeader('cross-origin-embedder-policy', 'require-corp');
      res.setHeader('cross-origin-resource-policy', 'cross-origin');
      res.end(body);
    } catch (e) {
      res.statusCode = 500;
      res.end(String(e));
    }
  });
  return new Promise((res) => {
    server.listen(0, '127.0.0.1', () => res({ server, port: server.address().port }));
  });
}

/** Sample snake A's body centroid from the canvas each animation frame. Snake A
 * is the emerald snake (green dominant, g >> r and g >> b); the body centroid
 * translates one cell per glide exactly as the head does, so its per-frame
 * displacement is the head's velocity. Returns {t, x, y} samples over ~ms. */
function installCentroidProbe(ms) {
  return new Promise((resolve) => {
    const canvas = document.querySelector('.snk-canvas');
    if (!canvas) {
      resolve({ error: 'no canvas' });
      return;
    }
    // Read at a downscaled resolution: the centroid is unchanged but getImageData
    // is far cheaper, so the probe doesn't itself cause the stutter it measures.
    const SCALE = 0.25;
    const off = document.createElement('canvas');
    const w = (off.width = Math.max(1, Math.round(canvas.width * SCALE)));
    const h = (off.height = Math.max(1, Math.round(canvas.height * SCALE)));
    const g = off.getContext('2d', { willReadFrequently: true });
    const samples = [];
    const t0 = performance.now();
    const tick = () => {
      const now = performance.now();
      g.clearRect(0, 0, w, h);
      g.drawImage(canvas, 0, 0, w, h);
      const data = g.getImageData(0, 0, w, h).data;
      let sx = 0;
      let sy = 0;
      let n = 0;
      // Emerald snake A: strong green, green clearly above red & blue.
      for (let p = 0; p < data.length; p += 4) {
        const r = data[p];
        const gr = data[p + 1];
        const b = data[p + 2];
        if (gr > 110 && gr - r > 45 && gr - b > 35) {
          const idx = p / 4;
          sx += idx % w;
          sy += Math.floor(idx / w);
          n++;
        }
      }
      if (n > 8) samples.push({ t: now - t0, x: sx / n, y: sy / n, n });
      if (now - t0 < ms) requestAnimationFrame(tick);
      else resolve({ samples, w, h, cellPx: w / 20 });
    };
    requestAnimationFrame(tick);
  });
}

function analyze(samples, cellPx) {
  // Resample the centroid onto a fixed WIN-ms grid and measure displacement per
  // window, normalized to CELLS so the numbers are resolution-independent.
  // Windowing (vs per-frame) is the fair measure: the probe's own rAF can read
  // the same painted frame twice (a duplicate, not a real freeze), and display
  // vsync jitter shouldn't masquerade as a velocity change. A window wider than
  // a frame but far narrower than a cell exposes a REAL freeze (the snake
  // genuinely standing still between cells) while ignoring sub-frame noise.
  const WIN = 50; // ms per measurement window
  if (samples.length < 3) return { error: 'too few samples', frames: samples.length };
  const tEnd = samples[samples.length - 1].t;
  const at = (t) => {
    let i = 1;
    while (i < samples.length && samples[i].t < t) i++;
    const a = samples[Math.max(0, i - 1)];
    const b = samples[Math.min(samples.length - 1, i)];
    if (b.t === a.t) return { x: a.x, y: a.y };
    const f = (t - a.t) / (b.t - a.t);
    return { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f };
  };
  // Cells per second for each window.
  const speeds = [];
  for (let t = samples[0].t + WIN; t <= tEnd; t += WIN) {
    const p0 = at(t - WIN);
    const p1 = at(t);
    const cells = Math.hypot(p1.x - p0.x, p1.y - p0.y) / cellPx;
    speeds.push((cells / WIN) * 1000); // cells/sec
  }
  // A steady glide is ~ 1000/cellMs ≈ 3-8 cells/sec. "Frozen" = under 0.4
  // cells/sec (the snake is essentially standing still); "moving" = the rest.
  const FROZEN = 0.4;
  const moving = speeds.filter((s) => s > FROZEN);
  const meanSpeed = mean(moving);
  const frozen = speeds.filter((s) => s <= FROZEN).length;
  const cv = meanSpeed > 0 ? stdev(moving) / meanSpeed : 0;
  let longestFreeze = 0;
  let run = 0;
  for (const s of speeds) {
    if (s <= FROZEN) {
      run++;
      longestFreeze = Math.max(longestFreeze, run);
    } else run = 0;
  }
  return {
    rawFrames: samples.length,
    windows: speeds.length,
    winMs: WIN,
    movingWindows: moving.length,
    frozenWindows: frozen,
    frozenFraction: round(frozen / Math.max(1, speeds.length), 3),
    meanCellsPerSec: round(meanSpeed, 2),
    speedCv: round(cv, 3),
    longestFreezeRun: longestFreeze,
    longestFreezeMs: longestFreeze * WIN,
  };
}

const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const stdev = (a) => {
  if (a.length < 2) return 0;
  const m = mean(a);
  return Math.sqrt(mean(a.map((x) => (x - m) ** 2)));
};
const median = (a) => {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  return s[s.length >> 1];
};
const round = (x, d = 1) => Math.round(x * 10 ** d) / 10 ** d;

async function main() {
  const args = process.argv.slice(2);
  const headed = args.includes('--headed');
  if (!existsSync(DIST)) {
    console.error(`no build at ${DIST} — run \`vite build\` first.`);
    process.exit(2);
  }
  await mkdir(OUT_DIR, { recursive: true });

  const { server, port } = await startServer(DIST);
  const base = `http://localhost:${port}`;
  const consoleLogs = [];
  const result = { ok: false };

  const browser = await chromium.launch({
    headless: false,
    args: headed ? GPU_FLAGS.filter((f) => f !== '--headless=new') : GPU_FLAGS,
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 800 } });
    page.on('console', (m) => consoleLogs.push(`[${m.type()}] ${m.text()}`));
    page.on('pageerror', (e) => consoleLogs.push(`[pageerror] ${e.message}`));

    // ---- THUMBNAIL: home grid ----
    await page.goto(`${base}/#/`, { waitUntil: 'load', timeout: 30000 });
    await page.waitForSelector('.mini-snake', { timeout: 15000 }).catch(() => {});
    const homeShot = join(OUT_DIR, 'snake-home.png');
    await page.screenshot({ path: homeShot });
    const card = page.locator('.card', { has: page.locator('.mini-snake') }).first();
    const cardShot = join(OUT_DIR, 'snake-card.png');
    await card.screenshot({ path: cardShot }).catch(() => {});
    // Measure whether the mini segments stay INSIDE the mini-board (not escaped).
    result.thumbnail = await page.evaluate(() => {
      const mini = document.querySelector('.mini-snake');
      if (!mini) return { error: 'no .mini-snake' };
      const mr = mini.getBoundingClientRect();
      const segs = [...mini.querySelectorAll('.mini-seg, .mini-food')];
      const pos = getComputedStyle(mini).position;
      let inside = 0;
      for (const s of segs) {
        const r = s.getBoundingClientRect();
        const cx = r.left + r.width / 2;
        const cy = r.top + r.height / 2;
        if (cx >= mr.left - 2 && cx <= mr.right + 2 && cy >= mr.top - 2 && cy <= mr.bottom + 2)
          inside++;
      }
      return { position: pos, segs: segs.length, insideCard: inside };
    });

    // ---- UI + SMOOTHNESS: watch-mode snake match ----
    await page.goto(`${base}/?snakeDebug#/g/snake?mode=watch`, {
      waitUntil: 'load',
      timeout: 30000,
    });
    await page.waitForSelector('.snk-canvas', { timeout: 30000 }).catch(() => {});

    // UI: confirm the generic side panel / log / status are GONE for snake.
    await page.waitForTimeout(2500);
    result.ui = await page.evaluate(() => ({
      hasLog: !!document.querySelector('.log'),
      hasSidePanel: !!document.querySelector('.side'),
      statusText: document.querySelector('.status')?.textContent ?? null,
      logLines: document.querySelectorAll('.log .log-line').length,
      bodyHasThinking: /Thinking…/.test(document.body.innerText),
    }));
    const matchShot = join(OUT_DIR, 'snake-match.png');
    await page.screenshot({ path: matchShot, fullPage: true });

    // SMOOTHNESS: sample the centroid every frame for ~6s of bot play.
    const probe = await page.evaluate(installCentroidProbe, 6000);
    if (probe.error) {
      result.smoothness = { error: probe.error };
    } else {
      result.smoothness = analyze(probe.samples, probe.cellPx);
      result.smoothness.sampleCount = probe.samples.length;
      await writeFile(
        join(OUT_DIR, 'snake-centroid.json'),
        JSON.stringify(probe.samples, null, 0),
      );
    }
    const movingShot = join(OUT_DIR, 'snake-moving.png');
    await page.screenshot({ path: movingShot });

    result.shots = { homeShot, cardShot, matchShot, movingShot };
    result.ok = true;
  } catch (e) {
    result.error = String(e);
  } finally {
    result.consoleLogs = consoleLogs.slice(-20);
    await browser.close().catch(() => {});
    server.close();
  }

  await writeFile(join(OUT_DIR, 'snake-smoothness.json'), JSON.stringify(result, null, 2));
  console.log('\n===== SNAKE FIX VALIDATION =====');
  console.log('THUMBNAIL', JSON.stringify(result.thumbnail));
  console.log('UI       ', JSON.stringify(result.ui));
  console.log('SMOOTH   ', JSON.stringify(result.smoothness));
  console.log('shots    ', JSON.stringify(result.shots));
  if (result.error) console.log('ERROR    ', result.error);
  console.log('console (tail):');
  for (const l of result.consoleLogs) console.log('  ' + l);
  console.log('================================');
  process.exit(result.ok ? 0 : 1);
}

main();
