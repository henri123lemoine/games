// Headless real-browser DECOUPLING proof for snake play: the player's input
// latency and tick rate MUST be independent of the bot's compute.
//
// Runs the input-latency + tick-rate measurement twice in a real headless
// Chromium: once with the normal bot, once with `?snakeSlowBot=500` (a 500ms
// delay injected into every search batch — a deliberately crippled bot). If the
// player is truly decoupled, both runs show the SAME keypress→turn latency
// (≤ one ~120ms tick) and the SAME player tick rate. That side-by-side is the
// primary evidence.
//
//   node scripts/snake-decouple.mjs

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
  '.css': 'text/css; charset=utf-8',
  '.wasm': 'application/wasm',
  '.azweb': 'application/octet-stream',
  '.json': 'application/json; charset=utf-8',
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

const KEY_DIR = { ArrowUp: 'n', ArrowRight: 'e', ArrowDown: 's', ArrowLeft: 'w' };
const OPP = { n: 's', s: 'n', e: 'w', w: 'e' };
const PERP = { n: ['e', 'w'], s: ['e', 'w'], e: ['n', 's'], w: ['n', 's'] };
const KEY_OF = { n: 'ArrowUp', e: 'ArrowRight', s: 'ArrowDown', w: 'ArrowLeft' };

const head = (page) => page.evaluate(() => window.__snakeHead0 ?? null);
const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const median = (a) => {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  return s[s.length >> 1];
};
const round = (x, d = 1) => Math.round(x * 10 ** d) / 10 ** d;

/** Press a sequence of legal perpendicular turns and measure, for each, the
 * wall-clock from keypress to the rendered heading actually changing. Also
 * measures the player's cell tick rate from head displacement over time. Keeps
 * the snake from a wall by always turning toward board center. */
async function measure(page) {
  const latencies = [];
  const turns = [];
  const tickStart = Date.now();
  // Start a background head-path sampler (for tick-rate) that runs the whole time.
  await page.evaluate(() => {
    window.__dp = [];
    let last = -1;
    const tick = () => {
      const h = window.__snakeHead0;
      if (h && h.t !== last) {
        last = h.t;
        window.__dp.push({ t: h.t, x: h.x, y: h.y });
      }
      window.__dpRaf = requestAnimationFrame(tick);
    };
    window.__dpRaf = requestAnimationFrame(tick);
  });
  let cur = (await head(page))?.dir ?? 'e';
  for (let n = 0; n < 10; n++) {
    const over = await page.evaluate(() => !!document.querySelector('.snk-overlay.snk-show'));
    if (over) break;
    const h0 = await head(page);
    if (!h0) break;
    // Turn toward center on the axis perpendicular to the current heading, to
    // stay alive across many turns.
    const center = 9;
    const [a, b] = PERP[cur];
    const want =
      (a === 'n' && h0.y > center) || (a === 's' && h0.y < center) ||
      (a === 'e' && h0.x < center) || (a === 'w' && h0.x > center)
        ? a
        : b;
    if (want === OPP[cur] || want === cur) {
      await page.waitForTimeout(150);
      continue;
    }
    const tPress = Date.now();
    await page.keyboard.press(KEY_OF[want]);
    let turned = null;
    const deadline = Date.now() + 700;
    while (Date.now() < deadline) {
      const h = await head(page);
      if (h && h.dir === want) {
        turned = Date.now();
        break;
      }
      await page.waitForTimeout(6);
    }
    if (turned) {
      latencies.push(turned - tPress);
      turns.push({ want, ok: true });
      cur = want;
    } else {
      turns.push({ want, ok: false });
    }
    await page.waitForTimeout(200); // travel ~1.5 cells before the next turn
  }
  // Pull the path; compute tick rate over the ALIVE span (sampled while moving).
  const path = await page.evaluate(() => {
    cancelAnimationFrame(window.__dpRaf);
    return window.__dp ?? [];
  });
  let len = 0;
  for (let i = 1; i < path.length; i++)
    len += Math.hypot(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y);
  const span = path.length > 1 ? path[path.length - 1].t - path[0].t : 0;
  const cellsPerSec = span > 0 ? (len / span) * 1000 : 0;
  return {
    elapsedMs: Date.now() - tickStart,
    turns: turns.length,
    followed: turns.filter((t) => t.ok).length,
    latencyMs: {
      samples: latencies.length,
      median: round(median(latencies)),
      mean: round(mean(latencies)),
      max: round(Math.max(0, ...latencies)),
    },
    tickCellsPerSec: round(cellsPerSec, 2),
    pathSamples: path.length,
  };
}

async function runOnce(base, slowBotMs) {
  const browser = await chromium.launch({ headless: false, args: GPU_FLAGS });
  const consoleLogs = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 800 } });
    page.on('console', (m) => consoleLogs.push(`[${m.type()}] ${m.text()}`));
    page.on('pageerror', (e) => consoleLogs.push(`[pageerror] ${e.message}`));
    const slow = slowBotMs > 0 ? `&snakeSlowBot=${slowBotMs}` : '';
    await page.goto(`${base}/?snakeDebug${slow}#/g/snake`, { waitUntil: 'load', timeout: 30000 });
    await page.waitForSelector('.snk-canvas', { timeout: 30000 });
    await page.locator('.snk-stage').click({ position: { x: 5, y: 5 } }).catch(() => {});
    await page.waitForFunction(() => !!window.__snakeHead0, { timeout: 10000 }).catch(() => {});
    const res = await measure(page);
    res.errors = consoleLogs.filter((l) => /pageerror|\[error\]/.test(l)).slice(0, 5);
    return res;
  } finally {
    await browser.close().catch(() => {});
  }
}

async function main() {
  if (!existsSync(DIST)) {
    console.error(`no build at ${DIST} — run \`vite build\` first.`);
    process.exit(2);
  }
  await mkdir(OUT_DIR, { recursive: true });
  const { server, port } = await startServer(DIST);
  const base = `http://localhost:${port}`;

  const fast = await runOnce(base, 0);
  const slow = await runOnce(base, 500);
  server.close();

  const out = { fast, slow };
  await writeFile(join(OUT_DIR, 'snake-decouple.json'), JSON.stringify(out, null, 2));

  // Decoupled iff slowing the bot didn't materially change player latency or tick rate.
  const latOk =
    fast.latencyMs.median <= 160 &&
    slow.latencyMs.median <= 160 &&
    Math.abs(slow.latencyMs.median - fast.latencyMs.median) <= 60;
  const tickOk =
    fast.tickCellsPerSec > 0 &&
    slow.tickCellsPerSec > 0 &&
    Math.abs(slow.tickCellsPerSec - fast.tickCellsPerSec) / fast.tickCellsPerSec <= 0.25;
  const pass = latOk && tickOk;

  console.log('\n===== SNAKE DECOUPLING PROOF =====');
  console.log('FAST bot:', JSON.stringify(fast.latencyMs), 'tick', fast.tickCellsPerSec, 'cells/s', `(turns ${fast.followed}/${fast.turns})`);
  console.log('SLOW bot (+500ms/batch):', JSON.stringify(slow.latencyMs), 'tick', slow.tickCellsPerSec, 'cells/s', `(turns ${slow.followed}/${slow.turns})`);
  console.log('latency unchanged by slow bot:', latOk);
  console.log('tick rate unchanged by slow bot:', tickOk);
  console.log('DECOUPLED:', pass);
  if (fast.errors?.length) console.log('fast errors:', fast.errors);
  if (slow.errors?.length) console.log('slow errors:', slow.errors);
  console.log('==================================');
  process.exit(pass ? 0 : 1);
}

main();
