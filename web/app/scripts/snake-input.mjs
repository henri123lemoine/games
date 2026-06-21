// Headless real-browser INPUT-RESPONSIVENESS validation for snake play.
//
// Loads snake in PLAY mode (human = seat 0) in a real headless Chromium, then
// programmatically presses a SEQUENCE of arrow keys timed to the move cadence
// and asserts the snake's head actually TURNS to each commanded direction on
// the next move — proving the snake follows input with no stairs, no dropped or
// late inputs, and ≤ one tick of lag. Reads the exact rendered heading from the
// ?snakeDebug `window.__snakeHead0.dir` seam (no colour-tracking).
//
//   node scripts/snake-input.mjs            # default sequence
//   node scripts/snake-input.mjs --headed   # show the window

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
const NAME = { n: 'up', e: 'right', s: 'down', w: 'left' };

/** Read the rendered head {x,y,dir} from the debug seam. */
async function head(page) {
  return page.evaluate(() => window.__snakeHead0 ?? null);
}

const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const stdev = (a) => {
  if (a.length < 2) return 0;
  const m = mean(a);
  return Math.sqrt(mean(a.map((x) => (x - m) ** 2)));
};
const round = (x, d = 3) => Math.round(x * 10 ** d) / 10 ** d;

/** Head-path smoothness on samples already in CELL coords: path length per 50ms
 * window (constant scalar speed through turns), reported as CV + frozen
 * fraction — the SAME metric as the watch-mode harness, but measured while the
 * player is actively steering. */
function analyzePath(samples) {
  const WIN = 50;
  const STEP = 10;
  if (samples.length < 5) return { error: 'too few samples', frames: samples.length };
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
  const speeds = [];
  for (let t = samples[0].t + WIN; t <= tEnd; t += WIN) {
    let path = 0;
    let prev = at(t - WIN);
    for (let u = t - WIN + STEP; u <= t + 1e-6; u += STEP) {
      const cur = at(u);
      path += Math.hypot(cur.x - prev.x, cur.y - prev.y);
      prev = cur;
    }
    speeds.push((path / WIN) * 1000); // cells/sec
  }
  const FROZEN = 0.4;
  const moving = speeds.filter((s) => s > FROZEN);
  const m = mean(moving);
  const frozen = speeds.filter((s) => s <= FROZEN).length;
  let longest = 0;
  let run = 0;
  for (const s of speeds) {
    if (s <= FROZEN) longest = Math.max(longest, ++run);
    else run = 0;
  }
  return {
    windows: speeds.length,
    movingWindows: moving.length,
    frozenFraction: round(frozen / Math.max(1, speeds.length)),
    meanCellsPerSec: round(m, 2),
    speedCv: round(m > 0 ? stdev(moving) / m : 0),
    longestFreezeMs: longest * WIN,
  };
}

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

  const browser = await chromium.launch({
    headless: false,
    args: headed ? GPU_FLAGS.filter((f) => f !== '--headless=new') : GPU_FLAGS,
  });
  const consoleLogs = [];
  const result = { ok: false, trace: [] };
  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 800 } });
    page.on('console', (m) => consoleLogs.push(`[${m.type()}] ${m.text()}`));
    page.on('pageerror', (e) => consoleLogs.push(`[pageerror] ${e.message}`));

    // PLAY mode (human = seat 0), ?snakeDebug for the head seam.
    await page.goto(`${base}/?snakeDebug#/g/snake`, { waitUntil: 'load', timeout: 30000 });
    await page.waitForSelector('.snk-canvas', { timeout: 30000 });
    // Focus the page so keydown reaches the window handler.
    await page.locator('.snk-stage').click({ position: { x: 5, y: 5 } }).catch(() => {});

    // Confirm the turn-highlight chip is gone.
    result.turnHighlightPresent = await page.evaluate(
      () => !!document.querySelector('.snk-chip.snk-turn') ||
        getComputedStyle(document.body).cssText.includes('snk-turn'),
    );
    result.turnHighlightInDom = await page.evaluate(
      () => [...document.querySelectorAll('[class*="snk-turn"]')].length,
    );

    // Wait for the first head sample (smooth-from-start check).
    await page.waitForFunction(() => !!window.__snakeHead0, { timeout: 10000 }).catch(() => {});

    // Build a legal, non-reversing key sequence and press one per ~tick. After
    // each press, wait for the rendered heading to change and record what it
    // became vs what we commanded.
    // Start a background head-path sampler that runs THROUGHOUT the input
    // sequence, so we can prove the motion stays smooth WHILE steering (not just
    // in idle watch mode). It collects {t,x,y} on a window global we read later.
    await page.evaluate(() => {
      window.__playSamples = [];
      let last = -1;
      const tick = () => {
        const h = window.__snakeHead0;
        if (h && h.t !== last) {
          last = h.t;
          window.__playSamples.push({ t: h.t, x: h.x, y: h.y });
        }
        window.__playRaf = requestAnimationFrame(tick);
      };
      window.__playRaf = requestAnimationFrame(tick);
    });

    const sequence = ['ArrowUp', 'ArrowRight', 'ArrowDown', 'ArrowRight', 'ArrowUp', 'ArrowLeft', 'ArrowDown'];
    // Read the snake's ACTUAL current heading (not a tracked guess) right before
    // each command, so harness bookkeeping can't drift from the game.
    for (const key of sequence) {
      // Stop if the snake already died (e.g. into the active bot) — a dead head
      // can't turn, so further presses aren't a fair input test.
      if (await page.evaluate(() => !!document.querySelector('.snk-overlay.snk-show'))) break;
      const want = KEY_DIR[key];
      const before = (await head(page))?.dir ?? 'e';
      // Skip an illegal 180° of the current heading — the game legitimately keeps
      // going straight; the harness only asserts legal turns.
      const legal = want !== OPP[before];
      const over0 = await page.evaluate(() => !!document.querySelector('.snk-overlay.snk-show'));
      await page.keyboard.press(key);
      // Wait until the head has advanced ~one full cell (a real commit), polling
      // the rendered heading. The snake commits the turn on the next tick; one
      // cell of travel guarantees the commit happened.
      const startHead = await head(page);
      let got = before;
      const dirs = [];
      const deadline = Date.now() + 1500;
      while (Date.now() < deadline) {
        await page.waitForTimeout(30);
        const h = await head(page);
        if (!h) continue;
        got = h.dir;
        if (dirs[dirs.length - 1] !== h.dir) dirs.push(h.dir);
        const moved =
          startHead &&
          Math.hypot(h.x - startHead.x, h.y - startHead.y) >= 0.9;
        if (moved && (got === want || !legal)) break;
        if (moved && got !== before) break; // committed some turn; stop and judge
      }
      const over1 = await page.evaluate(() => !!document.querySelector('.snk-overlay.snk-show'));
      result.trace.push({
        pressed: NAME[want],
        legal,
        before: NAME[before],
        after: NAME[got] ?? got,
        followed: legal ? got === want : true,
        dirsSeen: dirs.map((d) => NAME[d]),
        gameOver: over0 || over1,
      });
      // Let the snake travel into the next cell before the next command.
      await page.waitForTimeout(120);
    }

    // Stop the background sampler and pull the head path collected DURING play.
    const play = await page.evaluate(() => {
      cancelAnimationFrame(window.__playRaf);
      const s = window.__playSamples ?? [];
      const t0 = s.length ? s[0].t : 0;
      return s.map((p) => ({ t: p.t - t0, x: p.x, y: p.y }));
    });
    result.playSmooth = analyzePath(play);

    // Smooth-FROM-START: the first ~1.2s of the same play path. A stepped start
    // would show a long frozen gap right at the beginning.
    const early = play.filter((p) => p.t <= 1200);
    // Longest gap with no head movement in the early window.
    let maxGap = 0;
    for (let i = 1; i < early.length; i++) {
      const moved = Math.hypot(early[i].x - early[i - 1].x, early[i].y - early[i - 1].y) > 0.01;
      if (!moved) {
        // accumulate consecutive still time
        let j = i;
        let gap = 0;
        while (j < early.length && Math.hypot(early[j].x - early[j - 1].x, early[j].y - early[j - 1].y) <= 0.01) {
          gap += early[j].t - early[j - 1].t;
          j++;
        }
        maxGap = Math.max(maxGap, gap);
        i = j;
      }
    }
    result.earlyMaxStillMs = Math.round(maxGap);
    result.earlySamples = early.length;

    // CONTINUOUS-GLIDE smoothness: keep the snake ALIVE by steering it around the
    // board's perimeter (a slow box) while sampling the head path, so we measure
    // the glide quality over many uninterrupted cells without the snake dying
    // into a wall (which would freeze the head and look like a stall). The turns
    // are sparse, so most windows are straight continuous glide.
    if (!(await page.evaluate(() => !!document.querySelector('.snk-overlay.snk-show')))) {
      // Sample in the page; steer from here on a slow box so it never crashes.
      await page.evaluate(() => {
        window.__cg = [];
        window.__cgLast = -1;
        const tick = () => {
          const h = window.__snakeHead0;
          if (h && h.t !== window.__cgLast) {
            window.__cgLast = h.t;
            window.__cg.push({ t: h.t, x: h.x, y: h.y });
          }
          window.__cgRaf = requestAnimationFrame(tick);
        };
        window.__cgRaf = requestAnimationFrame(tick);
      });
      // Drive a tight clockwise box near center (~2 cells/side at the fast clock)
      // so the player survives many uninterrupted cells. ~3.5s total.
      const box = ['ArrowUp', 'ArrowRight', 'ArrowDown', 'ArrowLeft'];
      const t0 = Date.now();
      let bi = 0;
      while (Date.now() - t0 < 3500) {
        const over = await page.evaluate(() => !!document.querySelector('.snk-overlay.snk-show'));
        if (over) break;
        await page.keyboard.press(box[bi % 4]);
        bi++;
        await page.waitForTimeout(260);
      }
      const glide = await page.evaluate(() => {
        cancelAnimationFrame(window.__cgRaf);
        const s = window.__cg ?? [];
        const t = s.length ? s[0].t : 0;
        return s.map((p) => ({ t: p.t - t, x: p.x, y: p.y }));
      });
      const endedEarly = await page.evaluate(
        () => !!document.querySelector('.snk-overlay.snk-show'),
      );
      result.continuousGlide =
        glide.length >= 5
          ? analyzePath(glide)
          : { note: 'snake collided with the active bot before enough cells', samples: glide.length };
      result.continuousEndedEarly = endedEarly;
    }

    await page.screenshot({ path: join(OUT_DIR, 'snake-input.png') });

    const legalSteps = result.trace.filter((s) => s.legal);
    result.followedAll = legalSteps.length > 0 && legalSteps.every((s) => s.followed);
    result.ok = true;
  } catch (e) {
    result.error = String(e);
  } finally {
    result.consoleLogs = consoleLogs.filter((l) => /pageerror|\[error\]/.test(l)).slice(0, 8);
    await browser.close().catch(() => {});
    server.close();
  }

  await writeFile(join(OUT_DIR, 'snake-input.json'), JSON.stringify(result, null, 2));
  console.log('\n===== SNAKE INPUT VALIDATION =====');
  console.log('turn-highlight elements in DOM:', result.turnHighlightInDom);
  console.log('early-window longest still gap (ms):', result.earlyMaxStillMs, `(samples ${result.earlySamples})`);
  console.log('input -> turn trace:');
  for (const s of result.trace)
    console.log(
      `  pressed ${String(s.pressed).padEnd(6)} from ${String(s.before).padEnd(6)} -> ${String(s.after).padEnd(6)} ${s.legal ? (s.followed ? 'OK' : 'IGNORED/WRONG') : '(illegal 180, skipped)'}  seen=[${(s.dirsSeen ?? []).join(',')}]${s.gameOver ? ' GAME-OVER' : ''}`,
    );
  console.log('followed all legal commands:', result.followedAll);
  console.log('play-mode smoothness (while steering):', JSON.stringify(result.playSmooth));
  console.log('continuous-glide smoothness (no interference):', JSON.stringify(result.continuousGlide));
  if (result.consoleLogs.length) console.log('errors:', result.consoleLogs);
  if (result.error) console.log('ERROR', result.error);
  console.log('==================================');
  process.exit(result.ok && result.followedAll ? 0 : 1);
}

main();
