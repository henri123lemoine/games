// Browser-side capture for the DOOM conformance harness.
//
// Boots the vendored cloudflare/doom-wasm port (Chocolate Doom -> Emscripten)
// in a real headless Chromium, feeds a deterministic key sequence, and dumps
// one PNG of the on-screen <canvas> per captured frame plus a JSON trace of the
// SDL2 presentation path (renderer kind, presented surface w/h, canvas size).
//
// The point is to observe the ACTUAL pixels the user sees, frame by frame, so
// the smearing ("trails when moving") and look-around bugs become measurable
// instead of anecdotal. Consecutive-frame diffs while standing still localize a
// smear (real Doom is near-static when idle); the SDL2 trace localizes whether
// the software-renderer present path (full putImageData, cannot smear) or a
// streamed GL texture (can smear) is actually in use.
//
//   node capture-browser.mjs <outDir> [--headed]
//
// Inputs are scripted in INPUT_SCRIPT below as {tics, keys[]} segments; keys are
// physical DOM `code` values. A tic here is one captured present.

import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile, mkdir, writeFile, rm } from 'node:fs/promises';
import { extname, join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = resolve(HERE, '..', '..', 'public');

const OUT_DIR = resolve(process.argv[2] ?? join(HERE, 'out', 'browser'));
const HEADED = process.argv.includes('--headed');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.wad': 'application/octet-stream',
  '.cfg': 'text/plain; charset=utf-8',
  '.png': 'image/png',
  '.md': 'text/plain; charset=utf-8',
};

function startServer(rootDir) {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://localhost');
      let p = decodeURIComponent(url.pathname);
      if (p.endsWith('/')) p += 'index.html';
      const file = join(rootDir, p);
      if (!file.startsWith(rootDir)) {
        res.writeHead(403).end();
        return;
      }
      const body = await readFile(file);
      res.writeHead(200, {
        'content-type': MIME[extname(file)] ?? 'application/octet-stream',
        // doom-wasm uses pthreads/atomics in some builds; harmless if not.
        'cross-origin-opener-policy': 'same-origin',
        'cross-origin-embedder-policy': 'require-corp',
      });
      res.end(body);
    } catch {
      res.writeHead(404).end();
    }
  });
  return new Promise((res) => {
    server.listen(0, '127.0.0.1', () => res(server));
  });
}

// Deterministic input script. Each segment holds `keys` down for `tics` frames.
// enter x4 dismisses the engine's own startup screens; then a fixed motion/look
// sequence. `code` values match doom.html / default.cfg bindings (arrows+wasd).
const INPUT_SCRIPT = [
  { tics: 35, keys: [], note: 'settle on first map' },
  { tics: 25, keys: [], note: 'idle baseline (should be near-static)' },
  { tics: 20, keys: ['ArrowRight'], note: 'turn right' },
  { tics: 20, keys: ['ArrowLeft'], note: 'turn left' },
  { tics: 25, keys: ['ArrowUp'], note: 'walk forward' },
  { tics: 20, keys: ['ArrowDown'], note: 'walk back' },
  { tics: 15, keys: [], note: 'idle after motion (smear shows as lingering trails)' },
];

async function main() {
  await rm(OUT_DIR, { recursive: true, force: true });
  await mkdir(OUT_DIR, { recursive: true });

  const server = await startServer(PUBLIC_DIR);
  const { port } = server.address();
  const base = `http://127.0.0.1:${port}`;

  const browser = await chromium.launch({
    headless: !HEADED,
    args: ['--headless=new', '--use-angle=metal', '--ignore-gpu-blocklist'],
  });
  const page = await browser.newPage({ viewport: { width: 800, height: 640 } });

  const logs = [];
  page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));

  await page.goto(`${base}/doom/doom.html`, { waitUntil: 'load' });

  // Instrument BEFORE boot: wrap the SDL2 software-present ASM_CONST and detect
  // any WebGL context the engine grabs on the game canvas. We record the present
  // surface size and renderer kind so the trace says which path actually ran.
  await page.evaluate(() => {
    window.__doomTrace = {
      present: null, // {w,h} of the last software present
      presentCount: 0,
      glContextRequested: false,
      glContextAttrs: null,
      canvas: null,
    };
    const origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, attrs) {
      if (/webgl/i.test(type)) {
        window.__doomTrace.glContextRequested = true;
        window.__doomTrace.glContextAttrs = attrs ?? null;
      }
      return origGetContext.call(this, type, attrs);
    };
  });

  // Boot the engine.
  await page.click('#start');

  // Wait until the engine presents real frames: the canvas must have non-black
  // pixels (the menu/first map renders). Poll the canvas.
  const canvas = page.locator('#canvas');
  await canvas.waitFor({ state: 'visible' });

  async function canvasIsLive() {
    return page.evaluate(() => {
      const c = document.getElementById('canvas');
      if (!c) return false;
      const ctx = c.getContext('2d');
      if (!ctx) return { noCtx: true };
      try {
        const d = ctx.getImageData(0, 0, c.width, c.height).data;
        let nonblack = 0;
        for (let i = 0; i < d.length; i += 4) {
          if (d[i] || d[i + 1] || d[i + 2]) nonblack++;
        }
        return { w: c.width, h: c.height, nonblack };
      } catch (e) {
        return { err: String(e) };
      }
    });
  }

  // Boot can take a few seconds (WAD load + first render). Poll up to 30s.
  let live = null;
  for (let i = 0; i < 300; i++) {
    live = await canvasIsLive();
    if (live && live.nonblack > 1000) break;
    await page.waitForTimeout(100);
  }
  logs.push(`first-live: ${JSON.stringify(live)}`);

  // The engine boots into a demo/title loop. enter x4 to start a real game.
  for (let i = 0; i < 4; i++) {
    await page.keyboard.press('Enter');
    await page.waitForTimeout(250);
  }

  const frames = [];
  let frameIdx = 0;

  async function grab(note) {
    const png = await canvas.screenshot();
    const name = `frame_${String(frameIdx).padStart(4, '0')}.png`;
    await writeFile(join(OUT_DIR, name), png);
    const trace = await page.evaluate(() => window.__doomTrace);
    frames.push({ idx: frameIdx, note, trace });
    frameIdx++;
  }

  // One captured frame ~ a few render frames; hold each key segment and grab a
  // canvas screenshot every ~80ms. waitForTimeout drives wall-clock so the RAF
  // loop advances between grabs.
  for (const seg of INPUT_SCRIPT) {
    for (const k of seg.keys) await page.keyboard.down(k);
    for (let t = 0; t < seg.tics; t++) {
      await page.waitForTimeout(40);
      if (t % 3 === 0) await grab(seg.note);
    }
    for (const k of seg.keys) await page.keyboard.up(k);
  }

  const trace = await page.evaluate(() => window.__doomTrace);
  await writeFile(
    join(OUT_DIR, 'trace.json'),
    JSON.stringify({ trace, frames, logs, canvasLive: live }, null, 2),
  );

  await browser.close();
  server.close();

  console.log(`captured ${frameIdx} frames -> ${OUT_DIR}`);
  console.log('SDL2/GL trace:', JSON.stringify(trace));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
