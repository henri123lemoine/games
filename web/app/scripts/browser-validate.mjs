// Headless real-browser validation harness for the arcade.
//
// Launches a real headless Chromium with WebGPU enabled, serves the built app,
// drives one game, and reports the backend that actually ran (GPU vs CPU), real
// ms/move, every console log (incl. any WebGPU-init failure), the page's
// navigator.gpu adapter info, and a screenshot — so browser/GPU/visual work can
// be validated without a human reloading localhost.
//
//   npm run validate:browser            # default game: snake
//   npm run validate:browser -- snake   # snake card vs the AlphaZero bot
//   npm run validate:browser -- coil    # coil standalone
//   node scripts/browser-validate.mjs snake --headed   # show the window
//
// WHY full Chromium + --headless=new and not headless:true:
//   Playwright's headless:true swaps in `chrome-headless-shell`, a stripped
//   binary that ships NO WebGPU (navigator.gpu is undefined there). The full
//   Chromium binary run with --headless=new exposes a real Metal adapter while
//   staying windowless. navigator.gpu is also gated on a SECURE CONTEXT, so the
//   app must be served over http://localhost (a file:// or data: URL has no
//   navigator.gpu — that, not headless, is the usual "no GPU" red herring).

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

// WebGPU on Metal, kept windowless via --headless=new (see header).
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

/** Static file server for the built app. SPA-style fallback to index.html so a
 * deep hash route still loads (the hash never reaches the server anyway, but a
 * bad path shouldn't 404 the shell). */
function startServer(rootDir) {
  const server = createServer(async (req, res) => {
    try {
      const urlPath = decodeURIComponent((req.url ?? '/').split('?')[0]);
      let filePath = join(rootDir, urlPath);
      if (urlPath.endsWith('/')) filePath = join(filePath, 'index.html');
      if (!existsSync(filePath)) filePath = join(rootDir, 'index.html');
      const body = await readFile(filePath);
      res.setHeader('content-type', MIME[extname(filePath)] ?? 'application/octet-stream');
      // The wasm engines want cross-origin isolation for SharedArrayBuffer; set
      // the headers so the served build matches a COOP/COEP host.
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

/** Per-game knobs: the hash route to load (watch mode lets both seats be bots so
 * moves auto-flow with no human input) and how we read its on-page state. */
const GAMES = {
  snake: {
    // ?snakeDebug (search, read by debugEnabled) + watch route (hash).
    route: '/?snakeDebug#/g/snake?mode=watch',
    overlaySelector: '.snk-debug',
    canvasSelector: '.snk-canvas',
    ready: '.snk-canvas',
  },
  coil: {
    route: '/#/coil',
    overlaySelector: null,
    canvasSelector: 'canvas',
    ready: 'canvas',
  },
};

async function pageGpuInfo(page) {
  return page.evaluate(async () => {
    const out = { hasGpu: 'gpu' in navigator, secureContext: isSecureContext };
    if (!out.hasGpu) return out;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      out.adapter = !!adapter;
      if (adapter) {
        const i = adapter.info ?? null;
        out.info = i
          ? { vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description }
          : null;
      }
    } catch (e) {
      out.adapterError = String(e);
    }
    return out;
  });
}

async function main() {
  const args = process.argv.slice(2);
  const headed = args.includes('--headed');
  const game = args.find((a) => !a.startsWith('-')) ?? 'snake';
  const cfg = GAMES[game];
  if (!cfg) {
    console.error(`unknown game "${game}"; known: ${Object.keys(GAMES).join(', ')}`);
    process.exit(2);
  }
  if (!existsSync(DIST)) {
    console.error(`no build at ${DIST} — run \`npx vite build\` first (npm run validate:browser builds for you).`);
    process.exit(2);
  }
  await mkdir(OUT_DIR, { recursive: true });

  const { server, port } = await startServer(DIST);
  const base = `http://localhost:${port}`;
  const url = base + cfg.route;
  console.log(`[validate] serving dist/ at ${base}`);
  console.log(`[validate] game=${game} url=${url} headed=${headed}`);

  const browser = await chromium.launch({
    headless: false, // full Chromium binary; --headless=new keeps it windowless
    args: headed ? GPU_FLAGS.filter((f) => f !== '--headless=new') : GPU_FLAGS,
  });

  const consoleLogs = [];
  const result = { game, url, ok: false };
  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 800 } });
    page.on('console', (m) => consoleLogs.push(`[${m.type()}] ${m.text()}`));
    page.on('pageerror', (e) => consoleLogs.push(`[pageerror] ${e.message}`));

    await page.goto(url, { waitUntil: 'load', timeout: 30000 });
    result.gpu = await pageGpuInfo(page);

    // Wait for the game to mount, then let the bots play several moves.
    await page.waitForSelector(cfg.ready, { timeout: 30000 }).catch(() => {});

    // Sample the overlay across the match so we report a frame with the bot
    // actually mid-move, not the idle end-screen (a short match can finish in a
    // few seconds; we keep the snapshot whose text mentions a real backend).
    const shot = join(OUT_DIR, `${game}.png`);
    const overlay = cfg.overlaySelector ? page.locator(cfg.overlaySelector) : null;
    let best = '';
    for (let i = 0; i < 10; i++) {
      await page.waitForTimeout(800);
      if (overlay) {
        const txt = (await overlay.textContent().catch(() => '')) ?? '';
        // Prefer a frame where the bot has reported a move (has "bot  GPU/CPU"
        // and isn't idle) — that is the moment worth screenshotting.
        if (/bot\s+(GPU|CPU)/i.test(txt) && !/idle/i.test(txt)) {
          best = txt;
          await page.screenshot({ path: shot });
        }
      } else if (i === 4) {
        await page.screenshot({ path: shot });
      }
    }
    if (overlay) {
      result.overlayVisible = await overlay.isVisible().catch(() => false);
      result.overlayText = best || ((await overlay.textContent().catch(() => '')) ?? '');
    }
    if (!existsSync(shot)) await page.screenshot({ path: shot });
    result.screenshot = shot;
    result.ok = true;
  } catch (e) {
    result.error = String(e);
  } finally {
    result.consoleLogs = consoleLogs;
    await browser.close().catch(() => {});
    server.close();
  }

  // Backend the snake overlay says actually ran, parsed from "bot   GPU/CPU".
  const m = (result.overlayText ?? '').match(/bot\s+(GPU|CPU)/i);
  result.backend = m ? m[1].toUpperCase() : null;
  const mv = (result.overlayText ?? '').match(/move\s+(\d+)\s*ms/i);
  result.msPerMove = mv ? Number(mv[1]) : null;

  await writeFile(join(OUT_DIR, `${game}.json`), JSON.stringify(result, null, 2));

  console.log('\n===== VALIDATION REPORT =====');
  console.log(`game           ${result.game}`);
  console.log(`page navigator.gpu ${JSON.stringify(result.gpu)}`);
  if (cfg.overlaySelector) {
    console.log(`overlay visible ${result.overlayVisible}`);
    console.log(`overlay text   ${JSON.stringify(result.overlayText)}`);
    console.log(`backend (overlay) ${result.backend}`);
    console.log(`ms/move (overlay) ${result.msPerMove}`);
  }
  console.log(`screenshot     ${result.screenshot}`);
  console.log('console logs:');
  for (const l of consoleLogs) console.log('  ' + l);
  if (result.error) console.log(`ERROR          ${result.error}`);
  console.log('=============================');
  process.exit(result.ok ? 0 : 1);
}

main();
