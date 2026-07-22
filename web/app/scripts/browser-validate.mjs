// Headless real-browser validation harness for the arcade.
//
// Launches a real headless Chromium with WebGPU enabled, serves the built app,
// drives one game, records console failures and a screenshot, and exercises
// Snake's human-facing input and geometry regressions without timing gameplay.
//
//   npm run validate:browser            # default game: snake
//   npm run validate:browser -- snake   # canonical Battlesnake vs BNS bots
//   npm run validate:browser -- coil    # coil standalone
//   npm run validate:browser -- settings # UI option -> worker/runtime wiring
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
import { readFile, mkdir, rm, writeFile } from 'node:fs/promises';
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

/** Per-game knobs: the hash route to load and how we read its on-page state. */
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
  settings: {
    route: '/#/g/chess',
    overlaySelector: null,
    canvasSelector: null,
    ready: '.seat-level',
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

const KEY_DIR = {
  ArrowUp: 'n',
  ArrowRight: 'e',
  ArrowDown: 's',
  ArrowLeft: 'w',
};

function towardCenter(head) {
  const dx = 5 - head.x;
  const dy = 5 - head.y;
  const first = Math.abs(dx) >= Math.abs(dy)
    ? (dx >= 0 ? 'ArrowRight' : 'ArrowLeft')
    : (dy >= 0 ? 'ArrowDown' : 'ArrowUp');
  const second = first === 'ArrowLeft' || first === 'ArrowRight'
    ? (dy >= 0 ? 'ArrowDown' : 'ArrowUp')
    : (dx >= 0 ? 'ArrowRight' : 'ArrowLeft');
  return [first, second];
}

function towardNearestEdge(head) {
  const candidates = [
    [head.x, 'ArrowLeft'],
    [10 - head.x, 'ArrowRight'],
    [head.y, 'ArrowUp'],
    [10 - head.y, 'ArrowDown'],
  ];
  candidates.sort((a, b) => a[0] - b[0]);
  return candidates[0][1];
}

async function snakeHead(page) {
  return page.evaluate(() => window.__snakeHead0 ?? null);
}

async function openHumanSnake(page, base) {
  await page.goto(`${base}/?snakeDebug#/g/snake`, { waitUntil: 'load', timeout: 30000 });
  await page.waitForSelector('.snk-canvas', { timeout: 30000 });
  await page.waitForFunction(() => !!window.__snakeHead0, null, { timeout: 10000 });
}

/** Regression checks for the human-facing bugs that a watch-mode screenshot
 * cannot exercise: start gating, buffered keys, off-board death, and torus
 * seams. The debug seam exposes rendered geometry, not engine-only state. */
async function validateSnakeInteractions(page, base) {
  await openHumanSnake(page, base);
  const initial = await snakeHead(page);
  const initialTurn = await page.locator('.snk-debug').textContent();
  const startPrompt = await page.locator('.snk-overlay').textContent();
  const pace = await page.locator('.speed').inputValue();
  const food = await page.locator('.qc-select[data-key="food"]').inputValue();
  await page.waitForTimeout(650);
  const heldTurn = await page.locator('.snk-debug').textContent();
  if (!/turn\s+0/.test(initialTurn) || !/turn\s+0/.test(heldTurn)) {
    throw new Error(`human game advanced before input: ${JSON.stringify({ initialTurn, heldTurn })}`);
  }
  if (!/choose your first move/i.test(startPrompt)) throw new Error('missing first-move gate');
  if (pace !== '1') throw new Error(`default human pace is ${pace}, expected normal`);
  if (food !== 'one') throw new Error(`default food preset is ${food}, expected one`);

  const [firstKey, secondKey] = towardCenter(initial);
  await page.keyboard.press(firstKey);
  await page.keyboard.press(secondKey);
  const expectedDirs = [KEY_DIR[firstKey], KEY_DIR[secondKey]];
  await page.waitForFunction(
    () => getComputedStyle(document.querySelector('.snk-overlay')).opacity === '0',
    null,
    { timeout: 1000 },
  );
  const observedDirs = [];
  for (let i = 0; i < 80; i++) {
    await page.waitForTimeout(25);
    const head = await snakeHead(page);
    if (head && observedDirs.at(-1) !== head.dir) observedDirs.push(head.dir);
    const firstAt = observedDirs.indexOf(expectedDirs[0]);
    if (firstAt >= 0 && observedDirs.indexOf(expectedDirs[1], firstAt + 1) >= 0) break;
  }
  const firstAt = observedDirs.indexOf(expectedDirs[0]);
  const secondAt = observedDirs.indexOf(expectedDirs[1], firstAt + 1);
  if (firstAt < 0 || secondAt < 0) {
    throw new Error(`buffered turns were not applied in order: ${observedDirs}`);
  }
  const inputShot = join(OUT_DIR, 'snake-buffered-turns.png');
  await page.screenshot({ path: inputShot });

  // Standard wall death: the rendered head/body must remain finite and local.
  await openHumanSnake(page, base);
  await page.locator('.speed').selectOption('0.7');
  const deathStart = await snakeHead(page);
  await page.keyboard.press(towardNearestEdge(deathStart));
  await page.waitForFunction(
    () => window.__snakeHead0 && window.__snakeHead0.alive === false,
    null,
    { timeout: 5000 },
  );
  const dead = await snakeHead(page);
  if (
    !Number.isFinite(dead.x) ||
    !Number.isFinite(dead.y) ||
    dead.x < -1 ||
    dead.x > 11 ||
    dead.y < -1 ||
    dead.y > 11 ||
    dead.maxLink > 1.5
  ) throw new Error(`death geometry escaped the board: ${JSON.stringify(dead)}`);

  // Wrapped crossing: adjacent rendered segments must stay local while the
  // modulo head coordinate crosses the seam.
  await openHumanSnake(page, base);
  await page.locator('.qc-select[data-key="mode"]').selectOption('wrapped');
  await page.waitForFunction(
    () => document.querySelector('.snk-overlay')?.textContent?.toLowerCase().includes('choose'),
    null,
    { timeout: 10000 },
  );
  await page.locator('.speed').selectOption('0.7');
  const wrapStart = await snakeHead(page);
  await page.keyboard.press(towardNearestEdge(wrapStart));
  let previous = wrapStart;
  let crossed = false;
  let largestLink = 0;
  for (let i = 0; i < 140 && !crossed; i++) {
    await page.waitForTimeout(25);
    const head = await snakeHead(page);
    if (!head) continue;
    largestLink = Math.max(largestLink, head.maxLink ?? 0);
    if (Math.abs(head.x - previous.x) > 5 || Math.abs(head.y - previous.y) > 5) crossed = true;
    previous = head;
  }
  if (!crossed) throw new Error('wrapped snake never crossed a seam');
  if (largestLink > 1.5) throw new Error(`wrapped body spanned ${largestLink.toFixed(2)} cells`);

  return {
    startHeld: true,
    normalPace: true,
    oneFoodDefault: true,
    bufferedDirections: expectedDirs,
    deathHead: { x: dead.x, y: dead.y, maxLink: dead.maxLink },
    wrappedCrossing: true,
    wrappedMaxLink: largestLink,
    inputScreenshot: inputShot,
  };
}

/** Observe the engine RPCs without modifying them. This catches the class of
 * bug where the rendered selector and `create` request look correct while the
 * separately constructed client-side bot receives a default instead. */
async function installSettingsProbe(page) {
  await page.addInitScript(() => {
    const NativeWorker = window.Worker;
    let nextWorker = 1;
    window.__engineRequests = [];
    window.Worker = new Proxy(NativeWorker, {
      construct(target, args) {
        const worker = new target(...args);
        const workerId = nextWorker++;
        const postMessage = worker.postMessage;
        worker.postMessage = function (message, transfer) {
          if (message?.op === 'create') {
            window.__engineRequests.push({ worker: workerId, op: message.op, opts: { ...message.opts } });
          } else if (message?.op === 'azNew' || message?.op === 'goNew' || message?.op === 'penteNew') {
            window.__engineRequests.push({
              worker: workerId,
              op: message.op,
              sims: message.sims,
              size: message.size,
              vcfDepth: message.vcfDepth,
              vcfNodes: message.vcfNodes,
            });
          }
          return postMessage.call(this, message, transfer);
        };
        return worker;
      },
    });
  });
}

async function clearRequests(page) {
  await page.evaluate(() => { window.__engineRequests = []; });
}

async function requests(page) {
  return page.evaluate(() => window.__engineRequests);
}

async function waitForRequests(page, op, count) {
  await page.waitForFunction(
    ({ wantedOp, wantedCount }) =>
      window.__engineRequests.filter((request) => request.op === wantedOp).length >= wantedCount,
    { wantedOp: op, wantedCount: count },
    { timeout: 30000 },
  );
}

async function validateSettings(page, base, gpu) {
  // Human vs Chess: changing the opponent's level serializes a per-seat spec;
  // the separately constructed client bot must receive the same budget.
  await page.waitForSelector('.seat-level', { timeout: 30000 });
  await waitForRequests(page, 'azNew', 1);
  await page.waitForTimeout(250);
  await clearRequests(page);
  await page.locator('.seat-level').selectOption('12000');
  await waitForRequests(page, 'create', 1);
  await waitForRequests(page, 'azNew', 1);
  const chess = await requests(page);
  const chessCreate = chess.find((request) => request.op === 'create');
  const chessNew = chess.find((request) => request.op === 'azNew');
  if (!chessCreate?.opts?.bots?.includes('sims=12000'))
    throw new Error(`Chess Hard missing from bots spec: ${JSON.stringify(chessCreate)}`);
  const chessExpected = gpu.adapter ? 12000 : 16;
  if (chessNew?.sims !== chessExpected)
    throw new Error(`Chess runtime sims=${chessNew?.sims}, expected ${chessExpected}`);

  // Watch mode: distinct seats require distinct search instances. Changing
  // only seat 0 to Trivial must leave seat 1 at Medium.
  await page.goto(`${base}/#/g/chess?mode=watch`, { waitUntil: 'load', timeout: 30000 });
  await page.waitForSelector('.seat-level', { timeout: 30000 });
  await waitForRequests(page, 'azNew', 2);
  await page.waitForTimeout(250);
  await clearRequests(page);
  await page.locator('.seat-level').first().selectOption('1');
  await waitForRequests(page, 'create', 1);
  await waitForRequests(page, 'azNew', 2);
  const watch = await requests(page);
  const watchCreate = watch.find((request) => request.op === 'create');
  const watchNews = watch.filter((request) => request.op === 'azNew');
  if (!watchCreate?.opts?.bots?.includes('sims=1') || !watchCreate.opts.bots.includes('sims=4800'))
    throw new Error(`independent Chess specs missing: ${JSON.stringify(watchCreate)}`);
  const watchExpected = gpu.adapter ? [1, 4800] : [1, 16];
  const watchActual = watchNews.map((request) => request.sims).sort((a, b) => a - b);
  if (JSON.stringify(watchActual) !== JSON.stringify(watchExpected))
    throw new Error(
      `independent Chess runtimes=${watchActual}, expected ${watchExpected}; create=${JSON.stringify(watchCreate)}`,
    );
  if (new Set(watchNews.map((request) => request.worker)).size !== 2)
    throw new Error(`watch seats did not get independent workers: ${JSON.stringify(watchNews)}`);

  // Go uses the same client-seat resolver but a different worker constructor
  // and a game-level board size; verify both arrive together.
  await page.goto(`${base}/#/g/go`, { waitUntil: 'load', timeout: 30000 });
  await page.waitForSelector('.seat-level', { timeout: 30000 });
  await waitForRequests(page, 'goNew', 1);
  await page.waitForTimeout(250);
  await clearRequests(page);
  await page.locator('.seat-level').selectOption('2400');
  await waitForRequests(page, 'create', 1);
  await waitForRequests(page, 'goNew', 1);
  const go = await requests(page);
  const goCreate = go.find((request) => request.op === 'create');
  const goNew = go.find((request) => request.op === 'goNew');
  if (!goCreate?.opts?.bots?.includes('sims=2400'))
    throw new Error(`Go Hard missing from bots spec: ${JSON.stringify(goCreate)}`);
  if (goNew?.sims !== 2400 || goNew?.size !== 19)
    throw new Error(`Go runtime config mismatch: ${JSON.stringify(goNew)}`);

  // Pente: secondary bot knobs must survive conversion from simple `bot=` to
  // per-seat `bots=` when the difficulty selector is changed.
  await page.goto(`${base}/#/g/pente`, { waitUntil: 'load', timeout: 30000 });
  await page.waitForSelector('.seat-level', { timeout: 30000 });
  await waitForRequests(page, 'penteNew', 1);
  await page.waitForTimeout(250);
  await clearRequests(page);
  await page.locator('.gear').evaluate((button) => button.click());
  await page.locator('[name="d-vcf-depth"]').fill('9');
  await page.locator('[name="d-vcf-nodes"]').fill('1234');
  await page.locator('.drawer-apply').click();
  await page.waitForSelector('.seat-level', { timeout: 30000 });
  await waitForRequests(page, 'penteNew', 1);
  await page.waitForTimeout(250);
  await clearRequests(page);
  await page.locator('.seat-level').selectOption('256');
  await waitForRequests(page, 'create', 1);
  await waitForRequests(page, 'penteNew', 1);
  const pente = await requests(page);
  const penteCreate = pente.find((request) => request.op === 'create');
  const penteNew = pente.find((request) => request.op === 'penteNew');
  for (const expected of ['sims=256', 'vcf-depth=9', 'vcf-nodes=1234']) {
    if (!penteCreate?.opts?.bots?.includes(expected))
      throw new Error(`Pente spec lost ${expected}: ${JSON.stringify(penteCreate)}`);
  }
  if (penteCreate.opts.size !== '19' || penteNew?.size !== 19)
    throw new Error(`Pente size desync: ${JSON.stringify({ penteCreate, penteNew })}`);
  if (penteNew.vcfDepth !== 9 || penteNew.vcfNodes !== 1234)
    throw new Error(`Pente runtime lost VCF knobs: ${JSON.stringify(penteNew)}`);

  return {
    chessCreate,
    chessNew,
    watchCreate,
    watchNews,
    goCreate,
    goNew,
    penteCreate,
    penteNew,
  };
}

async function validateCpuSettings(browser, base) {
  const page = await browser.newPage({ viewport: { width: 1100, height: 800 } });
  await page.addInitScript(() => {
    delete Navigator.prototype.gpu;
  });
  await installSettingsProbe(page);
  try {
    await page.goto(`${base}/#/g/chess`, { waitUntil: 'load', timeout: 30000 });
    await page.waitForSelector('.seat-level', { timeout: 30000 });
    await waitForRequests(page, 'azNew', 1);
    const levels = await page.locator('.seat-level option').evaluateAll((options) =>
      options.map((option) => option.value),
    );
    const seen = await requests(page);
    const create = seen.find((request) => request.op === 'create');
    const azNew = seen.find((request) => request.op === 'azNew');
    const note = await page.locator('.cpu-note:not(.gpu-mismatch-note)').textContent();
    if (JSON.stringify(levels) !== JSON.stringify(['1', '16']))
      throw new Error(`CPU difficulty ladder=${JSON.stringify(levels)}, expected ["1","16"]`);
    if (create?.opts?.sims !== '1' || azNew?.sims !== 1)
      throw new Error(`CPU default was not normalized to one sim: ${JSON.stringify({ create, azNew })}`);
    if (!/CPU FALLBACK ACTIVE/.test(note ?? '') || !/1 sim/.test(note ?? ''))
      throw new Error(`CPU fallback note does not report the runtime budget: ${JSON.stringify(note)}`);
    await page.goto(`${base}/#/g/pente`, { waitUntil: 'load', timeout: 30000 });
    await page.waitForSelector('.seat-level', { timeout: 30000 });
    await waitForRequests(page, 'penteNew', 1);
    await page.waitForTimeout(250);
    const penteSeen = await requests(page);
    const penteNew = penteSeen.find((request) => request.op === 'penteNew');
    const penteNote = await page
      .locator('.cpu-note:not(.gpu-mismatch-note)')
      .textContent();
    if (
      penteNew?.sims !== 1 ||
      penteNew?.size !== 19 ||
      penteNew?.vcfDepth !== 7 ||
      penteNew?.vcfNodes !== 1500
    ) throw new Error(`Pente CPU defaults mismatch: ${JSON.stringify(penteNew)}`);
    if (!/CPU FALLBACK ACTIVE/.test(penteNote ?? '') || !/1 sim/.test(penteNote ?? ''))
      throw new Error(`Pente CPU fallback is not surfaced: ${JSON.stringify(penteNote)}`);
    return { levels, create, azNew, note, penteNew, penteNote };
  } finally {
    await page.close();
  }
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

    if (game === 'settings') await installSettingsProbe(page);

    await page.goto(url, { waitUntil: 'load', timeout: 30000 });
    result.gpu = await pageGpuInfo(page);

    if (game === 'settings') {
      result.settings = await validateSettings(page, base, result.gpu);
      result.cpuSettings = await validateCpuSettings(browser, base);
      result.ok = true;
    } else {
      // Wait for the game to mount, then let it reach a representative frame.
      await page.waitForSelector(cfg.ready, { timeout: 30000 }).catch(() => {});

      const shot = join(OUT_DIR, `${game}.png`);
      await rm(shot, { force: true });
      const overlay = cfg.overlaySelector ? page.locator(cfg.overlaySelector) : null;
      for (let i = 0; i < 10; i++) {
        await page.waitForTimeout(800);
        if (i === 4) {
          await page.screenshot({ path: shot });
        }
      }
      if (overlay) {
        result.overlayVisible = await overlay.isVisible().catch(() => false);
        result.overlayText = (await overlay.textContent().catch(() => '')) ?? '';
      }
      if (!existsSync(shot)) await page.screenshot({ path: shot });
      result.screenshot = shot;
      if (game === 'snake') result.interactions = await validateSnakeInteractions(page, base);
      result.ok = true;
    }
  } catch (e) {
    result.error = String(e);
  } finally {
    result.consoleLogs = consoleLogs;
    await browser.close().catch(() => {});
    server.close();
  }

  await writeFile(join(OUT_DIR, `${game}.json`), JSON.stringify(result, null, 2));

  console.log('\n===== VALIDATION REPORT =====');
  console.log(`game           ${result.game}`);
  console.log(`page navigator.gpu ${JSON.stringify(result.gpu)}`);
  if (cfg.overlaySelector) {
    console.log(`overlay visible ${result.overlayVisible}`);
    console.log(`overlay text   ${JSON.stringify(result.overlayText)}`);
  }
  console.log(`screenshot     ${result.screenshot}`);
  if (result.interactions) console.log(`interactions   ${JSON.stringify(result.interactions)}`);
  if (result.settings) console.log(`settings       ${JSON.stringify(result.settings)}`);
  if (result.cpuSettings) console.log(`cpu settings   ${JSON.stringify(result.cpuSettings)}`);
  console.log('console logs:');
  for (const l of consoleLogs) console.log('  ' + l);
  if (result.error) console.log(`ERROR          ${result.error}`);
  console.log('=============================');
  process.exit(result.ok ? 0 : 1);
}

main();
