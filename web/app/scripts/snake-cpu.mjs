// Headless CPU-ONLY validation: the bot must play competently with NO WebGPU
// and a throttled CPU — never go straight into a wall and die, survive a
// meaningful number of moves, and navigate/eat. This is the gate for "the bot
// can't depend on heavy GPU/MCTS search; the fast CPU policy floor carries it."
//
// Chromium is launched WITHOUT the WebGPU flags, so navigator.gpu is absent and
// the bot takes the CPU forward path (isCpuFallback() === true). The CPU is then
// throttled 6x via CDP. We keep the PLAYER snake alive by steering toward
// center, and watch the BOT snake (seat 1): its survival length, its growth
// (food eaten), and whether it ever marches straight into a wall on the opening.
//
//   node scripts/snake-cpu.mjs

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

// Disable WebGPU at the browser level AND below we strip navigator.gpu via an
// init script, so the bot is forced onto the CPU forward path
// (isCpuFallback() === true). --headless=new keeps the full Chromium windowless.
const CPU_FLAGS = ['--headless=new', '--disable-features=WebGPU,Vulkan'];

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

const OPP = { n: 's', s: 'n', e: 'w', w: 'e' };
const KEY_OF = { n: 'ArrowUp', e: 'ArrowRight', s: 'ArrowDown', w: 'ArrowLeft' };
const PERP = { n: ['e', 'w'], s: ['e', 'w'], e: ['n', 's'], w: ['n', 's'] };

const wall = (x, y) => x <= 0 || y <= 0 || x >= 19 || y >= 19;

async function main() {
  if (!existsSync(DIST)) {
    console.error(`no build at ${DIST} — run \`vite build\` first.`);
    process.exit(2);
  }
  await mkdir(OUT_DIR, { recursive: true });
  const { server, port } = await startServer(DIST);
  const base = `http://localhost:${port}`;

  const browser = await chromium.launch({ headless: false, args: CPU_FLAGS });
  const consoleLogs = [];
  const result = { ok: false };
  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 800 } });
    // Belt-and-suspenders: make `'gpu' in navigator` FALSE before any page
    // script runs (isCpuFallback() checks exactly that), even if the binary
    // still exposes WebGPU. Delete the own prop and shadow the prototype getter.
    await page.addInitScript(() => {
      try {
        const proto = Object.getPrototypeOf(navigator);
        delete navigator.gpu;
        if (proto && 'gpu' in proto) delete proto.gpu;
      } catch {
        /* ignore */
      }
    });
    page.on('console', (m) => consoleLogs.push(`[${m.type()}] ${m.text()}`));
    page.on('pageerror', (e) => consoleLogs.push(`[pageerror] ${e.message}`));

    // Throttle the CPU hard so the heavy search can NEVER keep up — the policy
    // floor must carry the bot.
    const client = await page.context().newCDPSession(page);
    await client.send('Emulation.setCPUThrottlingRate', { rate: 6 });

    await page.goto(`${base}/?snakeDebug#/g/snake`, { waitUntil: 'load', timeout: 40000 });
    await page.waitForSelector('.snk-canvas', { timeout: 40000 });
    await page.locator('.snk-stage').click({ position: { x: 5, y: 5 } }).catch(() => {});

    // Confirm we are actually on the no-GPU path.
    result.gpuAbsent = await page.evaluate(() => !('gpu' in navigator));

    await page.waitForFunction(() => !!window.__snakeHead1, { timeout: 15000 }).catch(() => {});

    // Track the BOT (seat 1): its head trail, max length (food eaten), and
    // whether it ever sat on a wall cell (about to die into it). Keep the PLAYER
    // alive by steering toward center so the game lasts.
    const botTrail = [];
    let botMaxLen = 0;
    let botEverOnWall = false;
    let botStraightWallDeath = false;
    let ticks = 0;
    let playerDir = 'e';
    const t0 = Date.now();
    while (Date.now() - t0 < 20000) {
      const over = await page.evaluate(() => !!document.querySelector('.snk-overlay.snk-show'));
      if (over) break;
      const h0 = await page.evaluate(() => window.__snakeHead0 ?? null);
      const h1 = await page.evaluate(() => window.__snakeHead1 ?? null);
      if (h1) {
        const cell = { x: Math.round(h1.x), y: Math.round(h1.y) };
        const lastCell = botTrail[botTrail.length - 1];
        if (!lastCell || lastCell.x !== cell.x || lastCell.y !== cell.y) {
          botTrail.push(cell);
          ticks++;
        }
        botMaxLen = Math.max(botMaxLen, h1.len ?? 0);
        if (wall(cell.x, cell.y)) botEverOnWall = true;
      }
      // Steer the player toward center to survive (perpendicular turn inward).
      if (h0) {
        const [a, b] = PERP[playerDir];
        const want =
          (a === 'n' && h0.y > 10) || (a === 's' && h0.y < 10) ||
          (a === 'e' && h0.x < 10) || (a === 'w' && h0.x > 10) ? a : b;
        if (want !== OPP[playerDir] && want !== playerDir) {
          await page.keyboard.press(KEY_OF[want]);
          playerDir = want;
        }
      }
      await page.waitForTimeout(150);
    }

    // Did the bot die by marching straight into a wall on the opening? Check the
    // last few cells of the trail: a straight run into a border cell with no turn.
    const tail = botTrail.slice(-4);
    if (tail.length >= 3) {
      const allSameAxis =
        tail.every((c, i) => i === 0 || c.x === tail[0].x) ||
        tail.every((c, i) => i === 0 || c.y === tail[0].y);
      const endedOnWall = wall(tail[tail.length - 1].x, tail[tail.length - 1].y);
      botStraightWallDeath = allSameAxis && endedOnWall && ticks <= 6;
    }

    const finalLen1 = (await page.evaluate(() => window.__snakeHead1?.len ?? 0)) || botMaxLen;
    result.bot = {
      survivedCells: ticks,
      maxLen: botMaxLen,
      finalLen: finalLen1,
      foodEaten: Math.max(0, botMaxLen - 3),
      everWentStraightIntoWallDeath: botStraightWallDeath,
      distinctCells: new Set(botTrail.map((c) => `${c.x},${c.y}`)).size,
    };
    result.endedOver = await page.evaluate(() => !!document.querySelector('.snk-overlay.snk-show'));
    await page.screenshot({ path: join(OUT_DIR, 'snake-cpu.png') });
    result.ok = true;
  } catch (e) {
    result.error = String(e);
  } finally {
    result.consoleErrors = consoleLogs.filter((l) => /pageerror|\[error\]/.test(l)).slice(0, 8);
    result.backendLog = consoleLogs.find((l) => /\[snake\]/.test(l)) ?? null;
    await browser.close().catch(() => {});
    server.close();
  }

  await writeFile(join(OUT_DIR, 'snake-cpu.json'), JSON.stringify(result, null, 2));

  const b = result.bot ?? {};
  // Gate: bot survived a meaningful number of moves, never marched straight into
  // a wall death, and visibly navigated (visited many distinct cells).
  const pass =
    result.gpuAbsent === true &&
    b.everWentStraightIntoWallDeath === false &&
    (b.survivedCells ?? 0) >= 10 &&
    (b.distinctCells ?? 0) >= 8;

  console.log('\n===== SNAKE CPU-ONLY VALIDATION =====');
  console.log('navigator.gpu absent (CPU path):', result.gpuAbsent);
  console.log('backend log:', result.backendLog);
  console.log('bot survived (cells):', b.survivedCells);
  console.log('bot max length / food eaten:', b.maxLen, '/', b.foodEaten);
  console.log('bot distinct cells visited:', b.distinctCells);
  console.log('bot ever went straight into a wall death:', b.everWentStraightIntoWallDeath);
  console.log('PASS:', pass);
  if (result.consoleErrors?.length) console.log('errors:', result.consoleErrors);
  if (result.error) console.log('ERROR', result.error);
  console.log('=====================================');
  process.exit(pass ? 0 : 1);
}

main();
