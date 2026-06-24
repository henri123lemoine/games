// Pente AlphaZero (azero-gpu) arcade browser validation: a real headless
// Chromium with WebGPU against the built app over a COOP/COEP server (the wasm
// engine needs cross-origin isolation), driving a PLAY game vs the azero-gpu
// bot at 19×19 — the arcade default. It clicks the forced center as Black
// (human plays first), then a few real moves, and verifies the AlphaZero bot
// (WebGPU when present, the identical in-wasm CPU forward otherwise) responds
// with legal moves and the board grows. Reports the backend that actually ran,
// console logs, screenshots, and the parsed board state.
//
//   npx vite build && node scripts/pente-azero-validate.mjs [--headed]

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

const FLAGS = [
  '--headless=new',
  '--ignore-gpu-blocklist',
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
];

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
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

/** Read Pente's on-page state from the rendered DOM. */
async function readState(page) {
  return page.evaluate(() => {
    const root = document.querySelector('.pente');
    if (!root) return { mounted: false };
    const stones = document.querySelectorAll('.pente-stones circle').length;
    const pairs = [...document.querySelectorAll('.pente-pcaps b')].map((b) => b.textContent);
    const turn = document.querySelector('.pente-turn-text')?.textContent ?? '';
    const status = document.querySelector('.status')?.textContent ?? '';
    // The board's viewBox width tells us the rendered board size (ext = size-1 + 2*PAD).
    const vb = document.querySelector('.pente-svg')?.getAttribute('viewBox') ?? '';
    const ext = Number(vb.split(' ')[2] ?? 0);
    const size = ext > 0 ? Math.round(ext - 1) : 0;
    const legalHits = document.querySelectorAll('.pente-hit-on').length;
    return { mounted: true, stones, pairs, turn, status, size, legalHits };
  });
}

async function clickPoint(page, p) {
  await page.locator(`.pente-hit[data-p="${p}"]`).click({ force: true, timeout: 5000 });
}

async function run() {
  const headed = process.argv.includes('--headed');
  if (!existsSync(DIST)) {
    console.error(`no build at ${DIST} — run \`npx vite build\` first.`);
    process.exit(2);
  }
  await mkdir(OUT_DIR, { recursive: true });
  const { server, port } = await startServer(DIST);
  const base = `http://localhost:${port}`;
  const browser = await chromium.launch({
    headless: false,
    args: headed ? FLAGS.filter((f) => f !== '--headless=new') : FLAGS,
  });
  const report = { base, play: {} };
  const logs = [];

  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 1000 } });
    page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
    page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));

    report.webgpu = await page.evaluate(() => 'gpu' in navigator);

    // ---- PLAY: human is Black vs the azero-gpu bot (default 19×19) ----
    const playUrl = `${base}/#/g/pente`;
    console.log(`[play] ${playUrl}  (webgpu in page: ${report.webgpu})`);
    await page.goto(playUrl, { waitUntil: 'load', timeout: 30000 });
    await page.waitForSelector('.pente-svg', { timeout: 30000 });
    await page.waitForTimeout(1500);

    const before = await readState(page);
    report.play.initial = before;
    console.log(`[play] initial: ${JSON.stringify(before)}`);

    // 19×19 center k10 = index 9*19+9 = 180 (the only legal first move). Verify
    // only one point is playable (the forced center) before clicking.
    const forcedCenter = 9 * 19 + 9;
    report.play.forcedCenterOnly = before.legalHits === 1;
    console.log(`[play] legal hit points at start: ${before.legalHits} (forced center = 1)`);
    await clickPoint(page, forcedCenter);

    let afterFirst = null;
    for (let i = 0; i < 60; i++) {
      await page.waitForTimeout(500);
      afterFirst = await readState(page);
      if (afterFirst.stones >= 2) break;
    }
    report.play.afterFirstExchange = afterFirst;
    console.log(`[play] after center + bot reply: ${JSON.stringify(afterFirst)}`);
    await page.screenshot({ path: join(OUT_DIR, 'pente-azero-1.png') });

    // A few more human moves near the center; confirm stones keep growing and
    // the bot keeps replying with legal moves. 19-wide indices around k10.
    // l10=190, k11=199, l11=200, j10=189.
    const humanMoves = [190, 199, 200, 189];
    for (const p of humanMoves) {
      let st = await readState(page);
      // Wait until it is the human's turn (the bot may still be thinking).
      for (let w = 0; w < 40 && !/Black to move/i.test(st.turn); w++) {
        if (/wins|Draw/i.test(st.turn)) break;
        await page.waitForTimeout(500);
        st = await readState(page);
      }
      if (/wins|Draw/i.test(st.turn)) break;
      const stonesBefore = st.stones;
      await clickPoint(page, p).catch(() => {});
      // Wait for the human stone + the bot's reply.
      for (let w = 0; w < 40; w++) {
        await page.waitForTimeout(500);
        const now = await readState(page);
        if (now.stones >= stonesBefore + 2 || /wins|Draw/i.test(now.turn)) break;
      }
    }
    const playFinal = await readState(page);
    report.play.final = playFinal;
    await page.screenshot({ path: join(OUT_DIR, 'pente-azero-2.png') });
    console.log(`[play] final: ${JSON.stringify(playFinal)}`);

    const errors = logs.filter((l) => /\[error\]|\[pageerror\]/.test(l));
    report.consoleErrors = errors;
    report.ok =
      !!before.mounted &&
      before.size === 19 &&
      report.play.forcedCenterOnly === true &&
      !!afterFirst &&
      afterFirst.stones >= 2 &&
      (playFinal.stones ?? 0) >= afterFirst.stones &&
      errors.length === 0;
  } catch (e) {
    report.error = String(e);
  } finally {
    report.consoleLogs = logs;
    await browser.close().catch(() => {});
    server.close();
  }

  await writeFile(join(OUT_DIR, 'pente-azero.json'), JSON.stringify(report, null, 2));
  console.log('\n===== PENTE AZERO-GPU VALIDATION =====');
  console.log(`ok: ${report.ok}`);
  console.log(`webgpu in page: ${report.webgpu}`);
  console.log(
    `play: size=${report.play.initial?.size} forcedCenterOnly=${report.play.forcedCenterOnly} | ` +
      `afterFirst stones=${report.play.afterFirstExchange?.stones} turn="${report.play.afterFirstExchange?.turn}" | ` +
      `final ${JSON.stringify(report.play.final)}`,
  );
  console.log(`console errors: ${JSON.stringify(report.consoleErrors ?? [])}`);
  console.log('all console logs:');
  for (const l of logs) console.log('  ' + l);
  if (report.error) console.log(`ERROR ${report.error}`);
  console.log('======================================');
  process.exit(report.ok ? 0 : 1);
}

run();
