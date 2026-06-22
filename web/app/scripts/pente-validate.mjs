// Pente arcade browser validation: a real headless Chromium against the built
// app over a COOP/COEP server (the wasm engine needs cross-origin isolation),
// driving BOTH a watch game (bot vs bot — exercises render, captures, the
// capture-pair counters, and the win display) and a play game (click the
// forced center, then real moves, verify the in-engine alpha-beta bot
// responds). Reports console logs, screenshots, and the parsed board state, so
// the frontend is validated without a human reloading localhost.
//
//   npx vite build && node scripts/pente-validate.mjs [--headed]

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

const FLAGS = ['--headless=new', '--ignore-gpu-blocklist'];

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
    const winLineVisible =
      document.querySelector('.pente-winline')?.getAttribute('opacity') === '1';
    const winStones = document.querySelectorAll('.pente-win-stone').length;
    const litPips = document.querySelectorAll('.pente-pip-on').length;
    return { mounted: true, stones, pairs, turn, winLineVisible, winStones, litPips };
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
  const report = { base, watch: {}, play: {} };
  const logs = [];

  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 900 } });
    page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
    page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));

    // ---- WATCH: bot vs bot (default opts: size 13, depth 4) ----
    const watchUrl = `${base}/#/g/pente?mode=watch`;
    console.log(`[watch] ${watchUrl}`);
    await page.goto(watchUrl, { waitUntil: 'load', timeout: 30000 });
    await page.waitForSelector('.pente-svg', { timeout: 30000 });
    let watchState = null;
    let sawCapture = false;
    for (let i = 0; i < 80; i++) {
      await page.waitForTimeout(500);
      watchState = await readState(page);
      const pairs = (watchState.pairs ?? []).map((x) => Number(x));
      if (pairs.some((n) => n > 0)) sawCapture = true;
      if (/wins|Draw/i.test(watchState.turn)) break;
    }
    await page.screenshot({ path: join(OUT_DIR, 'pente-watch.png') });
    report.watch = { ...watchState, sawCapture, screenshot: join(OUT_DIR, 'pente-watch.png') };
    console.log(`[watch] final: ${JSON.stringify(report.watch)}`);

    // ---- PLAY: human is Black; click the forced center, then real moves ----
    const playUrl = `${base}/#/g/pente`;
    console.log(`[play] ${playUrl}`);
    await page.goto(playUrl, { waitUntil: 'load', timeout: 30000 });
    await page.waitForSelector('.pente-svg', { timeout: 30000 });
    await page.waitForTimeout(900);

    const before = await readState(page);
    report.play.initial = before;
    console.log(`[play] initial: ${JSON.stringify(before)}`);

    // 13x13 center g7 = index 6*13+6 = 84 (the only legal first move).
    await clickPoint(page, 84);
    let afterFirst = null;
    for (let i = 0; i < 25; i++) {
      await page.waitForTimeout(400);
      afterFirst = await readState(page);
      if (afterFirst.stones >= 2) break;
    }
    report.play.afterFirstExchange = afterFirst;
    console.log(`[play] after center + bot reply: ${JSON.stringify(afterFirst)}`);
    await page.screenshot({ path: join(OUT_DIR, 'pente-play-1.png') });

    // A few more human moves near the center; confirm stones keep growing and
    // nothing throws. h7=85, g6=71, h8=98, f6=70 on a 13-wide board.
    for (const p of [85, 71, 98, 70]) {
      const st = await readState(page);
      if (/wins|Draw/i.test(st.turn)) break;
      if (!/Black to move/i.test(st.turn)) {
        await page.waitForTimeout(500);
        continue;
      }
      await clickPoint(page, p).catch(() => {});
      await page.waitForTimeout(1000);
    }
    const playFinal = await readState(page);
    report.play.final = playFinal;
    await page.screenshot({ path: join(OUT_DIR, 'pente-play-2.png') });
    console.log(`[play] final: ${JSON.stringify(playFinal)}`);

    const errors = logs.filter((l) => /\[error\]|\[pageerror\]/.test(l));
    report.consoleErrors = errors;
    report.ok =
      !!before.mounted &&
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

  await writeFile(join(OUT_DIR, 'pente.json'), JSON.stringify(report, null, 2));
  console.log('\n===== PENTE VALIDATION =====');
  console.log(`ok: ${report.ok}`);
  console.log(
    `watch: turn="${report.watch.turn}" stones=${report.watch.stones} pairs=${JSON.stringify(report.watch.pairs)} sawCapture=${report.watch.sawCapture} winLine=${report.watch.winLineVisible} winStones=${report.watch.winStones}`,
  );
  console.log(
    `play:  afterFirst stones=${report.play.afterFirstExchange?.stones} turn="${report.play.afterFirstExchange?.turn}" | final ${JSON.stringify(report.play.final)}`,
  );
  console.log(`console errors: ${JSON.stringify(report.consoleErrors ?? [])}`);
  console.log('all console logs:');
  for (const l of logs) console.log('  ' + l);
  if (report.error) console.log(`ERROR ${report.error}`);
  console.log('============================');
  process.exit(report.ok ? 0 : 1);
}

run();
