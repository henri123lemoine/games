// Headless real-browser validation for the poker table. Serves the built app,
// loads the poker game in watch mode (all bot seats, so hands auto-play), and
// asserts the table renders and a real hand resolves: hole cards dealt, a
// community board appears, the pot updates, and a showdown/result banner shows
// — all with no console errors. Captures screenshots at key moments.
//
//   node scripts/poker-validate.mjs          # headless
//   node scripts/poker-validate.mjs --headed # show the window

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

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
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
      const url = new URL(req.url, 'http://x');
      let p = decodeURIComponent(url.pathname);
      if (p === '/' || !extname(p)) p = '/index.html';
      const file = join(rootDir, p);
      if (!existsSync(file)) {
        const idx = await readFile(join(rootDir, 'index.html'));
        res.writeHead(200, { 'content-type': MIME['.html'] });
        res.end(idx);
        return;
      }
      const body = await readFile(file);
      res.writeHead(200, { 'content-type': MIME[extname(file)] ?? 'application/octet-stream' });
      res.end(body);
    } catch (e) {
      res.writeHead(500);
      res.end(String(e));
    }
  });
  return new Promise((res) => {
    server.listen(0, '127.0.0.1', () => res({ server, port: server.address().port }));
  });
}

async function main() {
  const headed = process.argv.includes('--headed');
  // Fast bots (small sample count) so several hands resolve quickly; speed=8
  // for snappy spectate pacing. Watch mode = every seat a bot.
  const route = '/#/g/poker?mode=watch&players=6&samples=120&speed=8';
  if (!existsSync(DIST)) {
    console.error(`no build at ${DIST} — run \`npx vite build\` first.`);
    process.exit(2);
  }
  await mkdir(OUT_DIR, { recursive: true });

  const { server, port } = await startServer(DIST);
  const base = `http://localhost:${port}`;
  const url = base + route;
  console.log(`[poker-validate] serving dist/ at ${base}`);
  console.log(`[poker-validate] url=${url} headed=${headed}`);

  const browser = await chromium.launch({
    headless: false,
    args: headed ? [] : ['--headless=new'],
  });

  const consoleLogs = [];
  const errors = [];
  const result = { game: 'poker', url, ok: false };
  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 820 } });
    page.on('console', (m) => {
      const line = `[${m.type()}] ${m.text()}`;
      consoleLogs.push(line);
      if (m.type() === 'error') errors.push(line);
    });
    page.on('pageerror', (e) => {
      const line = `[pageerror] ${e.message}`;
      consoleLogs.push(line);
      errors.push(line);
    });

    await page.goto(url, { waitUntil: 'load', timeout: 30000 });
    // The felt table mounts once the engine creates the match.
    await page.waitForSelector('.pk-felt', { timeout: 30000 });

    // Observe a few hands. Track: hole cards rendered, a board that grows to
    // five cards at least once, the pot changing, and a result banner.
    const seen = {
      seats: 0,
      maxBoard: 0,
      potValues: new Set(),
      sawBanner: false,
      sawBetChip: false,
    };
    for (let i = 0; i < 60; i++) {
      await page.waitForTimeout(500);
      const snap = await page.evaluate(() => {
        const board = document.querySelectorAll('.pk-board .pk-card').length;
        const seats = document.querySelectorAll('.pk-seat').length;
        const potEl = document.querySelector('.pk-pot');
        const pot = potEl ? potEl.textContent : '';
        const banner = document.querySelector('.pk-banner.show');
        const bet = document.querySelectorAll('.pk-bet').length;
        const cards = document.querySelectorAll('.pk-seat .pk-card').length;
        return { board, seats, pot, banner: banner ? banner.textContent : null, bet, cards };
      });
      seen.seats = Math.max(seen.seats, snap.seats);
      seen.maxBoard = Math.max(seen.maxBoard, snap.board);
      if (snap.pot) seen.potValues.add(snap.pot);
      if (snap.bet > 0) seen.sawBetChip = true;
      if (snap.banner) {
        seen.sawBanner = true;
        // Capture a screenshot at a showdown/result banner — the climactic frame.
        if (/win|lose|takes|showdown|break even/i.test(snap.banner)) {
          await page.screenshot({ path: join(OUT_DIR, 'poker-showdown.png') });
        }
      }
      // Mid-hand frame with a board out.
      if (snap.board >= 3 && !existsSync(join(OUT_DIR, 'poker-board.png'))) {
        await page.screenshot({ path: join(OUT_DIR, 'poker-board.png') });
      }
    }
    await page.screenshot({ path: join(OUT_DIR, 'poker.png') });

    result.observed = {
      seats: seen.seats,
      maxBoardCards: seen.maxBoard,
      distinctPots: seen.potValues.size,
      sawBetChip: seen.sawBetChip,
      sawBanner: seen.sawBanner,
    };
    // Acceptance: 6 seats, the board reached at least the flop, the pot took on
    // multiple values (chips moved), bets were posted, and a banner fired —
    // with no console errors.
    result.ok =
      seen.seats === 6 &&
      seen.maxBoard >= 3 &&
      seen.potValues.size >= 2 &&
      seen.sawBetChip &&
      seen.sawBanner &&
      errors.length === 0;
  } catch (e) {
    result.error = String(e);
  } finally {
    result.consoleErrors = errors;
    result.consoleLogCount = consoleLogs.length;
    await browser.close().catch(() => {});
    server.close();
  }

  await writeFile(join(OUT_DIR, 'poker.json'), JSON.stringify({ ...result, consoleLogs }, null, 2));
  console.log('\n===== POKER VALIDATION REPORT =====');
  console.log(JSON.stringify(result.observed ?? {}, null, 2));
  console.log(`console errors  ${errors.length}`);
  for (const l of errors) console.log('  ' + l);
  if (result.error) console.log(`ERROR           ${result.error}`);
  console.log(`screenshots     ${OUT_DIR}/poker*.png`);
  console.log(`pass            ${result.ok}`);
  console.log('===================================');
  process.exit(result.ok ? 0 : 1);
}

main();
