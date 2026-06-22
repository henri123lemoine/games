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

    // Observe a continuous session. Track per poll: the board size (a reset
    // 5→0→grow marks a new hand), the dealer button seat (must rotate), and the
    // first seat's stack (must carry/change across hands, not reset each hand).
    const seen = {
      seats: 0,
      maxBoard: 0,
      potValues: new Set(),
      sawBanner: false,
      sawBetChip: false,
      hands: 0, // distinct dealt hands observed
      buttons: new Set(), // distinct dealer-button seats seen
      seat0Stacks: new Set(), // distinct seat-0 stack readings
      sawGameOver: false, // did the shell ever show a 'game over'/rematch?
    };
    let prevBoard = -1;
    let everSawFullBoard = false;
    for (let i = 0; i < 90; i++) {
      await page.waitForTimeout(500);
      const snap = await page.evaluate(() => {
        const board = document.querySelectorAll('.pk-board .pk-card').length;
        const seats = document.querySelectorAll('.pk-seat').length;
        const potEl = document.querySelector('.pk-pot');
        const pot = potEl ? potEl.textContent : '';
        const banner = document.querySelector('.pk-banner.show');
        const bet = document.querySelectorAll('.pk-bet').length;
        // Dealer button seat: the .pk-seat whose pod contains the .pk-dealer chip.
        let button = -1;
        document.querySelectorAll('.pk-seat').forEach((el) => {
          if (el.querySelector('.pk-dealer')) button = Number(el.getAttribute('data-seat'));
        });
        // Seat 0's stack text (e.g. "198 bb").
        const s0 = document.querySelector('.pk-seat[data-seat="0"] .pk-stack');
        const stack0 = s0 ? s0.textContent.trim() : '';
        // Any shell-level "game over" / rematch UI (should NOT appear mid-session).
        const statusEl = document.querySelector('.status, .result, .game-over');
        const status = statusEl ? statusEl.textContent : '';
        return { board, seats, pot, banner: banner?.textContent ?? null, bet, button, stack0, status };
      });
      seen.seats = Math.max(seen.seats, snap.seats);
      seen.maxBoard = Math.max(seen.maxBoard, snap.board);
      if (snap.board === 5) everSawFullBoard = true;
      // A new hand: the board dropped back toward empty after having had cards.
      if (prevBoard >= 3 && snap.board <= 1) seen.hands++;
      prevBoard = snap.board;
      if (snap.pot) seen.potValues.add(snap.pot);
      if (snap.bet > 0) seen.sawBetChip = true;
      if (snap.button >= 0) seen.buttons.add(snap.button);
      if (snap.stack0) seen.seat0Stacks.add(snap.stack0);
      if (/game over|rematch|wins\.$/i.test(snap.status || '')) seen.sawGameOver = true;
      if (snap.banner) {
        seen.sawBanner = true;
        if (/win|lose|takes|showdown|break even/i.test(snap.banner)) {
          await page.screenshot({ path: join(OUT_DIR, 'poker-showdown.png') });
        }
      }
      if (snap.board >= 3 && !existsSync(join(OUT_DIR, 'poker-board.png'))) {
        await page.screenshot({ path: join(OUT_DIR, 'poker-board.png') });
      }
    }
    await page.screenshot({ path: join(OUT_DIR, 'poker.png') });
    // +1 because the first hand never had a "reset" edge to count it.
    const handsPlayed = seen.hands + 1;

    result.observed = {
      seats: seen.seats,
      handsPlayed,
      everSawFullBoard,
      maxBoardCards: seen.maxBoard,
      distinctButtonSeats: seen.buttons.size,
      buttonSeats: [...seen.buttons].sort((a, b) => a - b),
      distinctSeat0Stacks: seen.seat0Stacks.size,
      distinctPots: seen.potValues.size,
      sawBetChip: seen.sawBetChip,
      sawBanner: seen.sawBanner,
      sawGameOver: seen.sawGameOver,
    };
    // Acceptance (continuous session): 6 seats; SEVERAL hands played in a row;
    // the board reached the river at least once; the dealer button ROTATED
    // (>=2 distinct seats); seat 0's stack CARRIED/changed across hands (>=2
    // distinct readings, i.e. not reset to a fresh stack each hand); bets and a
    // result banner fired; and the shell NEVER showed a 'game over' mid-session.
    // with no console errors.
    result.ok =
      seen.seats === 6 &&
      handsPlayed >= 3 &&
      everSawFullBoard &&
      seen.buttons.size >= 2 &&
      seen.seat0Stacks.size >= 2 &&
      seen.potValues.size >= 2 &&
      seen.sawBetChip &&
      seen.sawBanner &&
      !seen.sawGameOver &&
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
