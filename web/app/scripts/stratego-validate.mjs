// Headless real-browser validation for Stratego: confirms the published
// opponent is ataraxios (the trained net, the only bot the site lists), that
// the 55 MB ATRX1 artifact is fetched and loads in the wasm engine, and that
// the net answers a human move.
//
//   npx vite build && node scripts/stratego-validate.mjs          # headless
//   node scripts/stratego-validate.mjs --headed                   # show it

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
  '.bin': 'application/octet-stream',
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
  if (!existsSync(DIST)) {
    console.error(`no build at ${DIST} — run \`npx vite build\` first.`);
    process.exit(2);
  }
  await mkdir(OUT_DIR, { recursive: true });

  const { server, port } = await startServer(DIST);
  const base = `http://localhost:${port}`;
  console.log(`[stratego-validate] serving dist/ at ${base}`);

  const browser = await chromium.launch({
    headless: false,
    args: headed ? [] : ['--headless=new'],
  });

  const result = { game: 'stratego', ok: false };
  try {
    const errors = [];
    let artifactStatus = null;
    const page = await browser.newPage({ viewport: { width: 1200, height: 860 } });
    page.on('pageerror', (e) => errors.push(String(e.message)));
    page.on('console', (m) => {
      if (m.type() === 'error') errors.push(m.text());
    });
    page.on('response', (r) => {
      if (r.url().includes('artifacts/ataraxios.bin')) artifactStatus = r.status();
    });

    await page.goto(base + '/#/g/stratego', { waitUntil: 'load', timeout: 30000 });

    // The human is seat 0 (red, moves first) on a random pre-deployed board,
    // so the generic frontend shows the board and the legal-move buttons once
    // the engine and artifact are up.
    await page.waitForSelector('.generic-view', { timeout: 60000 });
    await page.waitForSelector('.action-btn', { timeout: 120000 });

    const rosterLabels = await page.evaluate(() =>
      [...document.querySelectorAll('.seat-select')].map(
        (sel) => sel.selectedOptions[0]?.textContent ?? null,
      ),
    );
    const viewBefore = await page.evaluate(
      () => document.querySelector('.generic-view').textContent,
    );
    await page.screenshot({ path: join(OUT_DIR, 'stratego-before.png') });

    // Play the first legal move, then wait for ataraxios to answer: the turn
    // comes back to the human (buttons reappear) with a changed board.
    const t0 = Date.now();
    await page.click('.action-btn');
    await page.waitForFunction(
      (before) => {
        const view = document.querySelector('.generic-view')?.textContent ?? '';
        const buttons = document.querySelectorAll('.action-btn').length;
        return view !== before && buttons > 0 && view.includes('Player 0 to move');
      },
      viewBefore,
      { timeout: 180000, polling: 500 },
    );
    const botReplyMs = Date.now() - t0;

    const viewAfter = await page.evaluate(
      () => document.querySelector('.generic-view').textContent,
    );
    await page.screenshot({ path: join(OUT_DIR, 'stratego-after.png') });
    await page.close();

    const botLabels = rosterLabels.filter((l) => l !== 'You');
    const opponentIsAtaraxios =
      botLabels.length === 1 && botLabels[0] === 'Ataraxios';
    result.rosterLabels = rosterLabels;
    result.artifactStatus = artifactStatus;
    result.botReplyMs = botReplyMs;
    result.boardChanged = viewAfter !== viewBefore;
    result.consoleErrors = errors;
    result.ok =
      opponentIsAtaraxios &&
      artifactStatus === 200 &&
      result.boardChanged &&
      errors.length === 0;
  } catch (e) {
    result.error = String(e);
  } finally {
    await browser.close().catch(() => {});
    server.close();
  }

  await writeFile(join(OUT_DIR, 'stratego.json'), JSON.stringify(result, null, 2));
  console.log('\n===== STRATEGO VALIDATION REPORT =====');
  console.log('roster labels:', result.rosterLabels);
  console.log('artifact fetch status:', result.artifactStatus);
  console.log('bot reply after human move:', result.botReplyMs, 'ms');
  console.log('console errors:', result.consoleErrors);
  if (result.error) console.log(`ERROR ${result.error}`);
  console.log('screenshots', `${OUT_DIR}/stratego-before.png, ${OUT_DIR}/stratego-after.png`);
  process.exit(result.ok ? 0 : 1);
}

main();
