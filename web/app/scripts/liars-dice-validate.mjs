// Headless real-browser validation for Liar's Dice: confirms the neural
// ("history") bot is the default opponent, and that consecutive bot bids are
// paced slowly enough for a human to follow.
//
//   node scripts/liars-dice-validate.mjs          # headless
//   node scripts/liars-dice-validate.mjs --headed # show the window

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
  if (!existsSync(DIST)) {
    console.error(`no build at ${DIST} — run \`npx vite build\` first.`);
    process.exit(2);
  }
  await mkdir(OUT_DIR, { recursive: true });

  const { server, port } = await startServer(DIST);
  const base = `http://localhost:${port}`;
  console.log(`[ld-validate] serving dist/ at ${base}`);

  const browser = await chromium.launch({
    headless: false,
    args: headed ? [] : ['--headless=new'],
  });

  const result = { game: 'liars-dice', ok: false };
  try {
    // ---------- 1. default bot is Neural (history) ----------
    const defaultUrl = base + '/#/g/liars-dice';
    const errors1 = [];
    const page1 = await browser.newPage({ viewport: { width: 1200, height: 860 } });
    page1.on('pageerror', (e) => errors1.push(String(e.message)));
    page1.on('console', (m) => {
      if (m.type() === 'error') errors1.push(m.text());
    });
    await page1.goto(defaultUrl, { waitUntil: 'load', timeout: 30000 });
    await page1.waitForSelector('.ld-seat', { timeout: 30000 });
    await page1.waitForFunction(
      () => document.querySelectorAll('.seat-select').length >= 4,
      { timeout: 15000 },
    );
    const rosterLabels = await page1.evaluate(() =>
      [...document.querySelectorAll('.seat-select')].map(
        (sel) => sel.selectedOptions[0]?.textContent ?? null,
      ),
    );
    await page1.screenshot({ path: join(OUT_DIR, 'ld-default.png') });
    await page1.close();

    const botLabels = rosterLabels.filter((l) => l !== 'You');
    const defaultIsNeural = botLabels.length > 0 && botLabels.every((l) => l === 'Neural');
    result.defaultBot = { rosterLabels, defaultIsNeural, consoleErrors: errors1 };

    // ---------- 2. bot moves are paced (~>=700ms apart) ----------
    const watchUrl = base + '/#/g/liars-dice?mode=watch';
    const errors2 = [];
    const page2 = await browser.newPage({ viewport: { width: 1200, height: 860 } });
    page2.on('pageerror', (e) => errors2.push(String(e.message)));
    page2.on('console', (m) => {
      if (m.type() === 'error') errors2.push(m.text());
    });
    await page2.goto(watchUrl, { waitUntil: 'load', timeout: 30000 });
    await page2.waitForSelector('.ld-seat', { timeout: 30000 });

    // Only count consecutive BID transitions (the fast-paced turn loop the
    // pacing fix targets) — while the LIAR/EXACT reveal banner is showing, the
    // table is mid-reveal (already generously paced on its own, ~900-1200ms
    // per step) so those frames are excluded rather than treated as bids.
    const changeTimes = [];
    let prevKey = null;
    const start = Date.now();
    for (let i = 0; i < 300; i++) {
      await page2.waitForTimeout(80);
      const key = await page2.evaluate(() => {
        if (document.querySelector('.ld-banner.ld-show')) return null;
        const bidMain = document.querySelector('.ld-bid-main');
        return bidMain ? bidMain.innerHTML : 'no-bid';
      });
      if (key !== null && prevKey !== null && key !== prevKey) changeTimes.push(Date.now());
      prevKey = key;
      if (Date.now() - start > 20000) break;
    }
    await page2.screenshot({ path: join(OUT_DIR, 'ld-watch.png') });
    await page2.close();

    const deltas = [];
    for (let i = 1; i < changeTimes.length; i++) deltas.push(changeTimes[i] - changeTimes[i - 1]);
    const n = deltas.length;
    const totalMs = changeTimes.length ? changeTimes[changeTimes.length - 1] - changeTimes[0] : 0;
    const avgMs = n > 0 ? totalMs / n : 0;
    const minMs = n > 0 ? Math.min(...deltas) : 0;

    result.pacing = {
      distinctStateChanges: changeTimes.length,
      transitions: n,
      totalMs,
      avgMsPerTransition: Math.round(avgMs),
      minMsPerTransition: minMs,
      deltas,
      consoleErrors: errors2,
    };

    result.ok =
      defaultIsNeural &&
      errors1.length === 0 &&
      n >= 4 &&
      avgMs >= 650 &&
      errors2.length === 0;
  } catch (e) {
    result.error = String(e);
  } finally {
    await browser.close().catch(() => {});
    server.close();
  }

  await writeFile(join(OUT_DIR, 'liars-dice.json'), JSON.stringify(result, null, 2));
  console.log('\n===== LIAR\'S DICE VALIDATION REPORT =====');
  console.log('default bot roster labels:', result.defaultBot?.rosterLabels);
  console.log('pacing:', result.pacing);
  if (result.error) console.log(`ERROR ${result.error}`);
  console.log('screenshots', `${OUT_DIR}/ld-default.png, ${OUT_DIR}/ld-watch.png`);
  console.log('pass', result.ok);
  console.log('===================================');
  process.exit(result.ok ? 0 : 1);
}

main();
