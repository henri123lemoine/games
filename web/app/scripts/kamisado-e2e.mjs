// End-to-end check for the Kamisado arcade page (run from web/app after
// `npm run build`): serves dist/, drives two games against the default
// depth-16 alpha-beta bot in headless Chromium, and asserts both the UI flow
// and the bot's play.
//
//   Game 1: the scripted human blunders d1-d7 — the bot must find the forced
//           refutation (g8-b3 then b3-d1, through a collapsed pass) and win.
//   Game 2: the human plays a1-a5 (the winning opening); the obligation flow
//           must auto-select Black's single obligated tower on the next turn.
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile, mkdir } from 'node:fs/promises';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const APP_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DIST = join(APP_DIR, 'dist');
const OUT = join(APP_DIR, '.validation', 'kamisado');
const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.wasm': 'application/wasm', '.png': 'image/png',
  '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.woff2': 'font/woff2',
};

const server = createServer(async (req, res) => {
  const path = req.url.split('?')[0];
  const file = join(DIST, path === '/' ? 'index.html' : path);
  try {
    const body = await readFile(file);
    res.writeHead(200, { 'content-type': MIME[extname(file)] ?? 'application/octet-stream' });
    res.end(body);
  } catch {
    const body = await readFile(join(DIST, 'index.html'));
    res.writeHead(200, { 'content-type': 'text/html' });
    res.end(body);
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}/`;

await mkdir(OUT, { recursive: true });
// CHROMIUM_PATH overrides the browser binary (for hosts whose preinstalled
// Chromium doesn't match the npm playwright version); --headless=new keeps
// the full binary windowless.
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--headless=new'],
});
const page = await browser.newPage({ viewport: { width: 1100, height: 900 } });
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push(String(e)));

const assert = (cond, what) => { if (!cond) throw new Error(`FAILED: ${what}`); };

// ---- Game 1: blunder d1-d7; the bot must convert its forced win. ----
await page.goto(url);
await page.waitForSelector('.card[data-game="kamisado"]', { timeout: 30000 });
await page.screenshot({ path: join(OUT, '1-home.png') });
await page.click('.card[data-game="kamisado"]');
await page.waitForSelector('.km-tower.km-pickable', { timeout: 30000 });
assert((await page.locator('.km-tower.km-pickable').count()) === 8, 'opening offers all 8 towers');
await page.screenshot({ path: join(OUT, '2-board.png') });

await page.locator('.km-tower.km-pickable').nth(3).click(); // the d1 tower
assert((await page.locator('.km-tile.km-target').count()) === 13, 'd1 tower has 13 destinations');
await page.screenshot({ path: join(OUT, '3-selected.png') });
await page.locator('.km-tile.km-target').first().click(); // topmost target = d7, the blunder

const t0 = Date.now();
await page.waitForFunction(
  () => document.querySelector('.status')?.textContent?.includes('takes the round'),
  { timeout: 60000 },
);
const result = await page.locator('.status').innerText();
assert(result.includes('White takes the round'), `bot converts the forced win (got: ${result})`);
console.log(`game 1: bot punished d1-d7 and won in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
await page.screenshot({ path: join(OUT, '4-bot-wins.png') });

// ---- Game 2: the winning opening; check the obligation auto-select. ----
await page.goto(url);
await page.click('.card[data-game="kamisado"]');
await page.waitForSelector('.km-tower.km-pickable', { timeout: 30000 });
await page.locator('.km-tower.km-pickable').nth(0).click(); // the a1 tower
await page.waitForSelector('.km-tile.km-target', { timeout: 5000 });
await page.locator('.km-grid > .km-tile').nth(24).click(); // a5

await page.waitForSelector('.km-tower.km-pickable', { timeout: 60000 });
assert((await page.locator('.km-tower.km-pickable').count()) === 1, 'exactly one obligated tower');
assert((await page.locator('.km-tower.km-selected').count()) === 1, 'obligated tower auto-selected');
assert((await page.locator('.km-tile.km-target').count()) > 0, 'its destinations are lit');
const msg = await page.locator('.km-msg').innerText();
assert(/Black.*must move/.test(msg), `obligation message (got: ${msg})`);
console.log(`game 2: after a1-a5 the bot replied; status: ${JSON.stringify(msg)}`);
assert((await page.locator('.km-tower.km-obliged').count()) === 1, 'obligation pulse on the tower');
await page.screenshot({ path: join(OUT, '5-obligation.png') });

assert(errors.length === 0, `no console errors (got: ${errors.join(' | ')})`);
console.log('kamisado e2e: all assertions passed');
await browser.close();
server.close();
