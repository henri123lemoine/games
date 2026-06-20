// Mouse-look probe: boots a loader page into gameplay, then — WITHOUT touching
// the keyboard — wiggles the mouse over the canvas and measures whether the view
// changes. In a correct keyboard-only DOOM the view must stay put when only the
// mouse moves; if it turns/walks, stray ungrabbed mouse input is driving the
// player (the "acts strange when looking around" bug).
//
//   node probe-mouse.mjs <htmlFile> [label]

import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile, mkdir, writeFile, rm } from 'node:fs/promises';
import { extname, join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from './png.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const SANDBOX = join(HERE, 'sandbox');
const HTML = process.argv[2] ?? 'doom-baseline.html';
const LABEL = process.argv[3] ?? HTML;
const OUT = join(HERE, 'out', 'mouse', LABEL.replace(/[^\w.-]/g, '_'));

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.wad': 'application/octet-stream',
  '.cfg': 'text/plain; charset=utf-8',
};

function startServer(root) {
  const s = createServer(async (req, res) => {
    try {
      let p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
      if (p.endsWith('/')) p += HTML;
      const body = await readFile(join(root, p));
      res.writeHead(200, { 'content-type': MIME[extname(p)] ?? 'application/octet-stream' });
      res.end(body);
    } catch {
      res.writeHead(404).end();
    }
  });
  return new Promise((r) => s.listen(0, '127.0.0.1', () => r(s)));
}

function delta(aBuf, bBuf) {
  const a = PNG.decode(aBuf);
  const b = PNG.decode(bBuf);
  // Only look at the 3D viewport (exclude the bottom HUD ~ last 100px of 480).
  const viewH = Math.floor(a.height * 0.78);
  let diff = 0;
  for (let y = 0; y < viewH; y++) {
    for (let x = 0; x < a.width; x++) {
      const i = (y * a.width + x) * 4;
      let d = 0;
      for (let c = 0; c < 3; c++) {
        const dc = Math.abs(a.data[i + c] - b.data[i + c]);
        if (dc > d) d = dc;
      }
      if (d > 16) diff++;
    }
  }
  return { diffPixels: diff, viewPixels: viewH * a.width };
}

const sleep = (p, ms) => p.waitForTimeout(ms);

async function main() {
  await rm(OUT, { recursive: true, force: true });
  await mkdir(OUT, { recursive: true });
  const server = await startServer(SANDBOX);
  const { port } = server.address();
  const base = `http://127.0.0.1:${port}`;
  const browser = await chromium.launch({
    headless: true,
    args: ['--headless=new', '--use-angle=metal', '--ignore-gpu-blocklist'],
  });
  const page = await browser.newPage({ viewport: { width: 700, height: 560 } });
  const logs = [];
  page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));

  await page.goto(`${base}/${HTML}`, { waitUntil: 'load' });
  await page.click('#start').catch(() => {});
  const canvas = page.locator('#canvas');
  await canvas.waitFor({ state: 'visible' });

  // Wait for live gameplay.
  for (let i = 0; i < 250; i++) {
    const png = PNG.decode(await canvas.screenshot());
    let nb = 0;
    for (let j = 0; j < png.data.length; j += 4)
      if (png.data[j] || png.data[j + 1] || png.data[j + 2]) nb++;
    if (nb > 2000) break;
    await sleep(page, 100);
  }
  await sleep(page, 1000);

  const box = await canvas.boundingBox();
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;

  // Settle, baseline frame (no input at all).
  await page.mouse.move(cx, cy);
  await sleep(page, 600);
  const before = await canvas.screenshot();
  await writeFile(join(OUT, 'before.png'), before);

  // Wiggle the mouse horizontally across the canvas WITHOUT pressing any key.
  for (let k = 0; k < 6; k++) {
    await page.mouse.move(box.x + 40, cy, { steps: 8 });
    await sleep(page, 90);
    await page.mouse.move(box.x + box.width - 40, cy, { steps: 8 });
    await sleep(page, 90);
  }
  await page.mouse.move(cx, cy);
  await sleep(page, 400);
  const after = await canvas.screenshot();
  await writeFile(join(OUT, 'after.png'), after);

  const d = delta(before, after);
  const report = {
    label: LABEL,
    mouseMovedView: d,
    fracViewChanged: +(d.diffPixels / d.viewPixels).toFixed(4),
    verdict:
      d.diffPixels / d.viewPixels > 0.05
        ? 'VIEW MOVED on mouse-only input (stray mouse-look BUG)'
        : 'view stable under mouse-only input (ok)',
  };
  await writeFile(join(OUT, 'report.json'), JSON.stringify({ report, logs }, null, 2));
  console.log(JSON.stringify(report, null, 2));
  await browser.close();
  server.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
