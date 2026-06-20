// Parametric DOOM smear probe: boots a loader page, drives a deterministic input
// sequence INTO GAMEPLAY (past the title/skill menus), then measures the
// frame-buffer smear and reports the renderer Chocolate Doom created.
//
//   node probe.mjs <htmlFile> [label]
//
// SMEAR METRIC: in gameplay, do a sustained ~180-degree turn, then STOP and hold
// still for two samples 350ms apart. Correct Doom repaints the full viewport
// every frame, so once the view is static the two idle frames are near-identical
// (delta ~0). The GL bug leaves stale geometry from the spin lingering on the
// canvas (it never clears the framebuffer / the upscale texture is 0x0), so the
// two idle frames differ a lot. We also flag a WebGL context grab and a 0x0
// upscale texture, both signatures of the broken GL present path.

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
const OUT = join(HERE, 'out', 'probe', LABEL.replace(/[^\w.-]/g, '_'));

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.wad': 'application/octet-stream',
  '.cfg': 'text/plain; charset=utf-8',
};

function startServer(rootDir) {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://localhost');
      let p = decodeURIComponent(url.pathname);
      if (p.endsWith('/')) p += HTML;
      const body = await readFile(join(rootDir, p));
      res.writeHead(200, { 'content-type': MIME[extname(p)] ?? 'application/octet-stream' });
      res.end(body);
    } catch {
      res.writeHead(404).end();
    }
  });
  return new Promise((r) => server.listen(0, '127.0.0.1', () => r(server)));
}

function pngDelta(aBuf, bBuf) {
  const a = PNG.decode(aBuf);
  const b = PNG.decode(bBuf);
  if (a.width !== b.width || a.height !== b.height)
    return { diffPixels: -1, maxChan: -1, fracDiff: -1 };
  let diff = 0;
  let maxChan = 0;
  for (let i = 0; i < a.data.length; i += 4) {
    let d = 0;
    for (let c = 0; c < 3; c++) {
      const dc = Math.abs(a.data[i + c] - b.data[i + c]);
      if (dc > d) d = dc;
    }
    if (d > 16) diff++;
    if (d > maxChan) maxChan = d;
  }
  const total = (a.width * a.height) | 0;
  return { diffPixels: diff, maxChan, total, fracDiff: +(diff / total).toFixed(4) };
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
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));

  await page.goto(`${base}/${HTML}`, { waitUntil: 'load' });
  await page.evaluate(() => {
    window.__t = { gl: false, attrs: null };
    const g = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, a) {
      if (/webgl/i.test(type)) {
        window.__t.gl = true;
        window.__t.attrs = a ?? null;
      }
      return g.call(this, type, a);
    };
  });

  await page.click('#start').catch(() => {});
  const canvas = page.locator('#canvas');
  await canvas.waitFor({ state: 'visible' });

  // Wait for the engine to present live (non-black) frames.
  let live = false;
  for (let i = 0; i < 250; i++) {
    const png = PNG.decode(await canvas.screenshot());
    let nb = 0;
    for (let j = 0; j < png.data.length; j += 4)
      if (png.data[j] || png.data[j + 1] || png.data[j + 2]) nb++;
    if (nb > 2000) {
      live = true;
      break;
    }
    await sleep(page, 100);
  }

  // -warp 1 1 drops straight into E1M1 gameplay, no menu navigation needed.
  // Give the map a moment to settle, then dismiss any "press a key" prompt.
  await sleep(page, 800);
  await writeFile(join(OUT, 'gameplay_start.png'), await canvas.screenshot());

  // SUSTAINED SPIN, then stop and sample two idle frames.
  await page.keyboard.down('ArrowRight');
  await sleep(page, 900);
  await page.keyboard.up('ArrowRight');
  await sleep(page, 500); // settle

  const idleA = await canvas.screenshot();
  await writeFile(join(OUT, 'idleA.png'), idleA);
  await sleep(page, 350);
  const idleB = await canvas.screenshot();
  await writeFile(join(OUT, 'idleB.png'), idleB);
  const idleDelta = pngDelta(idleA, idleB);

  // Save a mid-spin frame for eyeballing the smear directly.
  await page.keyboard.down('ArrowLeft');
  await sleep(page, 220);
  const spin = await canvas.screenshot();
  await page.keyboard.up('ArrowLeft');
  await writeFile(join(OUT, 'spin.png'), spin);

  const t = await page.evaluate(() => window.__t);
  const report = {
    label: LABEL,
    live,
    glContextRequested: t.gl,
    glAttrs: t.attrs,
    createdRenderer: logs.find((l) => l.includes('Created renderer')) ?? '(none logged)',
    upscaleTexture: logs.find((l) => l.includes('CreateUpscaledTexture')) ?? '(none logged)',
    idleDelta,
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
