// Quick timing probe: capture the canvas densely from boot to see WHEN the
// built-in demo actually renders (and for how long), so the differential can
// target the right window.
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile, mkdir, writeFile, rm } from 'node:fs/promises';
import { extname, join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from './png.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const SANDBOX = join(HERE, 'sandbox');
const HTML = process.argv[2] ?? 'doom-demo-soft.html';
const OUT = join(HERE, 'out', 'timing', HTML.replace(/[^\w.-]/g, '_'));
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm', '.wad': 'application/octet-stream', '.lmp': 'application/octet-stream', '.cfg': 'text/plain' };

function srv(root) {
  const s = createServer(async (req, res) => {
    try { let p = decodeURIComponent(new URL(req.url, 'http://x').pathname); if (p.endsWith('/')) p += HTML; const b = await readFile(join(root, p)); res.writeHead(200, { 'content-type': MIME[extname(p)] ?? 'application/octet-stream' }); res.end(b); } catch { res.writeHead(404).end(); }
  });
  return new Promise((r) => s.listen(0, '127.0.0.1', () => r(s)));
}
function sig(png) { // cheap per-frame signature: mean + nonblack count over viewport
  let sum = 0, nb = 0; const n = png.data.length;
  for (let i = 0; i < n; i += 4) { const v = png.data[i] + png.data[i + 1] + png.data[i + 2]; sum += v; if (v) nb++; }
  return { mean: +(sum / (n / 4) / 3).toFixed(1), nb };
}
async function main() {
  await rm(OUT, { recursive: true, force: true }); await mkdir(OUT, { recursive: true });
  const server = await srv(SANDBOX); const { port } = server.address(); const base = `http://127.0.0.1:${port}`;
  const browser = await chromium.launch({ headless: true, args: ['--headless=new', '--use-angle=metal', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 700, height: 560 } });
  const logs = []; page.on('console', (m) => logs.push(m.text()));
  await page.goto(`${base}/${HTML}`, { waitUntil: 'load' });
  await page.click('#start').catch(() => {});
  const canvas = page.locator('#canvas');
  await canvas.waitFor({ state: 'visible' });
  const sigs = [];
  const t0 = Date.now();
  for (let i = 0; i < 120; i++) { // ~6s at 50ms
    const png = PNG.decode(await canvas.screenshot());
    const s = sig(png); s.t = Date.now() - t0; sigs.push(s);
    if (i % 10 === 0) await writeFile(join(OUT, `t_${String(s.t).padStart(5, '0')}.png`), PNG.encode(png));
    await page.waitForTimeout(50);
  }
  // Report distinct "scenes" by mean change.
  let prev = null, changes = 0;
  for (const s of sigs) { if (prev !== null && Math.abs(s.mean - prev) > 3) changes++; prev = s.mean; }
  console.log('frames:', sigs.length, 'scene-changes(meanjump>3):', changes);
  console.log('mean trace:', sigs.filter((_, i) => i % 5 === 0).map((s) => `${s.t}:${s.mean}`).join(' '));
  console.log('demo logs:', logs.filter((l) => /demo|game started|map|gametic/i.test(l)).slice(0, 8));
  await writeFile(join(OUT, 'sigs.json'), JSON.stringify(sigs, null, 2));
  await browser.close(); server.close();
}
main().catch((e) => { console.error(e); process.exit(1); });
