// Differential conformance harness for browser DOOM.
//
// Plays the WAD's deterministic built-in demo (`-playdemo demo1`) under two
// renderers and compares the on-screen canvas frame by frame:
//
//   REFERENCE = software renderer (full putImageData every present == exactly
//               the engine's I_VideoBuffer, i.e. "real Doom" output)
//   SUBJECT   = whatever loader page you pass (e.g. the GL build that smears)
//
// Because both runs replay the SAME demo, the engine produces identical frames
// tic-for-tic; any canvas difference is purely the presentation/blit path. Two
// independent browser runs are not tic-synchronised in wall-clock, so we capture
// a dense burst from each and, for every subject frame, find its BEST-matching
// reference frame (minimum mean-abs pixel distance). The residual after best
// alignment is the smear: pixels in the subject frame that match no reference
// frame at that location. divergence ~0 == conformant.
//
//   node differential.mjs <subjectHtml> <referenceHtml> [label]
//
// Outputs: out/diff/<label>/{subject_*,ref_*}.png, worst-frame diff image, and
// a report.json with the per-frame and aggregate divergence.

import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile, mkdir, writeFile, rm } from 'node:fs/promises';
import { extname, join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from './png.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const SANDBOX = join(HERE, 'sandbox');

// Both pages replay the WAD's built-in demo1 (deterministic, identical engine
// frames tic-for-tic), so the only difference between runs is the renderer.
const SUBJECT = process.argv[2] ?? 'doom-demo-gl.html';
const REFERENCE = process.argv[3] ?? 'doom-demo-soft.html';
const LABEL = process.argv[4] ?? 'gl-vs-soft';
const OUT = join(HERE, 'out', 'diff', LABEL.replace(/[^\w.-]/g, '_'));

const BURST_FRAMES = 130; // dense capture from boot (~7s at 55ms)
const FRAME_MS = 55; // capture cadence
const WINDOW = 36; // gameplay window length (frames) selected by motion

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.wad': 'application/octet-stream',
  '.lmp': 'application/octet-stream',
  '.cfg': 'text/plain; charset=utf-8',
};

function startServer(root, indexName) {
  const s = createServer(async (req, res) => {
    try {
      let p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
      if (p.endsWith('/')) p += indexName;
      const body = await readFile(join(root, p));
      res.writeHead(200, { 'content-type': MIME[extname(p)] ?? 'application/octet-stream' });
      res.end(body);
    } catch {
      res.writeHead(404).end();
    }
  });
  return new Promise((r) => s.listen(0, '127.0.0.1', () => r(s)));
}

const sleep = (p, ms) => p.waitForTimeout(ms);

// Capture a burst of decoded RGBA frames from a loader page once it goes live.
async function captureRun(browser, base, html) {
  const page = await browser.newPage({ viewport: { width: 700, height: 560 } });
  const logs = [];
  page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
  let gl = false;
  await page.goto(`${base}/${html}`, { waitUntil: 'load' });
  await page.evaluate(() => {
    window.__gl = false;
    const g = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (t, a) {
      if (/webgl/i.test(t)) window.__gl = true;
      return g.call(this, t, a);
    };
  });
  await page.click('#start').catch(() => {});
  const canvas = page.locator('#canvas');
  await canvas.waitFor({ state: 'visible' });

  // Wait for the first non-black frame (boot complete).
  for (let i = 0; i < 250; i++) {
    const png = PNG.decode(await canvas.screenshot());
    let nb = 0;
    for (let j = 0; j < png.data.length; j += 4)
      if (png.data[j] || png.data[j + 1] || png.data[j + 2]) nb++;
    if (nb > 2000) break;
    await sleep(page, 60);
  }

  // Dense burst from here through the title screen into the demo gameplay; the
  // gameplay window is selected later by motion (renderer-agnostic).
  const frames = [];
  for (let i = 0; i < BURST_FRAMES; i++) {
    frames.push(PNG.decode(await canvas.screenshot()));
    await sleep(page, FRAME_MS);
  }
  gl = await page.evaluate(() => window.__gl);
  await page.close();
  return { frames, gl, logs };
}

// Mean absolute RGB distance over the 3D viewport (exclude bottom HUD, which is
// static and identical in both, so it doesn't dilute the smear signal).
function frameDist(a, b) {
  const w = a.width;
  const h = a.height;
  const viewH = Math.floor(h * 0.78);
  let sum = 0;
  let n = 0;
  for (let y = 0; y < viewH; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      sum += Math.abs(a.data[i] - b.data[i]);
      sum += Math.abs(a.data[i + 1] - b.data[i + 1]);
      sum += Math.abs(a.data[i + 2] - b.data[i + 2]);
      n += 3;
    }
  }
  return sum / n;
}

function diffImage(a, b) {
  const w = a.width;
  const h = a.height;
  const out = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const j = i * 4;
    const d = Math.max(
      Math.abs(a.data[j] - b.data[j]),
      Math.abs(a.data[j + 1] - b.data[j + 1]),
      Math.abs(a.data[j + 2] - b.data[j + 2]),
    );
    out[j] = d > 16 ? 255 : 0;
    out[j + 1] = d > 16 ? Math.min(255, d) : 0;
    out[j + 2] = 0;
    out[j + 3] = 255;
  }
  return { width: w, height: h, data: out };
}

async function main() {
  await rm(OUT, { recursive: true, force: true });
  await mkdir(OUT, { recursive: true });

  // Both runs share the same sandbox; index name only matters for "/".
  const server = await startServer(SANDBOX, SUBJECT);
  const { port } = server.address();
  const base = `http://127.0.0.1:${port}`;
  const browser = await chromium.launch({
    headless: true,
    args: ['--headless=new', '--use-angle=metal', '--ignore-gpu-blocklist'],
  });

  const ref = await captureRun(browser, base, REFERENCE);
  const sub = await captureRun(browser, base, SUBJECT);

  // Select each run's GAMEPLAY window: the contiguous WINDOW frames with the most
  // total inter-frame motion (the demo's active movement, not the static title).
  function gameplayWindow(frames) {
    const mot = [];
    for (let i = 1; i < frames.length; i++) mot.push(frameDist(frames[i], frames[i - 1]));
    let bestStart = 0;
    let bestSum = -1;
    for (let s = 0; s + WINDOW < frames.length; s++) {
      let sum = 0;
      for (let k = s; k < s + WINDOW; k++) sum += mot[k] ?? 0;
      if (sum > bestSum) {
        bestSum = sum;
        bestStart = s;
      }
    }
    return frames.slice(bestStart, bestStart + WINDOW);
  }
  const refF = gameplayWindow(ref.frames);
  const subF = gameplayWindow(sub.frames);

  // For each subject frame, best-matching reference frame.
  const perFrame = [];
  let worst = { dist: -1 };
  for (let i = 0; i < subF.length; i++) {
    let best = Infinity;
    let bestJ = -1;
    for (let j = 0; j < refF.length; j++) {
      const d = frameDist(subF[i], refF[j]);
      if (d < best) {
        best = d;
        bestJ = j;
      }
    }
    perFrame.push({ subjectFrame: i, bestRef: bestJ, dist: +best.toFixed(3) });
    if (best > worst.dist) worst = { dist: best, i, bestJ };
  }

  const dists = perFrame.map((p) => p.dist).sort((a, b) => a - b);
  const median = dists[Math.floor(dists.length / 2)];
  const p90 = dists[Math.floor(dists.length * 0.9)];

  // Self-consistency sanity: reference vs reference best-match (should be ~0,
  // proving the harness alignment isn't the source of any divergence).
  const refSelf = [];
  for (let i = 1; i < refF.length; i++) {
    let best = Infinity;
    for (let j = 0; j < refF.length; j++) {
      if (j === i) continue;
      const d = frameDist(refF[i], refF[j]);
      if (d < best) best = d;
    }
    refSelf.push(best);
  }
  refSelf.sort((a, b) => a - b);
  const refSelfMedian = refSelf[Math.floor(refSelf.length / 2)] ?? 0;

  // Save the worst diverging subject frame, its best reference, and the diff.
  if (worst.i != null) {
    await writeFile(join(OUT, 'worst_subject.png'), PNG.encode(subF[worst.i]));
    await writeFile(join(OUT, 'worst_reference.png'), PNG.encode(refF[worst.bestJ]));
    await writeFile(join(OUT, 'worst_diff.png'), PNG.encode(diffImage(subF[worst.i], refF[worst.bestJ])));
  }
  // A few representative frames for eyeballing.
  for (const idx of [0, Math.floor(subF.length / 2), subF.length - 1]) {
    if (subF[idx]) await writeFile(join(OUT, `subject_${idx}.png`), PNG.encode(subF[idx]));
  }

  // Decisive boot-time signals (single-run, no cross-run alignment noise): a
  // grabbed WebGL context + a 0x0 upscale texture are the GL-smear fingerprint.
  const upscale0x0 = (logs) => logs.some((l) => /CreateUpscaledTexture.*0x0/.test(l));
  const report = {
    label: LABEL,
    subject: { html: SUBJECT, gl: sub.gl, upscaleTexture0x0: upscale0x0(sub.logs) },
    reference: { html: REFERENCE, gl: ref.gl, upscaleTexture0x0: upscale0x0(ref.logs) },
    framesCompared: subF.length,
    // The pixel divergence is alignment-limited (two browser runs are not
    // tic-synchronised), so treat it as a coarse check; the boot signals above
    // are the hard conformance gate.
    divergence: {
      median: +median.toFixed(3),
      p90: +p90.toFixed(3),
      worst: +worst.dist.toFixed(3),
    },
    harnessFloor: { refSelfMedian: +refSelfMedian.toFixed(3) },
    note: 'mean |dRGB| per channel over the 3D viewport, best-aligned. Coarse (cross-run timing-limited).',
  };
  await writeFile(
    join(OUT, 'report.json'),
    JSON.stringify({ report, perFrame, subjectLogs: sub.logs, refLogs: ref.logs }, null, 2),
  );
  console.log(JSON.stringify(report, null, 2));

  await browser.close();
  server.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
