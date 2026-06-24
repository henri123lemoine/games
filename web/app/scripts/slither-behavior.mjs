// Behavioral validation for slither: does the trained bot actually punish basic
// play in the corrected world (pellets=250, worms=6)?
//
// It launches the real headless Chromium, serves the built app, opens the
// slither screen, and drives the PLAYER worm with deliberately NAIVE mouse
// steering (slow circles / straight lines, like an unskilled human — no clever
// play). It then measures, over a meaningful game duration, whether the bot
// worms hunt the player and whether KILLS happen.
//
//   npm run validate:slither            # ~90s run, headless, screenshots
//   node scripts/slither-behavior.mjs --secs 120 --pattern lines --headed
//
// HOW METRICS ARE READ WITHOUT TOUCHING GAME CODE:
//   The slither HUD paints the leaderboard, the player's length, rank, and the
//   pellet field straight to the canvas; the "You died" panel is the only piece
//   in the DOM. So we wrap CanvasRenderingContext2D.fillText via addInitScript
//   to scrape the rendered leaderboard rows / length / rank each frame into
//   window.__slrHud, and read the death panel from the DOM. No frontend changes,
//   so this never collides with the game code another session may be editing.
//
// WebGPU/secure-context launch notes live in browser-validate.mjs.

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

const GPU_FLAGS = [
  '--headless=new',
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  '--use-angle=metal',
  '--ignore-gpu-blocklist',
];

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.weights': 'application/octet-stream',
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

// Injected before any app code: scrape the canvas-painted HUD into window.__slrHud
// so the harness can poll the live leaderboard / length / rank.
const HUD_SCRAPER = `
(() => {
  const proto = CanvasRenderingContext2D.prototype;
  const orig = proto.fillText;
  window.__slrHud = { lb: {}, length: null, rank: null, frame: 0 };
  // The leaderboard rows render "<n>. you" / "<n>. bot <seat>" then the length
  // string separately; we pair a row label with the next number drawn.
  let pendingLabel = null;
  proto.fillText = function (text, x, y) {
    try {
      const t = String(text);
      const row = t.match(/^(\\d+)\\.\\s+(you|bot \\d+)$/);
      if (row) {
        pendingLabel = row[2];
      } else if (pendingLabel && /^\\d+$/.test(t)) {
        window.__slrHud.lb[pendingLabel] = Number(t);
        pendingLabel = null;
      }
      const len = t.match(/^length (\\d+)$/);
      if (len) window.__slrHud.length = Number(len[1]);
      const rk = t.match(/^rank #(\\d+) of (\\d+)$/);
      if (rk) { window.__slrHud.rank = Number(rk[1]); window.__slrHud.worms = Number(rk[2]); }
    } catch (e) {}
    return orig.apply(this, arguments);
  };
  const raf = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) => raf((t) => { window.__slrHud.frame++; return cb(t); });
})();
`;

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : def;
}

async function main() {
  const secs = Number(arg('secs', '90'));
  const pattern = arg('pattern', 'circle'); // circle | lines | wander
  const headed = process.argv.includes('--headed');
  if (!existsSync(DIST)) {
    console.error(`no build at ${DIST} — run \`npx vite build\` first.`);
    process.exit(2);
  }
  await mkdir(OUT_DIR, { recursive: true });

  const { server, port } = await startServer(DIST);
  const url = `http://localhost:${port}/#/coil`;
  console.log(`[coil-behavior] serving dist/ at http://localhost:${port}`);
  console.log(`[coil-behavior] url=${url} secs=${secs} pattern=${pattern} headed=${headed}`);

  const browser = await chromium.launch({
    headless: false,
    args: headed ? GPU_FLAGS.filter((f) => f !== '--headless=new') : GPU_FLAGS,
  });

  const logs = [];
  const result = { secs, pattern, url, ok: false, samples: [], deaths: [], shots: [] };
  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 800 } });
    page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
    page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
    await page.addInitScript(HUD_SCRAPER);

    await page.goto(url, { waitUntil: 'load', timeout: 30000 });
    await page.waitForSelector('.slr-canvas', { timeout: 30000 });
    // Wait out the "Loading the trained bot…" boot element.
    await page
      .waitForFunction(() => !document.querySelector('.slr-boot'), { timeout: 30000 })
      .catch(() => {});
    await page.waitForTimeout(500);

    const box = await page.locator('.slr-canvas').boundingBox();
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;
    const R = Math.min(box.width, box.height) * 0.3;

    // Drive the player with the chosen NAIVE pattern. Deliberately unskilled:
    // no boosting, no reacting to threats, no food-seeking — the behavior the
    // old mis-deployed bot supposedly couldn't beat.
    await page.mouse.move(cx + R, cy);
    const endAt = Date.now() + secs * 1000;
    let theta = 0;
    let step = 0;
    let lastShot = 0;
    let everDead = false;
    let firstDeathAt = null;
    const startAt = Date.now();

    while (Date.now() < endAt) {
      step++;
      const elapsed = (Date.now() - startAt) / 1000;
      if (pattern === 'circle') {
        theta += 0.1;
        await page.mouse.move(cx + Math.cos(theta) * R, cy + Math.sin(theta) * R, { steps: 2 });
      } else if (pattern === 'lines') {
        // Sweep straight across, reverse at the edges (naive back-and-forth).
        const phase = (step % 40) / 40;
        const tx = cx + (phase < 0.5 ? phase * 4 - 1 : 3 - phase * 4) * R;
        await page.mouse.move(tx, cy + (Math.floor(step / 40) % 2 ? -R * 0.5 : R * 0.5), {
          steps: 2,
        });
      } else {
        // wander: large lazy lissajous, still no skill.
        theta += 0.07;
        await page.mouse.move(cx + Math.cos(theta) * R, cy + Math.sin(theta * 1.6) * R * 0.7, {
          steps: 2,
        });
      }
      await page.waitForTimeout(180);

      // Sample HUD + death state roughly every second.
      if (Date.now() - lastShot > 1000) {
        lastShot = Date.now();
        const snap = await page.evaluate(() => {
          const o = document.querySelector('.slr-overlay');
          const dead = !!o && o.classList.contains('slr-show');
          return {
            hud: window.__slrHud
              ? { lb: { ...window.__slrHud.lb }, length: window.__slrHud.length, rank: window.__slrHud.rank, worms: window.__slrHud.worms, frame: window.__slrHud.frame }
              : null,
            dead,
            deathTitle: document.querySelector('.slr-over-title')?.textContent ?? '',
            deathSub: document.querySelector('.slr-over-sub')?.textContent ?? '',
          };
        });
        result.samples.push({ t: Number(elapsed.toFixed(1)), ...snap });

        if (snap.dead && !everDead) {
          everDead = true;
          firstDeathAt = elapsed;
          result.deaths.push({ t: Number(elapsed.toFixed(1)), sub: snap.deathSub });
          const shot = join(OUT_DIR, `slither-death.png`);
          await page.screenshot({ path: shot });
          result.shots.push(shot);
          // Respawn and keep going so we observe whether it dies repeatedly.
          await page.locator('.slr-restart').click().catch(() => {});
          everDead = false;
        }
      }

      // Timed screenshots across the run for the visual record.
      const phaseShot = Math.floor(elapsed / Math.max(1, secs / 4));
      const shotPath = join(OUT_DIR, `slither-behavior-${phaseShot}.png`);
      if (!result.shots.includes(shotPath) && elapsed > phaseShot * (secs / 4)) {
        await page.screenshot({ path: shotPath });
        result.shots.push(shotPath);
      }
    }

    // Final frame.
    const finalShot = join(OUT_DIR, 'slither-behavior-final.png');
    await page.screenshot({ path: finalShot });
    result.shots.push(finalShot);
    result.firstDeathAt = firstDeathAt;
    result.ok = true;
  } catch (e) {
    result.error = String(e);
  } finally {
    result.consoleTail = logs.slice(-8);
    await browser.close().catch(() => {});
    server.close();
  }

  // --- analysis ---
  const deaths = result.deaths;
  const lens = result.samples.map((s) => s?.hud?.length).filter((n) => typeof n === 'number');
  const playerMaxLen = lens.length ? Math.max(...lens) : null;
  const botLens = [];
  for (const s of result.samples) {
    if (!s?.hud?.lb) continue;
    for (const [k, v] of Object.entries(s.hud.lb)) if (k !== 'you' && typeof v === 'number') botLens.push(v);
  }
  const botMaxLen = botLens.length ? Math.max(...botLens) : null;
  result.analysis = {
    durationSecs: secs,
    pattern,
    deathCount: deaths.length,
    firstDeathAtSecs: result.firstDeathAt,
    playerMaxLength: playerMaxLen,
    botMaxLength: botMaxLen,
    botOutgrewPlayer: botMaxLen != null && playerMaxLen != null ? botMaxLen > playerMaxLen : null,
  };

  await writeFile(join(OUT_DIR, 'slither-behavior.json'), JSON.stringify(result, null, 2));

  console.log('\n===== SLITHER BEHAVIOR REPORT =====');
  console.log(`pattern=${pattern}  duration=${secs}s  frames sampled=${result.samples.length}`);
  console.log(`player deaths: ${deaths.length}`);
  for (const d of deaths) console.log(`  died at ${d.t}s — ${d.sub}`);
  console.log(`first death at: ${result.firstDeathAt ?? 'never'}${result.firstDeathAt ? 's' : ''}`);
  console.log(`player max length: ${playerMaxLen}`);
  console.log(`bot max length:    ${botMaxLen}`);
  console.log(`bot outgrew player: ${result.analysis.botOutgrewPlayer}`);
  const lastLb = [...result.samples].reverse().find((s) => s?.hud?.lb && Object.keys(s.hud.lb).length);
  if (lastLb) console.log(`final leaderboard: ${JSON.stringify(lastLb.hud.lb)}`);
  console.log(`screenshots: ${result.shots.join(', ')}`);
  console.log('console tail:');
  for (const l of result.consoleTail) console.log('  ' + l);
  if (result.error) console.log(`ERROR ${result.error}`);
  console.log('===================================');
  process.exit(result.ok ? 0 : 1);
}

main();
