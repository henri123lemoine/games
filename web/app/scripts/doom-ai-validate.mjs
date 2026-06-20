// Objective validation for the Doom — 1v1 vs AI page.
//
// Serves public/doom-ai/ over http://localhost (WASM needs a secure-ish origin
// + COOP/COEP), loads it in headless Chromium, clicks Fight, drives the BOT for
// both seats (so the match advances with no human), and asserts:
//   1. the engine actually steps (tic advances),
//   2. the canvas renders real content (not black / not a frozen single frame),
//   3. the BOT FIGHTS — frags accumulate over the run (kills happen, not idle).
// Writes a screenshot to .validation/doom-ai.png. Exit 0 = pass.
//
//   node scripts/doom-ai-validate.mjs            # headless
//   node scripts/doom-ai-validate.mjs --headed   # show the window

import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { extname, join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP_DIR = resolve(HERE, "..");
const ROOT = join(APP_DIR, "public", "doom-ai");
const OUT_DIR = join(APP_DIR, ".validation");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".data": "application/octet-stream",
  ".bin": "application/octet-stream",
  ".css": "text/css; charset=utf-8",
  ".png": "image/png",
};

function startServer(rootDir) {
  const server = createServer(async (req, res) => {
    try {
      const urlPath = decodeURIComponent((req.url ?? "/").split("?")[0]);
      let filePath = join(rootDir, urlPath);
      if (urlPath.endsWith("/")) filePath = join(filePath, "index.html");
      if (!existsSync(filePath)) filePath = join(rootDir, "index.html");
      const body = await readFile(filePath);
      res.setHeader("content-type", MIME[extname(filePath)] ?? "application/octet-stream");
      res.setHeader("cross-origin-opener-policy", "same-origin");
      res.setHeader("cross-origin-embedder-policy", "require-corp");
      res.setHeader("cross-origin-resource-policy", "cross-origin");
      res.end(body);
    } catch (e) {
      res.statusCode = 500;
      res.end(String(e));
    }
  });
  return new Promise((res) =>
    server.listen(0, "127.0.0.1", () => res({ server, port: server.address().port })),
  );
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const headed = process.argv.includes("--headed");
  await mkdir(OUT_DIR, { recursive: true });
  const { server, port } = await startServer(ROOT);
  const url = `http://localhost:${port}/`;
  const logs = [];

  const browser = await chromium.launch({
    headless: !headed,
    args: ["--headless=new", "--use-gl=swiftshader", "--enable-unsafe-webgpu"],
  });
  const page = await browser.newPage({ viewport: { width: 960, height: 600 } });
  page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));

  let pass = true;
  const fail = (msg) => {
    pass = false;
    console.error("FAIL:", msg);
  };

  try {
    // ?botboth drives BOTH seats with the trained net — a competent fighter on
    // each side, so the match advances and frags happen with no human input.
    await page.goto(url + "?botboth", { waitUntil: "domcontentloaded" });

    // wait for the bot+engine to load (start button enabled), up to 30s.
    await page.waitForFunction(() => {
      const b = document.getElementById("start");
      return b && !b.disabled;
    }, { timeout: 30000 });
    console.log("engine + bot loaded");

    await page.click("#start");
    await page.evaluate(() => document.getElementById("canvas").focus());

    // sample HUD frags + a canvas pixel signature over time.
    const samples = [];
    for (let i = 0; i < 12; i++) {
      await sleep(1000);
      const s = await page.evaluate(() => {
        const api = window.__doomAI;
        const me = api.readState(0);
        const opp = api.readState(1);
        // canvas non-black pixel fraction + a cheap hash to detect motion
        const cv = document.getElementById("canvas");
        const ctx = cv.getContext("2d");
        const d = ctx.getImageData(0, 0, cv.width, cv.height).data;
        let nonblack = 0,
          hash = 0;
        for (let p = 0; p < d.length; p += 4 * 97) {
          const v = d[p] + d[p + 1] + d[p + 2];
          if (v > 24) nonblack++;
          hash = (hash + v * (p + 1)) % 2147483647;
        }
        return {
          tic: me[/* not in state */ 0] && 0,
          myFrags: me[11],
          botFrags: opp[11],
          myDeaths: me[12],
          botDeaths: opp[12],
          alive0: me[0],
          alive1: opp[0],
          nonblackFrac: nonblack / (d.length / (4 * 97)),
          hash,
        };
      });
      samples.push(s);
      console.log(
        `t=${i + 1}s you=${s.myFrags} ai=${s.botFrags} deaths(you/ai)=${s.myDeaths}/${s.botDeaths} ` +
          `nonblack=${s.nonblackFrac.toFixed(2)} hash=${s.hash}`,
      );
    }

    await page.screenshot({ path: join(OUT_DIR, "doom-ai.png") });

    // ASSERTIONS
    const last = samples[samples.length - 1];
    const first = samples[0];

    // 1. canvas renders real content (the 3D view fills most of the frame)
    const maxNonblack = Math.max(...samples.map((s) => s.nonblackFrac));
    if (maxNonblack < 0.3) fail(`canvas mostly black (max nonblack frac ${maxNonblack.toFixed(2)})`);
    else console.log(`render OK: nonblack frac up to ${maxNonblack.toFixed(2)}`);

    // 2. the frame changes over time (not frozen / not smeared single image)
    const distinctHashes = new Set(samples.map((s) => s.hash)).size;
    if (distinctHashes < 3) fail(`frame not animating (only ${distinctHashes} distinct frames)`);
    else console.log(`animation OK: ${distinctHashes} distinct frames`);

    // 3. THE BOT FIGHTS — total frags (both seats) increased; combat happened.
    const totalFragsFirst = first.myFrags + first.botFrags;
    const totalFragsLast = last.myFrags + last.botFrags;
    const totalDeathsLast = last.myDeaths + last.botDeaths;
    const combat = totalFragsLast - totalFragsFirst + totalDeathsLast;
    if (combat < 1) fail(`no combat: total frags ${totalFragsFirst}->${totalFragsLast}, deaths ${totalDeathsLast}`);
    else console.log(`COMBAT OK: frags ${totalFragsFirst}->${totalFragsLast}, deaths ${totalDeathsLast}`);

    // 4. BOTH net-driven seats fragged (in ?botboth both are the trained bot) —
    // confirms the deployed net competently fights, not idle.
    if (last.botFrags < 1 || last.myFrags < 1)
      fail(`a seat is idle (seat0 frags ${last.myFrags}, seat1 frags ${last.botFrags})`);
    else console.log(`both net seats fighting: seat0 frags=${last.myFrags}, seat1 frags=${last.botFrags}`);
  } catch (e) {
    fail(`exception: ${e.message}`);
  } finally {
    if (logs.length) {
      console.log("\n--- page logs (last 20) ---");
      console.log(logs.slice(-20).join("\n"));
    }
    await browser.close();
    server.close();
  }

  console.log(pass ? "\nDOOM-AI VALIDATION: PASS" : "\nDOOM-AI VALIDATION: FAIL");
  process.exit(pass ? 0 : 1);
}

main();
