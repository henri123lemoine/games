// The heavyweight immutable payloads (trained nets, solver tables) are served
// from the `arcade-assets` R2 bucket instead of shipping inside the deployed
// bundle. Objects are content-addressed (`<dir>/<stem>.<sha256><ext>`), so a
// URL never changes meaning: old deploys keep resolving the bytes they were
// built against, new uploads never overwrite anything, and everything is safe
// to cache forever (`Cache-Control: immutable`).
//
// This module is the single source of truth for which files that covers and
// where they live. `vite.config.ts` imports `computeManifest()` to bake the
// URLs into production builds (dev serves the same files from `public/`), and
// CI runs `verify` + `prune` so a deploy can never reference bytes that are
// not already in the bucket.
//
//   node scripts/r2-assets.mjs upload   # publish missing objects (wrangler login)
//   node scripts/r2-assets.mjs verify   # all URLs live? (CI gate, no auth)
//   node scripts/r2-assets.mjs prune    # drop the payloads from dist/
//   node scripts/r2-assets.mjs manifest # print the path -> URL map

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, rmSync } from 'node:fs';
import { basename, dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const APP_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const PUBLIC_DIR = join(APP_DIR, 'public');
const DIST_DIR = join(APP_DIR, 'dist');

const ASSET_HOST = 'https://arcade-assets.henrilemoine.com';
const BUCKET = 'arcade-assets';
const CACHE_CONTROL = 'public, max-age=31536000, immutable';
const CONTENT_TYPE = 'application/octet-stream';

/** Everything in public/ above ~1 MB whose fetch the app owns. The doom and
 * doom-ai payloads stay local: they are loaded by their own iframe-relative
 * emscripten glue, not through `assetUrl`. */
const R2_ASSETS = [
  'artifacts/ataraxios.bin',
  'artifacts/azero-chess.bin',
  'artifacts/ld-history-champion.bin',
  'artifacts/t21-solver-h3.bin',
  'artifacts/t21-solver-h6.bin',
  'azero/azero-chess.azweb',
  'azero/azero-go.azweb',
  'azero/azero-pente.azweb',
  'slither/slither.weights',
];

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function assets() {
  return R2_ASSETS.map((path) => {
    const ext = extname(path);
    const digest = sha256(readFileSync(join(PUBLIC_DIR, path)));
    const key = `${dirname(path)}/${basename(path, ext)}.${digest}${ext}`;
    return { path, digest, key, url: `${ASSET_HOST}/${key}` };
  });
}

export function computeManifest() {
  return Object.fromEntries(assets().map((a) => [a.path, a.url]));
}

/** The edge caches responses for these URLs (including brief 404s from
 * pre-upload probes), so existence and verification requests carry a unique
 * query string to force an origin fetch. R2 ignores it for object lookup. */
function uncached(url) {
  return `${url}?probe=${process.pid}.${process.hrtime.bigint()}`;
}

async function exists(url) {
  const resp = await fetch(uncached(url), { method: 'HEAD' });
  if (resp.ok) return true;
  if (resp.status === 404) return false;
  throw new Error(`HEAD ${url} -> HTTP ${resp.status}`);
}

async function verify() {
  const missing = [];
  for (const { path, url } of assets()) {
    if (await exists(url)) console.log(`ok      ${path}`);
    else {
      console.error(`MISSING ${path} -> ${url}`);
      missing.push(path);
    }
  }
  if (missing.length > 0) {
    console.error(
      `\n${missing.length} asset(s) not in the bucket. Publish them with\n` +
        `  node scripts/r2-assets.mjs upload\n` +
        `(needs \`npx wrangler login\` on a machine with the files).`,
    );
    process.exit(1);
  }
}

const WRANGLER = (process.env.WRANGLER ?? 'npx wrangler@4').split(' ');

function wrangler(...args) {
  execFileSync(WRANGLER[0], [...WRANGLER.slice(1), ...args], { stdio: 'inherit' });
}

async function upload() {
  for (const { path, digest, key, url } of assets()) {
    if (await exists(url)) {
      console.log(`exists  ${path}`);
      continue;
    }
    console.log(`upload  ${path} -> ${key}`);
    wrangler(
      ...['r2', 'object', 'put', `${BUCKET}/${key}`, '--remote'],
      ...['--file', join(PUBLIC_DIR, path)],
      ...['--content-type', CONTENT_TYPE],
      ...['--cache-control', CACHE_CONTROL],
    );
    const resp = await fetch(uncached(url));
    if (!resp.ok) throw new Error(`readback GET ${url} -> HTTP ${resp.status}`);
    const body = new Uint8Array(await resp.arrayBuffer());
    if (sha256(body) !== digest) {
      wrangler('r2', 'object', 'delete', `${BUCKET}/${key}`, '--remote');
      throw new Error(`bytes served at ${url} do not match their digest; object deleted`);
    }
    console.log(`verified ${url}`);
  }
}

function prune() {
  if (!existsSync(DIST_DIR)) throw new Error(`${DIST_DIR} missing - build first`);
  for (const { path } of assets()) rmSync(join(DIST_DIR, path));
  console.log(`pruned ${R2_ASSETS.length} R2-hosted payloads from dist/`);
}

const invokedAsCli =
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invokedAsCli) {
  const command = process.argv[2];
  if (command === 'upload') await upload();
  else if (command === 'verify') await verify();
  else if (command === 'prune') prune();
  else if (command === 'manifest') console.log(JSON.stringify(computeManifest(), null, 2));
  else {
    console.error(`usage: r2-assets.mjs upload|verify|prune|manifest`);
    process.exit(1);
  }
}
