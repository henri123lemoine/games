// The heavyweight immutable payloads (trained nets, solver tables) live in
// the `arcade-assets` R2 bucket, not in git. Objects are content-addressed
// (`<dir>/<stem>.<sha256><ext>`), so a URL never changes meaning: old deploys
// keep resolving the bytes they were built against, new uploads never
// overwrite anything, and everything is safe to cache forever
// (`Cache-Control: immutable`).
//
// `asset-manifest.json` (checked in) is the repo's record of each payload:
// its logical path and the sha256 of its bytes. `vite.config.ts` bakes the
// resulting URLs into every build via `computeManifest()`, CI's `verify`
// gates deploys on the objects actually existing, and `publish` is how a new
// or retrained artifact gets in: upload, checksum readback, manifest update.
//
//   node scripts/r2-assets.mjs publish <logical-path> <file>  # e.g. publish azero/azero-go.azweb runs/go/export.azweb
//   node scripts/r2-assets.mjs verify                         # all URLs live? (CI gate, no auth)
//   node scripts/r2-assets.mjs manifest                       # print the path -> URL map

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { basename, dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const APP_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MANIFEST_PATH = join(APP_DIR, 'asset-manifest.json');

const ASSET_HOST = 'https://arcade-assets.henrilemoine.com';
const BUCKET = 'arcade-assets';
const CACHE_CONTROL = 'public, max-age=31536000, immutable';
const CONTENT_TYPE = 'application/octet-stream';

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function urlFor(path, digest) {
  const ext = extname(path);
  return `${ASSET_HOST}/${dirname(path)}/${basename(path, ext)}.${digest}${ext}`;
}

function readManifest() {
  return JSON.parse(readFileSync(MANIFEST_PATH, 'utf8'));
}

export function computeManifest() {
  return Object.fromEntries(
    Object.entries(readManifest()).map(([path, digest]) => [path, urlFor(path, digest)]),
  );
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
  for (const [path, url] of Object.entries(computeManifest())) {
    if (await exists(url)) console.log(`ok      ${path}`);
    else {
      console.error(`MISSING ${path} -> ${url}`);
      missing.push(path);
    }
  }
  if (missing.length > 0) {
    console.error(
      `\n${missing.length} asset(s) not in the bucket. Publish each with\n` +
        `  node scripts/r2-assets.mjs publish <logical-path> <file>\n` +
        `(needs \`npx wrangler login\` on a machine with the files).`,
    );
    process.exit(1);
  }
}

const WRANGLER = (process.env.WRANGLER ?? 'npx wrangler@4').split(' ');

function wrangler(...args) {
  execFileSync(WRANGLER[0], [...WRANGLER.slice(1), ...args], { stdio: 'inherit' });
}

async function publish(path, file) {
  const digest = sha256(readFileSync(file));
  const url = urlFor(path, digest);
  const key = url.slice(`${ASSET_HOST}/`.length);
  if (await exists(url)) {
    console.log(`exists  ${url}`);
  } else {
    console.log(`upload  ${file} -> ${key}`);
    wrangler(
      ...['r2', 'object', 'put', `${BUCKET}/${key}`, '--remote'],
      ...['--file', file],
      ...['--content-type', CONTENT_TYPE],
      ...['--cache-control', CACHE_CONTROL],
    );
    const resp = await fetch(uncached(url));
    if (!resp.ok) throw new Error(`readback GET ${url} -> HTTP ${resp.status}`);
    if (sha256(new Uint8Array(await resp.arrayBuffer())) !== digest) {
      wrangler('r2', 'object', 'delete', `${BUCKET}/${key}`, '--remote');
      throw new Error(`bytes served at ${url} do not match their digest; object deleted`);
    }
    console.log(`verified ${url}`);
  }
  const manifest = readManifest();
  if (manifest[path] === digest) {
    console.log('manifest already up to date');
    return;
  }
  const isNew = !(path in manifest);
  manifest[path] = digest;
  const sorted = Object.fromEntries(Object.entries(manifest).sort(([a], [b]) => a.localeCompare(b)));
  writeFileSync(MANIFEST_PATH, JSON.stringify(sorted, null, 2) + '\n');
  console.log(
    `asset-manifest.json updated (${isNew ? 'new asset — wire its fetch through assetUrl' : 'digest bumped'}); commit it`,
  );
}

const invokedAsCli =
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invokedAsCli) {
  const [command, ...args] = process.argv.slice(2);
  if (command === 'publish' && args.length === 2) await publish(args[0], args[1]);
  else if (command === 'verify') await verify();
  else if (command === 'manifest') console.log(JSON.stringify(computeManifest(), null, 2));
  else {
    console.error('usage: r2-assets.mjs publish <logical-path> <file> | verify | manifest');
    process.exit(1);
  }
}
