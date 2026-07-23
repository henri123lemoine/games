# The web arcade

The lab compiled to WebAssembly behind one page: pick a game, pick opponents, play the bots or watch them play each other, and run live bot tournaments — all on the visitor's device. Design in [DESIGN.md](DESIGN.md).

```
engine/   wasm-bindgen cdylib over the lab's registry + matches (Rust)
app/      Vite + TypeScript shell, engine Web Worker, per-game frontends
```

## Build & run

```bash
# 1. The engine (requires the wasm32 target: rustup target add wasm32-unknown-unknown)
wasm-pack build web/engine --target web --out-dir pkg

# 2. The app
cd web/app
npm install
npm run dev        # local dev at http://localhost:5173
npm run build      # static site in web/app/dist
npm run preview    # serve the built site
```

Rebuild the engine whenever Rust changes; Vite picks up the new pkg on the next dev reload / build.

## Trained artifacts

Published models live in the arcade-assets R2 bucket ("Heavyweight payloads" below), fetched only when a bot needs them. After retraining (publish from `web/app/`):

```bash
node scripts/r2-assets.mjs publish artifacts/azero-chess.bin data/azero/chess.bin   # chess bot=azero (~22 MB)
# chess bot=azero-gpu (WebGPU; ~6 MB) — export to AZNET1, then check the
# tch forward against nn-infer's torch-free forward. Run from ml/aztrainer/:
DYLD_LIBRARY_PATH=... cargo run --release --bin chess -- export \
    --net <run>/latest.ot --out /tmp/azero-chess.azweb
DYLD_LIBRARY_PATH=... cargo run --release --bin chess -- verify-export \
    --net <run>/latest.ot --out /tmp/azero-chess.azweb
node scripts/r2-assets.mjs publish azero/azero-chess.azweb /tmp/azero-chess.azweb

# four-player chess uses the same AZNET1 parity gate with four value logits
cargo run --release --bin four-player-chess -- export \
    --net ../../runs/four-player-chess/latest.ot \
    --out /tmp/four-player-chess.azweb
cargo run --release --bin four-player-chess -- verify-export \
    --net ../../runs/four-player-chess/latest.ot \
    --out /tmp/four-player-chess.azweb
node scripts/r2-assets.mjs publish azero/four-player-chess.azweb /tmp/four-player-chess.azweb
```

Without a model file, every other bot works; selecting a net bot reports the missing artifact. `/azero-test.html`, `/four-player-chess-azero-test.html`, and `/go-azero-test.html` (also served in the built site) validate the WebGPU kernels against the reference forward, compare the WebGPU and in-wasm CPU forwards head-to-head (the two backends the bot picks between — see below), and print eval throughput — open the relevant page after publishing a new export. The reference end is `nn-infer`'s torch-free `AZNET1` forward; `aztrainer`'s `verify-export` is the parity gate that asserts the tch forward matches it, so CPU ≡ fixtures ≡ GPU stays locked.

### AlphaZero without a GPU

`azero-gpu` evaluates leaves with WebGPU when the browser has it; otherwise the driver hands the same `.azweb` net to the wasm engine, which runs the whole search against `nn-infer`'s reference `AZNET1` forward (`AzGoBot::play_cpu` / `AzChessBot::play_cpu`). The CPU forward is correctness-first, not fast, so the no-GPU path is locked to the trivial visit budget (1 simulation ≈ the network's raw policy move) and the match screen says so. Same net either way — the export parity gate above is what guarantees it.

## Deploying / embedding

`npm run build` produces a fully static site (`web/app/dist`) with relative asset paths (`base: './'`) — host it on any static host (GitHub Pages, Netlify, Vercel, nginx) at any path. To embed in a personal site, either:

- mount `dist/` under a route (e.g. `/arcade/`) and link or iframe it, or
- integrate the source: the app is framework-free; `new App(element).start()` from `src/main.ts` boots into any container element.

Everything runs client-side — no server component, no API keys, no state.

CI automates the personal-site embed: every push to main rebuilds the arcade and publishes `dist/` to the `arcade-dist` branch (single orphan commit). The personal-website repo mounts that branch at `henrilemoine.com/arcade/` on its own deploys — every site push plus a daily freshness cron — with no tokens, since this repo is public (`gh workflow run deploy.yml -R henri123lemoine/personal-website` forces an immediate refresh).

### Heavyweight payloads (R2)

The trained nets and solver tables (~160 MB) live in the `arcade-assets` R2 bucket at `https://arcade-assets.henrilemoine.com/`, not in git. Objects are content-addressed (`<dir>/<stem>.<sha256>.bin` — always `.bin`, so Cloudflare's default cache covers them with no zone configuration; `Cache-Control: immutable`); `web/app/asset-manifest.json` records each payload's logical path and sha256, and every build — dev included — bakes the resulting URLs in via `assetUrl` (`src/assets.ts`). CI fails the build if any manifest entry is missing from the bucket. Deploys are atomic — an old deploy keeps referencing the exact bytes it was built against, nothing is ever overwritten, and an unchanged artifact is never re-uploaded. Anything that needs the bytes on disk (tests, examples, the engine smoke) fetches them through `tools/fetch-asset.sh`, which caches checksum-verified copies in `web/app/.asset-cache/`.

`web/app/scripts/r2-assets.mjs` owns the key scheme and commands. After training or retraining an artifact, publish it and commit the manifest change:

```bash
npx wrangler login   # once
node scripts/r2-assets.mjs publish azero/azero-go.azweb <exported-file>   # upload + checksum readback + manifest update
```

Playing the net/solver games therefore needs network access even in dev. The doom / doom-ai payloads stay in the repo and in `dist/` — their emscripten glue loads iframe-relative.

Cloudflare-side configuration (set up once, 2026-07): the `arcade-assets` bucket with `arcade-assets.henrilemoine.com` as its custom domain, and bucket CORS allowing `GET`/`HEAD` from any origin (the objects are public and immutable; localhost previews and the Playwright harnesses fetch cross-origin from random ports). No zone configuration — the uniform `.bin` key suffix is what makes the edge cache apply by default.

## Performance notes

- The engine runs single-threaded inside a Web Worker (the UI never blocks). Browser-tuned defaults live in the shell (`DEFAULT_OPTS`); raise rollouts / sims / depth in the setup screen on fast machines.
- Tournaments parallelize across a pool of workers (one wasm instance per core) — that is where multi-core shows up today.
- Upgrade path for in-match parallelism: `wasm-bindgen-rayon`, which requires the host to serve COOP/COEP headers (cross-origin isolation). The solvers already gate rayon behind the `parallel` feature, so this is wiring, not redesign.

## Adding a game's frontend

The Rust recipe ([../ARCHITECTURE.md](../ARCHITECTURE.md)) plus one JSON method (`view_data`) makes a game playable here via the generic fallback frontend; [DESIGN.md](DESIGN.md) is the full contract. The polished frontend is one folder:

1. `web/app/src/frontends/<id>/index.ts` implementing `GameFrontend` (`src/frontends/types.ts`) against the game's own `view_data` schema.
2. Register it in `web/app/src/frontends/index.ts`.

The shell, engine, and other games do not change.
