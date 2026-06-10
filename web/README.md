# The web arcade

The lab compiled to WebAssembly behind one page: pick a game, pick opponents,
play the bots or watch them play each other, and run live bot tournaments —
all on the visitor's device. Design in [../WEB.md](../WEB.md).

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

Rebuild the engine whenever Rust changes; Vite picks up the new pkg on the
next dev reload / build.

## Optional artifacts

Trained models ship as static assets, fetched only when a bot needs them:

```bash
cp data/azero/chess.bin web/app/public/artifacts/azero-chess.bin   # chess bot=azero (~22 MB)
```

Without the file, every other bot works; selecting `bot=azero` reports the
missing artifact.

## Deploying / embedding

`npm run build` produces a fully static site (`web/app/dist`) with relative
asset paths (`base: './'`) — host it on any static host (GitHub Pages,
Netlify, Vercel, nginx) at any path. To embed in a personal site, either:

- mount `dist/` under a route (e.g. `/arcade/`) and link or iframe it, or
- integrate the source: the app is framework-free; `new App(element).start()`
  from `src/main.ts` boots into any container element.

Everything runs client-side — no server component, no API keys, no state.

## Performance notes

- The engine runs single-threaded inside a Web Worker (the UI never blocks).
  Browser-tuned defaults live in the shell (`WEB_DEFAULTS`); raise rollouts /
  sims / depth in the setup screen on fast machines.
- Tournaments parallelize across a pool of workers (one wasm instance per
  core) — that is where multi-core shows up today.
- Upgrade path for in-match parallelism: `wasm-bindgen-rayon`, which requires
  the host to serve COOP/COEP headers (cross-origin isolation). The solvers
  already gate rayon behind the `parallel` feature, so this is wiring, not
  redesign.

## Adding a game's frontend

The Rust recipe (ARCHITECTURE.md) plus one JSON method makes a game playable
here via the generic fallback frontend. The polished frontend is one folder:

1. `web/app/src/frontends/<id>/index.ts` implementing `GameFrontend`
   (`src/frontends/types.ts`) against the game's own `view_data` schema.
2. Register it in `web/app/src/frontends/index.ts`.

The shell, engine, and other games do not change.
