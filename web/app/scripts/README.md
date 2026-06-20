# Browser validation harness

`browser-validate.mjs` launches a **real headless Chromium with WebGPU**, serves
the built app over `http://localhost`, drives one game, and reports back the
backend that actually ran (GPU vs CPU), real ms/move, every console log, the
page's `navigator.gpu` adapter info, and a screenshot. It exists so browser /
GPU / visual behavior can be validated without a human reloading localhost.

```bash
npm run validate:browser            # builds dist/, validates snake (default)
npm run validate:browser -- snake   # snake card, two AlphaZero bots (watch mode)
npm run validate:browser -- slither # slither standalone

# Without rebuilding (faster; uses the existing dist/):
node scripts/browser-validate.mjs snake
node scripts/browser-validate.mjs snake --headed   # show the window
```

Outputs land in `web/app/.validation/<game>.png` and `<game>.json` (gitignored).

## Behavioral test: does the slither bot punish basic play?

`slither-behavior.mjs` is a deeper, behavioral check (not just config/visual). It
drives the slither PLAYER worm with deliberately **naive** mouse steering (slow
circles / straight back-and-forth — no boosting, no threat-avoidance, no
food-seeking) for a meaningful duration, then reports whether the trained bots
hunt and kill that basic player or get out-grown by it.

```bash
npm run validate:slither                       # ~90s circle run, headless
node scripts/slither-behavior.mjs --secs 120 --pattern lines
node scripts/slither-behavior.mjs --pattern wander --headed
```

It reads metrics without touching game code: the slither HUD paints the
leaderboard / length / rank straight to the canvas, so the harness wraps
`CanvasRenderingContext2D.fillText` (via `addInitScript`) to scrape those each
frame into `window.__slrHud`, and reads the "You died" panel from the DOM. It
reports player death count + when, player vs bot max length, the final
leaderboard, and timed screenshots (`.validation/slither-behavior-*.png`).

## Two gotchas this harness encodes

- **Use the full Chromium, not the headless shell.** Playwright's
  `headless: true` swaps in `chrome-headless-shell`, a stripped binary that ships
  **no WebGPU** (`navigator.gpu` is `undefined`, and it falls back to
  SwiftShader). The harness launches the full Chromium with `--headless=new`
  instead — windowless, but with a real Metal adapter (`vendor: apple,
  architecture: metal-3`).
- **`navigator.gpu` needs a secure context.** It is absent on `file://` and
  `data:` URLs. The harness serves over `http://localhost`, which counts as
  secure, so the adapter is exposed.

Launch flags that make headless WebGPU work on this Mac:
`--headless=new --enable-unsafe-webgpu --enable-features=Vulkan --use-angle=metal --ignore-gpu-blocklist`.

## Adding a game

Add an entry to the `GAMES` map in `browser-validate.mjs`: the hash `route` to
load (use `?mode=watch` so both seats are bots and moves auto-flow with no human
input), the `ready` selector to wait on, and — if the game has a debug HUD — the
`overlaySelector` to read backend/ms/move from.

The snake route is `/?snakeDebug#/g/snake?mode=watch`: `?snakeDebug` lives in the
URL **search** (read by `debugEnabled()`), while the game itself is reached via
the **hash** route. A bare `localhost:5173/?snakeDebug` shows only the home
screen and never mounts snake — which is why that URL appeared to "do nothing".
