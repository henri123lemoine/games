# DOOM browser conformance harness

Objective, reproducible tests for the arcade's playable browser DOOM
(`web/app/public/doom/`), built because the engine is **deterministic**: the same
inputs + RNG produce the same frames, so we can differentially test our build and
measure rendering bugs instead of eyeballing them.

The browser DOOM is **cloudflare/doom-wasm** — Chocolate Doom compiled to
WebAssembly via Emscripten. It is _not_ the `doomgeneric` substrate used by the
RL work in `doomrl/`; it is a different engine. The reference is therefore native
Chocolate-Doom behaviour, which this harness approximates by running the same
WASM engine under its **software renderer** (whose presentation is a full
`putImageData` of the engine's `I_VideoBuffer` every frame — i.e. exactly what
real DOOM shows) and diffing it against the hardware (WebGL) path.

## Setup

```bash
cd web/app
bash scripts/doom-conformance/setup.sh   # copies engine+WAD, extracts demo lumps
```

`setup.sh` reconstructs the sandbox from `public/doom/` (engine, WAD) and extracts
the WAD's built-in deterministic demos (`DEMO1..3`) to `.lmp` files. The large /
derivable assets are git-ignored; only the loader pages and configs are committed.

## What it measures

| script | question | decisive signal |
|---|---|---|
| `validate-production.mjs` | does the **shipping** `doom.html` pass? | software renderer on, mouse-look stable, live frame |
| `probe.mjs <page>` | renderer + idle smear | `glContextRequested`, `CreateUpscaledTexture 0x0` |
| `probe-mouse.mjs <page>` | does mouse-only input move the view? | `fracViewChanged` (should be ~0) |
| `differential.mjs <sub> <ref>` | per-frame divergence on the same demo | side-by-side + diff PNGs |
| `capture-browser.mjs <out>` | dump per-frame canvas PNGs during scripted input | visual trace |
| `demo-timing.mjs <page>` | when does the built-in demo render? | mean/motion trace |

Loader pages in `sandbox/`:
`doom-baseline.html` (GL, pre-fix) · `doom-soft.html` (software, fixed) and their
`-playdemo demo1` twins `doom-demo-gl.html` / `doom-demo-soft.html`.

```bash
node scripts/doom-conformance/validate-production.mjs        # the gate (exit 0 = pass)
node scripts/doom-conformance/probe.mjs doom-baseline.html baseline
node scripts/doom-conformance/probe-mouse.mjs doom-soft.html soft
node scripts/doom-conformance/differential.mjs doom-demo-gl.html doom-demo-soft.html
```

Outputs (PNGs + report.json) land under `scripts/doom-conformance/out/` (ignored).

## What was wrong, and the fix

Two independent bugs, both diagnosed objectively here:

1. **Smearing / trails when moving.** The WebGL (GLES2) renderer in this WASM
   build queries `max texture size` as **0×0** at init
   (`CreateUpscaledTexture: ... max texture size 0x0`), so the upscaled screen
   texture is never created and the GL framebuffer is never cleanly repainted —
   stale geometry persists (the head-to-head demo frames in
   `out/diff/.../` show ghosted enemies / streaked walls under GL, crisp under
   software). A prior fix put `force_software_renderer 1` in **`default.cfg`**,
   but Chocolate Doom reads that variable from its **extra** config
   (`websockets-doom.cfg`), so it never took effect — `glContextRequested`
   stayed `true`. Fix: ship `websockets-doom.cfg` with `force_software_renderer 1`
   and load it via `-extraconfig` from the engine's config dir. The software path
   blits a full surface every frame and cannot smear (`glContextRequested:false`).

2. **"Acts strange when looking around."** The shipped config had `use_mouse 1`
   with `grabmouse 0`: ungrabbed relative mouse motion turned the player, so the
   view lurched whenever the cursor moved (mouse-only input changed **23%** of the
   viewport). Fix: `use_mouse 0` (keyboard-only browser play) plus `novert 1` in
   the extra config — mouse-only input now changes **0%** of the view.

The `CreateUpscaledTexture 0x0` line still appears with the software renderer but
is moot there (the GL upscale texture is unused; presentation is `putImageData`),
so the conformance gate only fails on it when the GL renderer is actually active.
