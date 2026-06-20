# DOOM (vendored)

A self-contained browser port of id Software's DOOM, mounted by the arcade as a
standalone page (`doom.html`). It is not a `Game`-trait game: DOOM is a
real-time engine, so it runs entirely client-side outside the wasm match engine
— the arcade shell links a card to `doom.html`, which boots the port and the
shareware WAD.

## Contents

- `websockets-doom.js`, `websockets-doom.wasm` — the engine, Chocolate Doom
  compiled to WebAssembly via Emscripten.
- `doom1.wad` — the DOOM shareware IWAD (episode 1, "Knee-Deep in the Dead").
- `default.cfg` — key bindings and engine defaults (`startup_delay 0`).
- `doom.html` — the loader page (title overlay → boots the engine on click).
- `COPYING.md` — GPLv2, the engine's license.

## Provenance

- **Engine:** [cloudflare/doom-wasm](https://github.com/cloudflare/doom-wasm)
  @ `65e0d3ae2ffa604155eebd96ed40da6567bd08f4`, a WebAssembly port of
  [Chocolate Doom](https://github.com/chocolate-doom/chocolate-doom).
  Authors: Simon Howard, James Haley, Samuel Villarreal, Fabian Greffrath,
  Jonathan Dowland, Alexey Khokholov; wasm/WebSockets port by Cloudflare.
  Built locally with Emscripten 6.0.0 (`emcc`). Two source patches were applied
  so the RAF-driven main loop never calls `emscripten_sleep` from inside its own
  callback (which deadlocks on modern Emscripten): `src/d_loop.c` (`TryRunTics`
  returns instead of sleeping when no tic is ready) and `src/doom/d_main.c`
  (`D_RunFrame`'s wipe wait returns instead of busy-sleeping), both guarded by
  `#ifdef __EMSCRIPTEN__`. The `configure.ac` Emscripten link flags were
  modernised for Emscripten 6 (`EXPORTED_RUNTIME_METHODS`, memory growth).

- **WAD:** the freely redistributable DOOM shareware IWAD, `DOOM1.WAD` v1.9.
  MD5 `f0cefca49926d00903cf57551d901abe`, 4,196,020 bytes.

## Licenses

- The engine (Chocolate Doom + this port) is GPLv2 — see `COPYING.md`.
- `DOOM1.WAD` is id Software's shareware data, redistributable under the
  original shareware terms (the full game data is not included; this is the
  shareware episode only).

## Audio

Music is disabled (`-nomusic`) because the in-engine music synth init can hang
on environments without an audio device; sound effects remain enabled.
