# The arcade: web design

One page on a personal website where visitors pick a game, pick opponents, and
play against the lab's bots — or watch bots play each other and run live
tournaments. **Everything runs on the visitor's device**: the Rust workspace
(rules, search, learned policies) compiles to WebAssembly; there is no game
server. Each game gets its own polished, animated frontend with its own visual
identity — the page should read like a team of game developers each shipped
their game, not like one engine wearing eight skins.

This document is the contract for that build. It extends
[ARCHITECTURE.md](ARCHITECTURE.md); the dependency rules there still hold.

## Why this is mostly already built

The lab's two serving interfaces were designed for this day:

- **the registry** (`lab/src/registry.rs`): `game id + opts + bot specs →
  type-erased match` — the catalog.
- **`AnyMatch`** (`lab/src/runner.rs`): a uniform, viewer-scoped match surface
  (`advance / view / legal actions / apply input / result`), already hiding
  private information per seat.

The web build wraps these in wasm instead of a terminal loop. No algorithm, no
game rule, and no registry logic is rewritten.

## The three contracts

Extensibility comes from three small interfaces, one per layer. A game touches
all three; nothing else in the system changes when a game is added.

```
 Rust (per game)            wasm boundary (shared)         TypeScript (per game)
┌──────────────────┐      ┌───────────────────────┐      ┌─────────────────────┐
│ Game + GameUi    │      │ Engine                │      │ GameFrontend        │
│ + view_data():   │ ───▶ │  listGames() manifest │ ───▶ │  render(view)       │
│   game-private   │      │  createMatch(...)     │      │  animate(event)     │
│   JSON view      │      │  Match: step/view/    │      │  pickAction(legal)  │
│ (registry entry) │      │   legal/apply/result  │      │ (frontend manifest) │
└──────────────────┘      └───────────────────────┘      └─────────────────────┘
        runs in a Web Worker ────────────┘                 runs on the main thread
```

**Key identity choice: per-game view schemas are game-private.** The shared
layers (engine API, worker protocol, shell) treat views and transition events
as opaque JSON strings. The chess crate emits `{fen, lastMove, ...}` and the
chess frontend understands it; the dice crate emits hands/bids and the dice
frontend understands those. The schema is a private contract between a game's
Rust crate and its frontend package — exactly like `Eval` is a private
contract between a game and the search that consumes it. This is what lets
each frontend be genuinely independent (own layout, own animation system, own
aesthetic) without the shell knowing anything.

### 1. Rust: structured views and events

Two additions, both backwards-compatible:

- `GameUi::view_data(&State, viewer) -> Option<String>` (default `None`): the
  viewer-scoped state as JSON. Games keep `render()` for the terminal.
- `AnyMatch` events become structured: `advance()` returns
  `MatchEvent { seat, label, text, data: Option<String> }` instead of bare
  strings, where `data` is game-private transition JSON (what moved where, what
  was revealed) for animation. The terminal client prints `.text` and is
  otherwise unchanged. `advance()` also gains a one-step variant so spectator
  mode can animate move by move instead of receiving the whole game at once.

`game-core` stays dependency-free: `view_data`/`data` are `String`-typed;
games that emit JSON depend on `serde_json` themselves (or `format!` it).

`lab` splits into lib + bin: `lab/src/lib.rs` exports `registry`, `runner`,
`compare`; `main.rs` becomes a thin terminal client over the lib. The wasm
crate consumes the lib — the terminal and the web are now literally two
frontends over the same code, which was the promise all along.

### 2. The wasm boundary (`web/engine`)

A `cdylib` crate (`wasm-bindgen`) in the workspace, exposing:

```ts
listGames(): Manifest            // [{id, name, players, optsSchema, bots: [{id, summary, optsSchema}]}]
createMatch(req: string): Match  // {game, opts, seats: [bot-spec | "human"], seed}
Match.step(): string             // one event (chance folded in), or "" at human turn / game over
Match.view(seat): string         // game-private JSON, viewer-scoped
Match.legalActions(): string     // [{index, label}]
Match.applyHuman(input): string  // event, or throws with a re-prompt message
Match.isOver/resultText/turn()
loadArtifact(id, bytes)          // trained nets / solver tables, fetched as static assets
runPairs(spec): string           // paired bot-vs-bot games + Elo/SPRT verdicts (reuses game-core::stats)
```

Seeds come from JS; matches stay reproducible (shareable replays for free).
Trained artifacts (`data/azero/chess.bin`, Twenty-One tables) ship as static
files and load via `loadArtifact` — no train-at-startup in the browser.

Two portability fixes, both mechanical:

- **rayon** behind a `parallel` cargo feature in `solvers` (default on;
  off for wasm — `Rollout` gets a sequential path over the same chunks).
- **`Instant`** in alpha-beta's time budget → the `web-time` crate (drop-in
  `Instant` that works on wasm; zero code churn beyond the import).

Phase 2 (optional): `wasm-bindgen-rayon` restores real parallelism in the
browser. It requires cross-origin isolation headers (COOP/COEP) from the host,
so it is an upgrade, not a prerequisite — single-threaded release wasm is
already strong (the dice bot at 1–2k rollouts, alpha-beta at depth 5–6 with a
time budget).

### 3. TypeScript: the shell and per-game frontends

```
web/app/
  src/shell/          game picker · seat/bot config · match screen ·
                      spectate pacing · tournament screen (live Elo table)
  src/engine/         worker host: spawns the wasm worker, typed protocol
  src/frontends/
    _generic/         fallback: renders text view + action buttons
    chess/            SVG board, drag-to-move, piece animation
    liars-dice/       dice cups, bid ladder, reveal choreography
    connect4/ othello/ go/ g2048/ snake/ twentyone/
```

Each frontend is a framework-free TS package implementing one interface:

```ts
interface GameFrontend {
  gameId: string
  mount(host: HTMLElement, ctx: FrontendCtx): void   // ctx: submit(action), sounds, prefers-reduced-motion
  render(view: GameView): void                       // full redraw from game-private JSON
  animate(event: MatchEvent): Promise<void>          // resolves when the animation completes
  promptAction(legal: ActionInfo[]): void            // enable board-native input (click a square, drag a die)
  unmount(): void
}
```

The shell owns the match loop and *awaits* `animate()` between events — pacing
for spectating falls out of the same mechanism. Frontends register in a
manifest keyed by game id; **a game with no custom frontend automatically gets
`_generic`**, so a new Rust game is playable in the browser the moment its
registry entry exists, and the polished frontend can come later — same
default-capability philosophy as `NoSpec`/`Identity` in game-core.

The shell is the only place with a framework (Svelte or React — small, one
screen); frontends stay vanilla TS + DOM/canvas/SVG so each can have its own
visual identity and no shared styling constraints beyond a theme token file
(dark/light, spacing) for page coherence.

### Workers

The engine lives in a **Web Worker**: bot search (seconds of CPU) never
freezes the page. The protocol is the engine API verbatim plus a
`thinking(seat)` notification. Tournaments run matches in a small worker pool
(`navigator.hardwareConcurrency`-capped), streaming results to a live
standings table; the Elo/SPRT math is the existing `game-core::stats` compiled
into the same wasm.

## Adding a game (the acid test, extended)

1. Rust: `Game` + `GameUi` (+ `view_data`) + registry entry — *exactly today's
   recipe plus one JSON method*. The game is now playable on the web via the
   generic frontend, and joins bot-vs-bot/tournaments automatically.
2. Frontend: one folder in `web/app/src/frontends/<id>/` implementing
   `GameFrontend` against the game's own view schema, registered in the
   frontend manifest.

Nothing else changes — not the shell, not the engine crate, not other games.
Adding an *algorithm* still touches only `solvers` + registry bot specs, and
every game's web page picks it up as a selectable bot.

## Build phases

1. **Rust groundwork**: lab lib/bin split; `MatchEvent` + `view_data`;
   `parallel` feature + `web-time`; `web/engine` with the API above; artifact
   loading. Gate: terminal client byte-identical behavior, all tests green.
2. **Shell + generic frontend**: Vite app, worker host, game/bot pickers,
   match screen. Gate: all eight games playable in a browser (ugly but
   working), dice bot strength verified in wasm.
3. **Polished frontends**, roughly in order of payoff: liars-dice (the
   flagship), chess, connect4, othello, go, 2048, snake, twentyone. Each is an
   independent package — this phase parallelizes perfectly across agents.
4. **Spectate + tournaments**: pacing controls, live Elo table, match URLs
   with seeds (shareable replays).
5. **Site integration + performance**: embed as one page/ES module in the
   personal site; wasm size pass (`wasm-opt`, feature-trim); optional
   `wasm-bindgen-rayon` if the host allows COOP/COEP headers.

## Open questions for the site owner

- What is the personal website built with, and where is it hosted? This
  decides the embed form (ES module vs iframe vs route in the site's own
  framework) and whether COOP/COEP headers (browser threading) are available.
- Visual direction: one shared dark "arcade" shell with per-game identity
  inside the board area is the default assumption.
