# CLAUDE.md

A games lab: game-playing algorithms written once against a shared `Game`
trait, applied to many games. Read [ARCHITECTURE.md](ARCHITECTURE.md) before
restructuring anything — the layering and capability-trait contract there are
deliberate. (The directory is still named `twentyone` after the original
project; the repo has outgrown that framing.)

## Structure

- **game-core/**: foundations only — `Game`, `Agent`, `Rng`, the arena, and the
  capability traits (`Eval`, `Determinizer`, `SearchSpec`, `GameUi`). No
  algorithms here, ever.
- **solvers/**: the generic algorithms — `Cfr` (vanilla CFR+, exact
  exploitability; tiny 2p games), `Mccfr`, `AlphaBeta` (needs `Eval`, sharpened
  by `SearchSpec`), `Rollout` (needs `Determinizer`; common-random-numbers
  paired rollouts, rayon-parallel).
- **games/**: rules + knowledge per game. chess (perft-validated; the bot is
  generic `AlphaBeta` + chess's `MaterialEval`/`ChessSpec`), liars-dice
  (non-standard rules — see its README; the bot is generic `Rollout` + the
  belief `ProbabilisticAgent` + `BidConditioned` determinization), twentyone
  (bespoke decomposed CFR+ solver stays game-side by design; see BAKEOFF.md).
- **lab/**: the binding layer — registry (game id + opts + bot → type-erased
  `AnyMatch`) and the one terminal client for all games. A future web server
  reuses exactly these two interfaces; see ARCHITECTURE.md.

## Rules of the design

- Algorithms never live in game crates; game knowledge never lives in solvers.
  New shared knowledge becomes a capability trait in game-core with a default.
- Games depend on `game-core` only (solvers allowed in dev-dependencies).
- New game = `Game` + `GameUi` impl + one registry entry. New algorithm = one
  file in `solvers`. Nothing else changes.

## Workflow

```bash
cargo test --release                       # perft, Kuhn→Nash, invariants, search
cargo run --release -p lab -- list         # what's playable
cargo run --release -p lab -- play chess depth=6
cargo run --release -p lab -- play liars-dice players=5 dice=5
```

Keep `cargo fmt` + `cargo clippy --release --all-targets` clean before
committing. Evaluation convention: win share against a *field* of opponents,
hero rotated through every seat; "fair" is `1/players`. Measure one change at a
time (`liars-dice/examples/ab`) — single eval runs can be ~2σ lucky draws.
