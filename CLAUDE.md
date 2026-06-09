# CLAUDE.md

A games lab: algorithms for playing games (CFR variants, belief agents,
Monte-Carlo rollout search, alpha-beta) applied to multiple games through one
shared `Game` trait. (The directory is still named `twentyone` after the
original project; the repo has outgrown that framing.)

## Structure

- **cfr-core/**: the algorithms — `Game` trait, CFR+ (`Solver`, exact, tiny 2p
  games only) + exact best-response exploitability, external-sampling `Mccfr`,
  and the game-agnostic arena (`play_n`, `winrate_vs_field`, `playout_from`).
- **games/liars-dice/**: N-player Liar's Dice (non-standard rules; see its
  README) + belief/rollout agents. The strongest bot is `RolloutAgent`.
- **games/chess/**: chess with perft-validated move generation and an
  alpha-beta search agent.
- **games/twentyone/**: Twenty-One — the engine, the fast decomposed CFR+
  solver (`twentyone::Solver`, the strong way to solve the real game; see its
  BAKEOFF.md), and the `cfr_core::Game` adapter. Pure Rust; the former Python
  harness is deleted (findings preserved in BAKEOFF.md, history in git).

## Workflow

```bash
cargo test --release                                      # everything
cargo run --release -p liars-dice --example play 5 5 6    # play Liar's Dice
cargo run --release -p chess --example play               # play chess
cargo run --release -p twentyone --example play           # play Twenty-One
```

Keep `cargo fmt` + `cargo clippy --release --all-targets` clean before
committing. Evaluation convention: win rates are reported against a *field* of
opponents with the hero rotated through every seat; "fair" is `1/players`.
Measure one change at a time (`liars-dice/examples/ab`) — single eval runs can
be ~2σ lucky draws.
