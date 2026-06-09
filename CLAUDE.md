# CLAUDE.md

A games lab: algorithms for playing games (CFR variants, belief agents,
Monte-Carlo rollout search, alpha-beta) applied to multiple games through one
shared `Game` trait. (The directory is still named `twentyone` after the
original sub-project; the repo has outgrown that framing.)

## Structure

- **cfr-core/**: the algorithms — `Game` trait, CFR+ (`Solver`, exact, tiny 2p
  games only) + exact best-response exploitability, external-sampling `Mccfr`,
  and the game-agnostic arena (`play_n`, `winrate_vs_field`, `playout_from`).
- **games/liars-dice/**: N-player Liar's Dice (non-standard rules; see its
  README) + belief/rollout agents. The strongest bot is `RolloutAgent`.
- **games/chess/**: chess with perft-validated move generation and an
  alpha-beta search agent.
- **games/twentyone/**: Twenty-One wrapped as a `Game` (thin; wraps the legacy
  engine).
- **legacy/**: the original Twenty-One sub-project — `twentyone-core` (engine +
  fast decomposed CFR solver), `twentyone-py` (PyO3 bindings), `twentyone-rl`
  (Python RL harness). Self-contained, excluded from the Cargo workspace, has
  its own CLAUDE.md files.

## Workflow

```bash
cargo test --release                                  # everything
cargo run --release -p liars-dice --example play 5 5 6   # play Liar's Dice
cargo run --release -p chess --example play               # play chess

# legacy Twenty-One (Python):
cd legacy/twentyone-rl && uv sync
# after editing legacy Rust: uv sync --reinstall-package twentyone
```

Keep `cargo fmt` + `cargo clippy --release --all-targets` clean before
committing. Evaluation convention: win rates are reported against a *field* of
opponents with the hero rotated through every seat; "fair" is `1/players`.
