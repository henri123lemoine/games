# CLAUDE.md

A games lab: game-playing algorithms written once against a shared `Game` trait, applied to many games. (The directory is still named `twentyone` after the original project; the repo has outgrown that framing.)

Orient first: [README.md](README.md) has the crate map and quickstart; [ARCHITECTURE.md](ARCHITECTURE.md) is the authoritative design — the layering and capability-trait contract. Read ARCHITECTURE before restructuring anything; those rules are deliberate.

## Workflow

```bash
cargo test --release                       # perft, Kuhn→Nash, invariants, search
cargo run --release -p lab -- list         # what's playable
cargo run --release -p lab -- play chess depth=6
cargo run --release -p lab -- play liars-dice players=5 dice=5
```

The root commands cover the lab workspace and `nn-infer`, but the libtorch training crates under `ml/*` are each their own `[workspace]`, so the root `cargo test`/`clippy` skip them — check those in-tree:

```bash
cd ml/aztrainer && cargo test --release && cargo clippy --release --all-targets   # AlphaZero trainer (tch; CI covers this)
# the slither PPO stack (ml/slither-ppo, ml/slither-rl, ml/slitherinfer, ml/ppo-core) builds the same way, from its own dirs
```

Keep `cargo fmt` + `cargo clippy --release --all-targets` clean before committing. Evaluation convention: win share against a *field* of opponents, hero rotated through every seat; "fair" is `1/players`. Measure one change at a time (`liars-dice/examples/ab`) — single eval runs can be ~2σ lucky draws.
