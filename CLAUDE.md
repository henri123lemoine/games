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

Keep `cargo fmt` + `cargo clippy --release --all-targets` clean before committing. Evaluation convention: win share against a *field* of opponents, hero rotated through every seat; "fair" is `1/players`. Measure one change at a time (`liars-dice/examples/ab`) — single eval runs can be ~2σ lucky draws.
