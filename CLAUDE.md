# CLAUDE.md

A games lab: game-playing algorithms written once against a shared `Game` trait, applied to many games. (The directory is still named `twentyone` after the original project; the repo has outgrown that framing.)

Orient first: [README.md](README.md) has the crate map and quickstart; [ARCHITECTURE.md](ARCHITECTURE.md) is the authoritative design — the layering and capability-trait contract. Read ARCHITECTURE before restructuring anything; those rules are deliberate.

## Discipline

**No legacy code.** When a change supersedes existing code, delete the old code — no backward-compat shims, dual APIs, deprecated paths, or dead branches kept "just in case" or to avoid editing call sites. A new function that replaces an old one updates every call site and removes the old one. One way to do each thing.

**Ship the whole agreed scope.** When we agree to build A, B, C, D, E, build all of A–E in one pass. Never deliver A–D and then surface E as a "next step," an "optional" extra, or a question ("want me to do E?"). A working subset is not "done," and a comment noting the gap does not make the gap agreed. If scope genuinely needs to shrink, say so explicitly *before* cutting — never ship a subset and reframe the remainder as optional.

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
