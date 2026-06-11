# game-core — the foundations

The bottom layer of the games lab: the `Game` trait, the `Agent` interface,
the capability traits (`Eval`, `Determinizer`, `SearchSpec`,
`PolicyValueEncoder`, `GameUi`), the arena (`play`, `win_rate`,
`winrate_vs_field`), a reproducible `Rng`, and small shared utilities
(hashing, JSON escaping, distribution sampling, summary stats).

This crate deliberately contains **no algorithms** and depends on nothing.
Algorithms live in `solvers/`, written once against these interfaces; games
live in `games/*` and provide rules plus declared knowledge. The layering
contract — who may depend on whom, and why actions are indices and
information sets are u64 keys — is documented in
[ARCHITECTURE.md](../ARCHITECTURE.md) at the repo root.

```bash
cargo test -p game-core
```
