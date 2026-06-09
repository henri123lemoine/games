# Games lab

Algorithms for playing games — CFR variants, belief agents, Monte-Carlo rollout
search, alpha-beta — applied to multiple games through one shared `Game` trait.
(Started life as a Twenty-One solver; that sub-project now lives in `legacy/`.)

```
cfr-core/            the algorithms — a Game trait, CFR+/MCCFR, exact
                     exploitability, a game-agnostic arena (see cfr-core/README)
games/
  liars-dice/        N-player × D-dice × F-face Liar's Dice + strong belief and
                     Monte-Carlo-rollout agents (see games/liars-dice/README)
  chess/             chess: perft-validated move generation + alpha-beta agent
  twentyone/         Twenty-One expressed against the same Game trait
legacy/              the original Twenty-One sub-project: specialized engine +
                     fast solver, PyO3 bindings, Python RL harness
```

```bash
cargo test --release                                      # everything
cargo run --release -p liars-dice --example play 5 5 6    # play Liar's Dice
cargo run --release -p chess --example play               # play chess
```

The `legacy/` crates are excluded from the Cargo workspace and keep their own
`uv` flow, documented below.

---

# Twenty-One (the original sub-project, in `legacy/`)

A high-performance implementation of the Twenty-One card game designed for
reinforcement learning research.

## Quick Start

```bash
cd legacy/twentyone-rl
uv sync   # builds the Rust-backed `twentyone` extension into the venv

# Watch a game, see convergence, train and evaluate a solver, then play it
uv run examples/basic_play.py
uv run scripts/convergence.py --hearts 1
uv run scripts/train_solver.py --hearts 6 --iters 400000 --chunks 8 --abstract --eval-deals -1
uv run scripts/evaluate.py --solver data/solver_6h.bin --eval-deals -1
uv run scripts/play.py --solver data/solver_6h.bin
```

After editing Rust, force a rebuild with `uv sync --reinstall-package twentyone`
(uv caches the build by version and won't otherwise rebuild on Rust changes).

The solver computes a Nash-equilibrium strategy via external-sampling MCCFR+ in
Rust; see [twentyone-rl/README.md](twentyone-rl/README.md) for the approach.

## Game Rules

Twenty-One is a 2-player card game with a hearts system:

- Each player starts with 6 hearts
- Each round uses cards 1-11 (one of each)
- Players get 2 cards: one face-up (visible), one face-down (hidden)
- Goal: Get closest to 21 without going over
- Round winner deals damage equal to round number
- Game ends when a player reaches 0 hearts

## Architecture

**Performance**: Direct memory access between Python and Rust with zero serialization overhead. Efficient bitmask-based deck representation.

**Modularity**: Clean separation between game logic, bindings, and RL code. Each component can be used independently.

**Developer Experience**: Full type safety with Python type hints and Rust's type system. Deterministic testing with preset deck orders.
