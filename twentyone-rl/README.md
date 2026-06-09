# Twenty-One RL

Game-theoretic solving and evaluation for the Twenty-One card game, built on the
Rust `twentyone` environment as the single source of truth.

## Approach

Twenty-One is a two-player, zero-sum, imperfect-information game, so the right
notion of "optimal" is a **Nash equilibrium**. We compute one with
**external-sampling MCCFR+** (Monte-Carlo counterfactual regret minimization with
CFR+ refinements), implemented in Rust (`twentyone_core::Solver`) and exposed to
Python as `twentyone.Solver`.

Two design choices make this tractable and exact:

- **Lossless information sets.** Each decision is keyed by a sufficient statistic
  — `(round, my_hearts, opp_hearts, my_total, unseen_card_set, my_stood,
  opp_stood)` — that collapses raw histories without discarding any
  strategically relevant information. No hand-crafted abstraction.
- **Subgame decomposition.** Rounds are linked only by the public
  `(hearts0, hearts1, round)` carry-over (the deck is reshuffled each round with
  no hidden state crossing the boundary), so the game is solved by backward
  induction over that lattice, restricted to reachable carry-overs.

Quality is measured by **exact best-response exploitability** (NashConv / 2):
the solver walks the full game tree, enumerating every chance event, to compute
how much a perfect counter-strategy could gain. It is zero exactly at a Nash
equilibrium. Utilities are in `[-1, 1]` (win / loss).

## Setup

```bash
# Build and install the Rust-backed environment + solver into this venv
cd ../twentyone-py
maturin build --release -i ../twentyone-rl/.venv/bin/python3
cd ../twentyone-rl
uv pip install --no-cache --no-deps ../twentyone-py/target/wheels/twentyone-*.whl
```

## Demonstrating optimality (small variants)

On variants with few starting hearts the best response can be enumerated
exactly, so convergence to Nash is directly measurable. Exploitability falls
monotonically toward zero:

```bash
uv run scripts/convergence.py --hearts 1 --iters 1000 10000 100000 1000000
```

## Training and evaluating the full game

```bash
# Train a 6-heart (full game) strategy; checkpoints to data/solver_6h.bin
uv run scripts/train_solver.py --hearts 6 --iters 300000 --chunks 12 --eval-deals -1

# Win rates vs. baselines, with 95% Wilson confidence intervals
uv run scripts/evaluate.py --solver data/solver_6h.bin --games 2000 --eval-deals -1
```

`evaluate.py` plays the solver against a uniform-random agent and a ladder of
threshold heuristics ("draw while total < t"), alternating seats to cancel
first-player bias.

## Watching a game

```bash
uv run examples/basic_play.py
```

## Layout

- `scripts/agents.py` — `RandomAgent`, `ThresholdAgent`, `SolverAgent` (a uniform
  `act(env, player)` interface; the solver indexes the env's `sufficient_key`).
- `scripts/arena.py` — head-to-head matches with Wilson confidence intervals.
- `scripts/train_solver.py` — train and checkpoint a solver.
- `scripts/evaluate.py` — exploitability + win-rate report.
- `scripts/convergence.py` — exact-exploitability convergence on small variants.
- `src/twentyone_rl/display.py` — human-readable game rendering.
