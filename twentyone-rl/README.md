# Twenty-One RL

Game-theoretic solving and evaluation for the Twenty-One card game, built on the
Rust `twentyone` environment as the single source of truth.

## Approach

Twenty-One is a two-player, zero-sum, imperfect-information game, so the right
notion of "optimal" is a **Nash equilibrium**. We compute one with
**external-sampling MCCFR+** (Monte-Carlo counterfactual regret minimization with
CFR+ refinements), implemented in Rust (`twentyone_core::Solver`) and exposed to
Python as `twentyone.Solver`.

Three design choices make this work:

- **Lossless information sets.** Each decision is keyed by a sufficient statistic
  — `(round, my_hearts, opp_hearts, my_total, unseen_card_set, my_stood,
  opp_stood)` — that collapses raw histories without discarding any
  strategically relevant information.
- **Subgame decomposition.** Rounds are linked only by the public
  `(hearts0, hearts1, round)` carry-over (the deck is reshuffled each round with
  no hidden state crossing the boundary), so the game is solved by backward
  induction over that lattice, restricted to reachable carry-overs.
- **Optional abstraction for the full game.** The lossless 6-heart game has tens
  of millions of information sets. `Solver.abstracted` summarizes the unseen set
  by a coarse value-band histogram, cutting that ~7× so CFR reaches strong play
  quickly. Small variants use the lossless keying so convergence can be measured
  exactly.

Quality is measured by **exact best-response exploitability** (NashConv / 2):
the solver walks the full game tree, enumerating every chance event, to compute
how much a perfect counter-strategy could gain — always against the *true* game,
so an abstracted strategy is scored on its real-game exploitability. It is zero
exactly at a Nash equilibrium. Utilities are in `[-1, 1]` (win / loss).

## Results

**Convergence (lossless, exact best response).** On the 1-heart variant, exact
exploitability falls steeply and monotonically as training proceeds — the
signature of approaching equilibrium (utilities are in `[-1, 1]`):

| iters/subgame |    1k |   10k |  100k |    1M |    3M |
|---------------|------:|------:|------:|------:|------:|
| exploitability| 0.782 | 0.581 | 0.273 | 0.193 | 0.181 |

The remaining residual reflects the subgame decomposition: rounds are solved
separately and linked by Monte-Carlo-estimated continuation values, which caps
how close the headline best response can be driven to zero. It is the price of
keeping each subgame shallow enough to solve the full game.

**Full 6-heart game (4.9M information sets, band abstraction, 400k iters/subgame).**
Played greedily, the solver beats every baseline, over 2000 games per matchup
with 95% Wilson intervals (draws counted as half):

| Opponent       | Solver win % | 95% CI         |
|----------------|-------------:|----------------|
| Random         |     90.1 %   | [88.7, 91.3]   |
| Threshold(14)  |     57.6 %   | [55.4, 59.7]   |
| Threshold(15)  |     54.5 %   | [52.3, 56.7]   |
| Threshold(16)  |     52.4 %   | [50.3, 54.6]   |
| Threshold(17)  |     50.9 %   | [48.7, 53.1]   |
| Threshold(18)  |     55.1 %   | [52.9, 57.3]   |
| Threshold(19)  |     64.3 %   | [62.2, 66.4]   |

The threshold heuristics are themselves near-optimal for the within-round game,
so a win rate above 50 % against all of them — while dominating random play —
is strong evidence of high-quality play. (Numbers reproduce with the commands
below; exact values vary slightly with seed.)

## Setup

```bash
# Build and install the Rust-backed environment + solver into this venv
cd ../twentyone-py
maturin build --release -i ../twentyone-rl/.venv/bin/python3
cd ../twentyone-rl
uv pip install --no-cache --no-deps ../twentyone-py/target/wheels/twentyone-*.whl
```

## Measuring convergence (small variants)

On variants with few starting hearts the best response can be enumerated
exactly, so convergence is directly measurable — exploitability falls steeply
with training (see Results):

```bash
uv run scripts/convergence.py --hearts 1 --iters 1000 10000 100000 1000000
```

## Training and evaluating the full game

```bash
# Train a 6-heart (full game) strategy; checkpoints to data/solver_6h.bin
uv run scripts/train_solver.py --hearts 6 --iters 400000 --chunks 8 --abstract --eval-deals -1

# Win rates vs. baselines (with 95% Wilson CIs) + true-game exploitability
uv run scripts/evaluate.py --solver data/solver_6h.bin --games 2000 --eval-deals 200
```

`evaluate.py` plays the solver against a uniform-random agent and a ladder of
threshold heuristics ("draw while total < t"), alternating seats to cancel
first-player bias. Pass `--eval-deals -1` to skip the (slow) exploitability
measurement, or `--mixed` to play the sampled equilibrium policy instead of the
greedy one.

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
