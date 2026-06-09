# Liar's Dice

The companion project's non-standard Liar's Dice, as a `cfr_core::Game` and a set
of strong agents — generalised to **N players × D dice × F faces** so it scales
from 2-player toy games up to the 5-player, 5-dice, 6-face target.

## Rules (unchanged from the reference variant)

- Each player starts with `D` dice; everyone rolls privately each round.
- 1s are **not** wild. A raise is exactly **+1 quantity** (same face) or **+1
  face** (same quantity, wrapping `faces`→1 with +1 quantity). The first round
  opens at a forced `1×1`; later rounds open freely.
- On your turn: raise, **Call Liar**, or **Call Exact**.
  - *Call Liar*: if the true count of the bid face across all live dice is below
    the bid, the bidder loses a die, else the caller does.
  - *Call Exact*: if the count is exactly the bid, the caller loses nothing
    (and opens next); otherwise the caller loses a die.
- A player at zero dice is eliminated; the last player standing wins.

The full game is astronomically large (a single 5-dice hand is one of 252; the
joint hidden state across five players is in the billions, and the bid ladder is
~150 deep), so the agents are **belief-based and search-based**, not tabular CFR.

## Agents

- **`ProbabilisticAgent`** — reasons exactly about the unknown dice with the
  binomial distribution (each unknown die shows a face with probability `1/F`),
  with tunable thresholds for calling liar/exact, bid aggression, opponent
  bid-credibility, opening strength, and a soft (randomised) calling band so it
  isn't perfectly readable. Scales to any size; the default config is
  **self-play tuned** (see below).
- **`RolloutAgent`** — Monte-Carlo lookahead: at each decision it *determinizes*
  (resamples opponents' hidden dice consistent with its own hand), plays each
  candidate action forward to the end with the probabilistic policy on every
  seat, and picks the highest win-probability action. Stronger than the policy it
  rolls out; defers to the policy at the wide opening node to stay fast.

## How it was made strong (self-play, no human in the loop)

The default `ProbConfig` was found by **self-play hill-climbing against a diverse
league** (`examples/league`): conservative, aggressive, trusting, paranoid, and
exact-happy styles plus champion snapshots (fictitious play), maximising average
win share across the panel so it's robust rather than overfit to one opponent.

## Results (win share; *fair* = 1/players)

| eval | config | result |
|------|--------|--------|
| probabilistic vs random field | 5p5d6f | **0.98** (fair 0.20) |
| probabilistic vs random field | 2p5d6f | **0.999** |
| tuned vs a field of the untuned default | 5p5d6f | **0.47** (fair 0.20) |
| rollout vs a field of the probabilistic agent | 3p3d6f | **0.39** (fair 0.33) |

So the tuned belief agent dominates random and the untuned baseline, and the
rollout lookahead adds a further edge over the belief policy itself.

## Run it

```bash
# play against the bots (you are player 0)
cargo run --release -p liars-dice --example play 5 5 6

# evaluate / tune / search
cargo run --release -p liars-dice --example evaluate          # vs random, all configs
cargo run --release -p liars-dice --example league 5 5 6      # robust self-play tuning
cargo run --release -p liars-dice --example rollout_eval 5 5 6 # does lookahead help?
```

## Honest limits

These agents are strong and unexploitable-ish (mixed, belief-calibrated), and
beat every static style we throw at them. They do **not** model a *specific*
adaptive opponent, so a strong human who plays many games could probe for
patterns. Tabular CFR doesn't fit this game (the bid ladder makes external
sampling exponential; outcome-sampling MCCFR over an abstraction is the route to
a learned, opponent-agnostic equilibrium and is the natural next step).
