# cfr — algorithms vs. imperfect-information games

A small framework that separates the **algorithms** (counterfactual regret
minimization, exact best-response exploitability) from the **games** they run on,
so a new game is "implement one trait" and immediately gets a solver and a true
equilibrium-quality measurement. Same idea as OpenSpiel, scoped down.

## Layout

- `core/` (`cfr-core`) — the `Game` trait and the generic algorithms:
  - `Solver::solve` — CFR+ (vanilla: full-tree, both players + chance enumerated,
    explicit reach probabilities), which converges to a Nash equilibrium.
  - `Solver::exploitability` — **exact** best-response exploitability (NashConv)
    over *information sets* (the best responder commits to one action per infoset,
    so no perfect-information "strategy fusion").
  - `arena` — game-agnostic head-to-head (`play`, `win_rate`): any two agents in
    any `Game`, so the same evaluation tooling works for every game.
  - `core/tests/kuhn.rs` — Kuhn poker, the canonical CFR correctness test.
- `games/liars-dice/` (`liars-dice`) — the Liar's Dice variant from the companion
  project, implemented as a `Game` (private dice are chance-rolled each round, so
  it's modelled as the real imperfect-information game, not the env's
  roll-at-call simplification).
- `games/twentyone/` (`twentyone-game`) — Twenty-One as a `Game`, wrapping the
  engine in `twentyone-core` (the deal and each draw surface as chance nodes).
  The full game is far too large for the generic full-tree solver, so this hosts
  Twenty-One in the framework and validates the wrapper; the strong solver for the
  real game stays the specialized decomposed one in `twentyone-core`. The same
  `arena` runs on it unchanged.

## The `Game` trait

```rust
trait Game {
    type State; type Action;
    fn initial_state(&self) -> State;
    fn turn(&self, &State) -> Turn;            // Chance | Player(i)
    fn is_terminal(&self, &State) -> bool;
    fn returns(&self, &State, player) -> f64;  // terminal utility, zero-sum
    fn legal_actions(&self, &State) -> Vec<Action>;
    fn chance_outcomes(&self, &State) -> Vec<(Action, f64)>;
    fn apply(&self, &mut State, Action);
    fn infoset_key(&self, &State, player) -> u64;  // what `player` can observe
    fn state_key(&self, &State) -> Option<u64>;    // god's-eye, for memoization
}
```

## Results

**Correctness (Kuhn poker).** CFR+ drives exact exploitability to ~0 and the
average strategy to the known value of −1/18:

```
$ cargo test --release -p cfr-core --test kuhn      # passes
```

**Liar's Dice (1 die × 3 faces) — a true exploitability curve the reference
project never had** (it only reported a strategy-entropy proxy and had no exact
best response):

| iters/full-CFR | 10 | 100 | 1000 |
|----------------|-----|------|------|
| exploitability | 0.073 | 0.019 | 0.006 |

```
$ cargo run --release -p liars-dice --example exploitability 1 3
```

## Status and scope

Training is **vanilla CFR+** — exact and simple, but it walks the whole game tree
every iteration, so it only scales to small configs (the bid ladder × per-round
dice re-rolls grow the tree fast). That is enough to *measure* the framework and
get an honest exploitability curve. Scaling to the full 2×4 config (or large
Liar's Dice) needs Monte-Carlo sampling (MCCFR) in `Solver::solve` — the natural
next optimization; the `Game` interface and the exact-exploitability yardstick
stay the same.
