# Twenty-One

A 2-player hidden-information card game, with the game engine, a fast
specialized solver, and an adapter to the lab's generic `Game` trait — all in
this one crate. (This was the repo's original project; the Python RL harness it
once shipped with is gone — its findings live on in `BAKEOFF.md` and below, and
everything it did is now a `cargo run` away.)

## Rules

- Each player starts with 6 hearts; each round uses cards 1–11 (one of each).
- Both players get a face-up card (public) and a face-down card (hidden), then
  alternate choosing **draw** (face-up) or **stand**.
- Closest to 21 without busting wins the round; the winner deals damage equal
  to the round number. First player out of hearts loses.

## What's here

- `env` — the engine: allocation-conscious, bitmask deck, deterministic
  controllable-chance API (`deal_specific`/`draw_specific`) so solvers own the
  randomness. Rule invariants are tested in `tests/rules_invariants.rs`.
- `solver` — **decomposed external-sampling MCCFR+**: rounds are linked only by
  the public `(hearts, hearts, round)` carry-over, so the game is solved by
  backward induction over that lattice. Information sets are lossless
  sufficient statistics; the full 6-heart game optionally uses a value-band
  abstraction of the unseen set (~7× smaller, the bake-off champion).
  Exact best-response exploitability, parallel training, save/load.
- `game` — the engine adapted to `cfr_core::Game` (deal and draws surface as
  chance nodes), so Twenty-One plugs into the same arena as the other games.

## Results

Convergence on the 1-heart variant (exact best response; utilities in [-1, 1]):

| iters/subgame  |    1k |   10k |  100k |    1M |    3M |
|----------------|------:|------:|------:|------:|------:|
| exploitability | 0.782 | 0.581 | 0.273 | 0.193 | 0.181 |

The residual is the price of the decomposition (rounds linked by Monte-Carlo
continuation values). Full 6-heart game (band abstraction, 400k iters/subgame),
2000 games/matchup: beats Random **90.1%**, Threshold(14–16) **52–58%** — the
threshold heuristics are near-optimal within a round, so win-rate saturates and
exploitability is the metric that separates solvers.

`BAKEOFF.md` ranks the techniques tried under an equal 5-minute budget
(decomposed CFR+ with abstraction won; inference-time search *hurt*; Deep CFR
was not competitive on CPU).

## Run it

```bash
cargo run --release -p twentyone --example solve 1 100000          # train + exact exploitability
cargo run --release -p twentyone --example solve 6 400000 s.bin    # train the full game, save
cargo run --release -p twentyone --example play 6 s.bin            # play against it
cargo run --release -p twentyone --example play 6                  # quick-train then play
```
