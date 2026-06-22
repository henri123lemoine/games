# Poker — No-Limit Texas Hold'em

No-Limit Texas Hold'em (2–9 seats, 6-max by default) as a `game-core` `Game`,
with a hand-crafted equity bot that crushes casual play. No training, no CFR,
no GPU — pure CPU.

```bash
cargo run --release -p lab -- play poker players=6 samples=2000   # sit down
cargo run --release -p lab -- play poker players=2 seat=watch     # watch bots
cargo run --release -p poker --example bot_eval                   # bot vs baselines
cargo test --release -p poker                                     # rules + evaluator + bot
```

## The shape

One `Game` is **one hand**: chance deals the hole cards and the board one card
at a time (`Action::Deal(card)`, so the chance space stays enumerable at ≤ 52
outcomes per node and exact), players act in four betting rounds
(fold / check / call / raise / all-in), and `returns` is each seat's net chip
change for the hand **in big blinds** — the natural zero-sum poker scale, so an
arena's mean return is directly bb/hand. Side pots, all-ins, the blinds, the
big-blind option and heads-up button rules are all handled; a betting round
closes only once every live, non-all-in seat has acted and matched the bet.

A seat's information set is its two hole cards plus the public board and the
betting history; the rest of the deck is hidden.

## The bot

`PokerBot` (registry id `equity`) is tight-aggressive "ABC poker": it estimates
its hand's equity against the live field by Monte-Carlo — deal the opponents'
hole cards and the missing board a few thousand times, count showdown wins (ties
split) — then decides on pot odds plus position and a little randomized
aggression and bluffing. Tunable via `samples` (Easy/Medium/Hard in the arcade).

The generic `solvers::Rollout` also plays poker through the `HoleSampler`
determinizer (id `rollout`); `call` (a calling station) and `random` are
baselines. Measured edge (`examples/bot_eval`, hero rotated through all seats):

| matchup            | heads-up   | 6-max       |
|--------------------|:----------:|:-----------:|
| equity vs always-call | +186 bb/100 | +1085 bb/100 |
| equity vs random      | +1197 bb/100 | +344 bb/100 |

A strong human winner makes a few bb/100 against tough fields; these numbers are
"crushes a casual player" by a wide margin.

> **Measure poker by bb/hand, not win share.** The lab `compare` harness scores
> by who has the single top return per hand, which can't exceed the fair `1/N`
> by much when only one seat wins the pot — it will report the bot as "not ahead
> of the field" even though it dominates. The truthful metric is mean net
> bb/hand (`examples/bot_eval`).

## Hand evaluation

`cards.rs` evaluates the best five of seven cards categorically into a single
comparable `u32` (category, then packed kickers), so equal hands compare exactly
(split pots). It is cross-checked against a brute-force best-of-21 evaluator over
50,000 random hands.
