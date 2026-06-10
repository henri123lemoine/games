# Benchmarks

Cross-game benchmark round using the `lab compare` (SPRT) and `lab tourney`
harness, plus per-crate score benches for the single-player games.

- **Date:** 2026-06-10
- **Machine:** Apple M5 Max, 18 cores, 64 GB RAM, macOS 26.5
- **Toolchain:** rustc 1.92.0, `--release` builds
- **Harness defaults:** GSPRT on H0 elo=0 vs H1 elo=20, alpha=beta=0.05, paired
  seat-swapped games with a shared random opening book; N-player games use a
  hero-vs-field binomial SPRT (H1: win share >= 1/n + delta). Elo intervals
  below are the harness's 95% margins.

## Summary

| Game | Matchup | W-D-L (games) | Elo / score | Verdict |
|---|---|---|---|---|
| connect4 | alphabeta d7 vs d5 | 100-10-50 (160) | +112 +/- 56 | accepted H1 |
| chess | alphabeta-rich d5 vs alphabeta d5 | 45-41-10 (96) | +133 +/- 74 | accepted H1 |
| chess | alphabeta d5 vs azero (256 sims) | 99-0-1 (100, fixed) | fit gap ~760 | crush (exhibit) |
| othello | alphabeta d5 vs mcts 200 (time parity) | 28-2-2 (32) | +394 +/- 196 | accepted H1 |
| othello | alphabeta d5 vs mcts 2000 (10x time) | 30-0-2 (32) | +470 +/- 231 | accepted H1 |
| go 9x9 | mcts-eval 500 vs mcts 500 | 23-0-17 (40) | +53 +/- 107 | inconclusive |
| liars-dice 5p | rollout 400 vs belief field | 23-41 (64) | share 0.359 (fair 0.200) | accepted H1 |
| 2048 | mcts-eval sims=200 depth=8, 50 eps | — | mean 15950, median 15752 | 10/50 reach 2048 |
| snake 10x10 | mcts-eval sims=200 depth=12, 50 eps | — | length mean 3.0, median 3 | broken — never eats |

## connect4 — harness sanity: depth 7 vs depth 5

```
lab compare connect4 a=alphabeta:depth=7 b=alphabeta:depth=5 max-games=400 seed=1
```

Accepted H1 after 160 games: **100-10-50, elo +112 +/- 56**, 0.11 s wall.
Deeper search wins by a wide, significant margin, and the SPRT terminates in a
fraction of the game budget — exactly the cheap sanity check the harness is
for. Connect-4 games at these depths cost well under a millisecond each.

## chess — rich eval vs material eval, fixed depth 5

```
lab compare chess a=alphabeta-rich:depth=5 b=alphabeta:depth=5 max-games=400 seed=42
```

Accepted H1 after 96 games: **45-41-10, elo +133 +/- 74**. The tapered
rich eval is worth on the order of +130 elo over material+PST at equal depth,
though the interval is wide — treat the magnitude as rough, the direction as
settled. Notably draw-heavy (41/96): equal-depth alpha-beta mirrors often peter
out into repetition once the opening book's 6 random plies wear off.

## chess — learned vs handcrafted: alphabeta vs azero

```
lab compare chess a=alphabeta:depth=5 b=azero:net=data/azero/chess.bin,sims=256 max-games=16 seed=7
lab tourney chess bots=alphabeta:depth=5,azero:net=data/azero/chess.bin,sims=256 games=100 seed=7
```

The SPRT is a formality — accepted H1 after the minimum 16 games (16-0-0,
point estimate +2400 elo). The 100-game fixed sample is the real exhibit:
**alphabeta wins 99-0-1** (8.7 s wall; the tourney's regularized
Bradley-Terry fit reports a compressed +/-380 split, i.e. a ~760-elo gap —
the raw record is more honest). The distilled azero MLP at 256 PUCT sims is
nowhere near a depth-5 material searcher; this benchmarks the small net we
trained, not AlphaZero-the-method. Azero did steal exactly one game, courtesy
of the randomized 6-ply opening.

## othello — alpha-beta vs MCTS at time parity

```
lab compare othello a=alphabeta:depth=5 b=mcts:sims=200 max-games=400 seed=21
lab compare othello a=alphabeta:depth=5 b=mcts:sims=2000 max-games=400 seed=21
```

**Parity basis:** self-play CPU time per player per game, measured with
40-game `lab tourney` self-matches — alphabeta:depth=5 costs ~11 ms,
mcts:sims=200 ~12.4 ms (mcts:sims=1000 ~52 ms, sims=2000 ~101 ms). So
sims=200 is CPU parity within ~12%.

At parity, accepted H1 after 32 games: **28-2-2, elo +394 +/- 196**. Giving
MCTS 10x the compute (sims=2000) does not help: **30-0-2, elo +470 +/- 231**.
Vanilla MCTS with uniform-random rollouts is simply miscalibrated for othello
— random playouts are nearly uninformative about disc-flip dynamics — so the
weighted-squares alpha-beta wins at any affordable sims budget. Both
intervals are wide (early SPRT stops); the direction is unambiguous.

## go 9x9 — eval-truncated MCTS vs vanilla MCTS

```
lab compare go a=mcts-eval:sims=500 b=mcts:sims=500 max-games=40 batch=8 seed=9
```

Inconclusive after 40 games: **23-0-17, elo +53 +/- 107** (LLR 0.29, bounds
+/-2.94). The GoEval-truncated playouts trend positive but 9x9 games at 500
sims are slow enough that this budget cannot separate elo-20-sized effects;
the honest verdict is "raise max-games to decide". Run during integration
wrap-up after the original benchmark agent hit the session's spend limit.


## liars-dice 5p5d6f — rollout vs belief field

```
lab compare liars-dice a=rollout:rollouts=400 b=belief max-games=400 seed=17
```

Hero-vs-field binomial SPRT (H0 share 0.200, H1 share 0.300, hero rotated
through all 5 seats). Accepted H1 after 64 games: **23-41, win share 0.359**
— nearly double the fair share against four belief-agent opponents. Matches
the league result that the determinized-rollout agent is the house champion.
Note the share drifted down from 0.500 over the first 16 games, so 0.359 is
itself an optimistic-side estimate of a real but smaller edge.

## 2048 — registered bot score bench

```
cargo run --release -p g2048 --example bench 50 200 8 1
```

50 episodes of the registered `mcts-eval` bot (sims=200, depth=8 — the
`lab play 2048` defaults), 14.8 s total: **score mean 15950, median 15752,
min 7112, max 32180; 41/50 episodes reach the 1024 tile, 10/50 reach 2048.**
Solid play for 200 sims/move — the Heuristic2048-truncated search reliably
builds 1024+ positions, and a fifth of runs hit 2048.

## snake 10x10 — registered bot score bench

```
cargo run --release -p snake --example bench 50 200 12 1
```

**Degenerate: length mean 3.0, median 3, min 3, max 3** — across all 50
episodes the bot dies at the starting length without eating a single food.
This is not a bench-harness artifact: `lab play snake bot=mcts-eval` (and
`bot=mcts`, and sims=2000) crashes at length 3 on every seed tried, usually
by spiralling into itself. Likely cause: SnakeEval's food-shaping term is at
most ~0.01 on the returns scale, far below the UCB exploration term, so the
search is effectively value-blind at these budgets. The registered snake bot
should be considered broken until the eval/exploration scaling is revisited.

## Deferred / caveats

- The azero compare entry was wired for this round (the net is loaded once
  and Arc-shared across the per-game builders); `twentyone` still has no
  compare entry.
- `game_core::Rng::new` does `seed | 1`, so seeds 2k and 2k+1 collide — bench
  example episode seeds are spaced by 2 to avoid it.
- Several accepted-H1 elo intervals are wide (+/-200 and up) because the SPRT
  stops as soon as the verdict is decided; the verdicts are significant at
  alpha=0.05, the point estimates are not precise.
