# Technique bake-off (historical record)

A controlled comparison of the strength levers, each given an **equal 5-minute
wall-clock training budget** and required to play in **< 0.1 s/move**. Strength
was measured two ways: head-to-head win rate (parallel gauntlet, 1500
games/pairing, draws = half) and exact best-response exploitability on the small
1-heart variant (lower = closer to unbeatable). Machine: 18 cores, CPU only.

These runs were driven by a since-deleted Python harness (`twentyone-rl`, in git
history); the solver, exploitability, and all techniques below live in this
crate (`twentyone::Solver`), and `examples/solve.rs` reproduces the
train-and-measure loop.

## Ranking (best → worst)

| # | technique | what it is | 5-min train | win-rate rank | notes |
|---|-----------|-----------|------------:|:-------------:|-------|
| 1 | **Decomposed CFR+ + abstraction** | round-by-round CFR+, unseen-set band abstraction (the "throw compute at it" baseline) | 550k iters/subgame | **1st (59.9%)** | champion; also best exploitability. Plays instantly. |
| 2 | **Decomposed CFR+, lossless** | same, no abstraction | 200k iters/subgame, **44M** infosets | 2nd (58.6%) | abstraction-off; heavily undertrained at this budget (≈1 visit/infoset) yet competitive on common lines. Needs ~7 GB RAM. |
| 3 | **Full-game CFR (recursive continuations)** | one multi-round tree, exact cross-round values, no continuation approximation | **33k** iters | 3rd (58.1%) | per-iteration cost grows with game length (traverser branches every round), so ~16× fewer iters — yet nearly matches #1 by concentrating on the opening rounds. Did **not** beat the decomposition's exploitability floor in budget. |
| 4 | **Inference search** (PIMC *and* 1-ply blueprint lookahead) | within-round look-ahead on the exact cards, opponent = blueprint | n/a (uses #1) | **below the table** | Both lose to playing #1 greedily: PIMC **47.6%** (CI [45.8, 49.4]); the strategy-fusion-free 1-ply lookahead **47.3%** (CI [45.8, 48.8]); ~11 ms/move. The CFR table already plays the hidden card correctly with the *equilibrium* belief; search on the exact cards uses a wrong (uniform) belief and a determinized future, so it underperforms. |
| 5 | **Deep CFR** | neural advantage/strategy nets | — | not competitive at this budget | A Python+torch traversal tops out ~85k node-evals/s (12 µs/net-eval); the Rust tabular solver processes ~1000× more game situations/second. Deep CFR pays off when the game is too large for a table *and* a GPU is available — neither holds here. Would rank last at a 5-min CPU budget. |

1-heart exploitability (exact best response, lower = better), at matched iters:

| iters | decomposed (#1) | full-game (#3) |
|------:|:---------------:|:--------------:|
| 200k  | ~0.22 | 0.267 |
| 800k–1M | **0.19** | 0.212 |
| 3M | **0.18** | — |

## Takeaways

- **For a game this size, the simple tabular CFR+ with abstraction wins.** It is
  exact, embarrassingly parallel, and does so many updates/second that the fancier
  methods can't catch up in equal wall-clock.
- **Recursive continuations** is theoretically cleaner (no continuation
  approximation) and competitive per-budget, but the multiplicative cost of a
  full-game traversal makes it slower per iteration; it neither beat the win-rate
  champion nor the exploitability floor here.
- **Abstraction is a net win at a fixed budget** — the lossless model spends its
  whole budget discovering 44M information sets it can't train, while the
  abstracted model converges on ~5M and plays them well.
- **Inference-time search hurts** — two independent methods (determinized PIMC
  and a strategy-fusion-free 1-ply blueprint lookahead) both lose to simply
  trusting the equilibrium table, because the table already encodes the correct
  belief-conditioned play. Only full imperfect-information re-solving with proper
  ranges (DeepStack/Libratus) could plausibly help, and it has little to gain
  while the blueprint is already near-optimal within a round.
- **The real frontier is exploitability, not win-rate** — the threshold
  heuristics are already near-optimal within a round, so win-rate against them
  saturates near 55–60% for any competent solver. The differences that matter
  show up in best-response exploitability.

## What would actually push it further (beyond a 5-min CPU budget)

- Full imperfect-information **continual re-solving** (the correct version of #4):
  refine the current round subgame online against the opponent *range* (not a
  determinization), using the blueprint's value table at the leaves. Avoids
  strategy fusion; needs careful range bookkeeping.
- **Deep CFR with a Rust-side traversal + batched GPU inference**, for a fair shot
  at the neural approach — only worth it on a larger variant where the table
  stops fitting in memory.
