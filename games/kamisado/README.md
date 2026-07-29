# Kamisado

Single-round Kamisado (Peter Burley, 2008) against the shared `Game` contract:
an 8×8 board of fixed colors, eight towers a side, every move a forward or
diagonal-forward slide, and the twist that the opponent's landing square
dictates which of your towers must move next. First tower to reach the far
rank wins. Black moves first.

## Encoding choices

* **The board** is the official grid, hard-coded and test-pinned: every rank
  and file holds each color once, the a1–h8 diagonal is all Brown, h1–a8 all
  Orange, and the grid is 180°-rotation symmetric. Combined with each tower
  starting on its own color, that symmetry makes the two sides perfectly
  interchangeable — only moving first matters.
* **Obligation passes are collapsed.** When the required tower is blocked the
  official rules pass the obligation along (the opponent moves the tower of
  the color under the blocked one, and so on). The position is unchanged
  while obligations bounce, so the chain is deterministic: `apply` follows it
  to the first movable tower — which may hand the turn straight back, so
  turns don't strictly alternate — or detects a revisit, which means the
  passing recurs forever: a **deadlock**, lost by the player who made the
  last actual move. No null actions ever appear in the tree.
* **No draws, no repetitions.** Every action advances a tower at least one
  rank, so a round lasts at most 112 actions and states never repeat.

Rules and colors are cross-validated against the
[hamisado](https://github.com/sphynx/hamisado) analysis project: a
test-only walker replays its conventions (explicit pass moves, two passes end
the round) and reproduces its published perft counts, 102 / 1 150 / 11 182 /
105 024 / 901 006 / 7 399 924 for depths 1–6 — obligation colors steer the
tree from depth 2 on, so this pins the whole grid.

## The bot

`KamisadoEval` (rank progress + towers with a clear slide to the goal rank +
blocked-tower penalty) and `KamisadoSpec` (wins first, long advances first,
demote moves whose landing color hands the opponent a standing one-move win)
feed the generic `solvers::AlphaBeta`:

```bash
cargo run --release -p lab -- play kamisado depth=14
cargo run --release -p lab -- play kamisado bot=mcts sims=5000
```

## The game is weakly solved: first player wins

Kamisado is search-friendly — no draws, branching ≈ 10 under the forced-tower
rule, strict progress bound — and `examples/solve.rs` weakly solves it with
the same generic alpha-beta the bot uses: terminal scores live on a "mate"
scale no heuristic leaf can reach, so a mate-scale root score is a proof, and
the heuristic evaluation still prunes the undecided regions (which is exactly
where a pure win/loss/unknown prover dies: it cannot cut an undecided branch
and degenerates to full-width minimax).

```
cargo run --release -p kamisado --example solve

depth 15  eval +0.634   best d1-d6    nodes      8220309  4.1s
depth 16  WIN in 17     best a1-a5    nodes     10910869  5.7s

Kamisado is a first-player (Black) WIN: forced in 17 actions, found at search depth 16.
proof line: B:a1-a5 W:d8-d3 B:f1-f7 W:e8-e4 B:h1-h7 W:c8-c7 B:e1-b4 W:f8-d6
            B:c1-c3 W:h8-f6 B:a5-a7 W:d6-e5 W:e5-d4 W:d4-e3 B:c3-c6 W:a8-b7 B:h7-h8
```

(~11M nodes; the depth-16 round proves a 17-action win through transposition
grafting from earlier rounds. Note the three consecutive White moves in the
proof line — Black's obligated towers are walled in and pass.)

The example then re-derives the verdict with none of the alpha-beta machinery:
a bounded AND-OR search over the rules alone (exists-a-Black-move /
for-all-White-moves, memoized) — no evaluation, no windows, so `true` is a
proof from the rules and `false` is exhaustive:

```
  win within 15 actions: false  nodes    272469030  113.0s
  win within 16 actions: false  nodes    586590783  195.0s
  win within 17 actions: true   nodes    603190976  4.4s
```

Black's fastest forced win takes *exactly* 17 actions (~4.5 GB memo at peak).

This reproduces the known result — the hamisado project first proved the
first-player win (depth 17 in its pass-inclusive move counting), which is the
source of the claim echoed on
[Wikipedia](https://en.wikipedia.org/wiki/Kamisado) — under the official
deadlock rule (hamisado ends the round after two consecutive passes; this
crate follows the rulebook's longer obligation chains, and the proof holds).
