# Snake: redesign to a competitive 1v1 game

**Status:** Done (option 1). The arcade's `snake` entry is the 2-player
`snake::Duel` (20x20) with the generic MCTS-eval bot as the baseline; the
polished arrow-key/swipe frontend lives in `web/app/src/frontends/snake/`,
and the game joins watch mode and the tournament lab via its `eval` entry. The
single-player `snake::Snake` is retained for the terminal client and its
tests/examples. This note records *why* it was pulled and *how it came back*;
the rationale below is the spec that was implemented.

## Why it was pulled

The arcade's thesis is "play against the lab's bots." Solo Snake has no
opponent, so the bot — the whole point of the site — has nowhere to live. It
is also trivially easy as a solo game, and the old web frontend steered with a
left/straight/right **button pad** (relative turns), which is an unintuitive
control scheme bolted onto what should be an arrow-key/swipe game. The fix is
not a frontend polish pass; it's to make Snake the kind of game the rest of
the arcade is: a contest against an agent.

## Target

Two snakes on one shared grid — you versus a lab bot (and bot-vs-bot in watch
mode, like every other versus game). Classic competitive Snake / "Tron-snake":

- Both snakes move one cell per tick.
- Food spawns on empty cells; eating grows the snake and scores.
- A snake dies hitting a wall, its own body, or the opponent's body.
- Head-to-head collision: the longer snake survives, or both die if equal.
- Win = opponent dies first; on a step/time cap, higher score wins.

This makes the bot meaningful, fits the existing seat-roster UI for free
(`White`/`Black` → `Snake A`/`Snake B`), and joins watch mode and the
tournament lab automatically once it has an `eval` entry.

## The one real design fork: simultaneous moves

Snake is naturally **simultaneous** — both snakes commit a direction, then the
tick resolves at once. The lab's `Game` trait is sequential/turn-based (chess,
dice). Two ways to model it:

1. **Alternating sub-ticks** (cheapest): within one visual tick, snake A picks,
   then snake B picks seeing A's choice, then both advance. Slightly unfair to
   the first mover, but simple and entirely within the current trait. Good
   enough for a casual arcade game and a fine first version.
2. **True simultaneous** (cleaner, bigger): add simultaneous-move support to
   `game-core` (a turn type that collects one action per player before
   resolving). Higher value beyond Snake (it's the missing piece for any
   real-time/simultaneous game), but a real change to the core contract and to
   every solver that assumes alternation.

Recommendation: ship (1) first to get the game and the bot working end to end;
revisit (2) only if the first-mover bias is visibly unfair in practice.

## Implementation sketch

Rust (`games/snake`):
- A 2-player game (generalize `Snake` to N snakes, or a new `snake::Duel`):
  `num_players() == 2`, the resolution rule above, terminal + winner.
- `Eval` for the search: flood-fill / reachable-territory (Voronoi between the
  two heads) + length + distance-to-food + a stay-alive term. This is the
  standard strong heuristic for competitive snake.
- Reuse MCTS / AlphaBeta over the perfect-information game; later, an RL/azero
  policy is the natural "strong bot" and fits the lab's azero stack.
- `GameUi::view_data`: both snakes (head-first cell lists), food, per-snake
  score, status — extend the current single-snake schema.
- Registry entry: `solo: false`, bots e.g. `mcts | mcts-eval | greedy`, with an
  `eval` entry so it joins compare/tournaments.

Web (`web/app/src/frontends/snake/`):
- New frontend rendering two snakes; keep the real-time auto-tick loop from the
  old frontend (git history: `web/app/src/frontends/snake/index.ts` before this
  change), but: **arrow/WASD + swipe**, no on-screen turn-button pad; per-tick
  the bot's move is computed within the tick budget.
- Remove `snake` from `HIDDEN_GAMES`; it then appears with the seat roster,
  watch mode, and the tournament lab with no further shell changes.

## Effort

The frontend is small. The bulk is the 2-player game + a competent bot; the
simultaneous-move decision above governs how much (if any) `game-core` churn
it pulls in.
