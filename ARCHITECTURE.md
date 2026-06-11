# Architecture

The lab exists to answer one question well: **how do game-playing algorithms
generalize across games?** Everything below follows from refusing to let either
side own the other — algorithms must not be rewritten per game, and games must
not import algorithm internals.

## The layers

```
                 ┌─────────────────────────────────────────────┐
                 │ lab            registry · type-erased match │  ← CLI today,
                 │                · generic terminal client    │    web server next
                 └───────┬─────────────────────────┬───────────┘
                         │                         │
        ┌────────────────┴──────┐   ┌──────────────┴──────────────┐
        │ solvers               │   │ games/*                     │
        │ generic algorithms:   │   │ rules + game knowledge:     │
        │ cfr · mccfr ·         │   │ Game impl · Eval ·          │
        │ alpha-beta · rollout  │   │ Determinizer · SearchSpec · │
        │ · exploitability      │   │ GameUi (+ bespoke solvers)  │
        └───────────┬───────────┘   └──────────────┬──────────────┘
                    │                              │
                 ┌──┴──────────────────────────────┴──┐
                 │ game-core    Game · Agent · arena · │
                 │ capability traits · Rng             │
                 └─────────────────────────────────────┘
```

Dependency rule: `game-core` depends on nothing; `solvers` and `games/*` depend
only on `game-core` (games may use `solvers` in dev-dependencies for tests and
experiments); `lab` binds everything. Games never depend on solvers at the
library level, so adding an algorithm never recompiles a game and vice versa.

## The contract: capability traits

A game implements `Game` (rules: chance/decision nodes, legal actions, terminal
returns, information-set keys). That alone earns it the arena, CFR/MCCFR (if
small enough), and exploitability. Every further power is unlocked by declaring
*knowledge*, never by writing an algorithm:

| the game declares | in trait | which unlocks |
|---|---|---|
| a static value estimate | `Eval` | `solvers::AlphaBeta` (negamax, alpha-beta, quiescence, iterative deepening) |
| noisy actions + move ordering | `SearchSpec` | sharp pruning & horizon extension in the same search |
| how to sample hidden info | `Determinizer` | `solvers::Rollout` (determinized Monte-Carlo with common random numbers) |
| per-player view, action labels/parsing, transition narration | `GameUi` | the universal client in `lab` — no game writes a play loop |

Concretely: chess ships piece-square tables (`MaterialEval`) and "captures are
noisy, MVV-LVA first" (`ChessSpec`) — about 60 lines of knowledge — and receives
a full tournament-shaped search engine. Liar's Dice ships "bidders plausibly
hold the face they bid" (`BidConditioned`) and receives parallel determinized
rollouts. Neither contains a line of search machinery.

**Bespoke algorithms are allowed but live with their game.** Twenty-One's
round-decomposed CFR+ solver exploits structure (rounds linked only by public
hearts) that no generic interface should pretend to capture; it stays in
`games/twentyone` and is exposed to the rest of the lab as an ordinary `Agent`.

## Adding things

**A game** (the acid test of the design): implement `Game` + `GameUi`, register
one entry in `lab/src/registry.rs`. It immediately gets the arena, the CLI, and
(for perfect-information games) alpha-beta the moment you write a ~30-line
`Eval`. Nothing else in the repo changes.

**An algorithm**: write it once in `solvers` against `Game` plus whatever
capability traits it needs. It immediately runs on every game that has them. If
it needs knowledge no trait captures yet, add a trait to `game-core` with a
sane default — never reach into a specific game.

## Identity choices worth knowing

- **Actions are indices.** Agents return an index into `legal_actions(state)`,
  which must be stably ordered per information set. This keeps `Action` types
  fully game-private, makes tabular methods line up, and gives serving a
  wire-format for free (index + label).
- **Information sets are u64 keys** (hashes of sufficient statistics).
  Collision odds at tens of millions of infosets are negligible (~2⁻²⁵); the
  payoff is flat, fast tables.
- **One human-quality randomness contract.** `Agent::act` receives a single
  uniform `r` for mixed strategies; deterministic agents ignore it. Matches are
  reproducible from a seed.
- **Measure one change at a time.** Evaluation is win share against a *field*
  with the hero rotated through seats (fair = 1/players); single runs can be
  ~2σ lucky (it happened — see `games/liars-dice/examples/ab.rs`).

## The path to the website

The full client-side web design (wasm engine, per-game frontends, tournaments
in the browser) is specified in [WEB.md](WEB.md). The short version: `lab`
already contains the two pieces any serving layer needs, deliberately
separated from the terminal:

1. **The registry** (`lab/src/registry.rs`): `game id + options + bot id →
   Box<dyn AnyMatch>` — the catalog of what can be played.
2. **`AnyMatch`** (`lab/src/runner.rs`): a type-erased match with a uniform,
   string/index-based surface — `advance()` (chance + bot moves, narrated),
   `view()` (the human's information only), `legal_labels()`, `apply_human()`,
   `result_text()`.

A web service is a thin loop over exactly these calls: `POST /match {game,
opts}` → store the `AnyMatch` in a session → return `view + labels` → `POST
/match/:id/move {index|text}` → `apply_human` + `advance` → repeat. Hidden
information is already respected because `view`/narration are viewer-scoped.
What to add when that day comes: serde on the messages, a structured
(JSON) variant of `render` for rich clients, and artifact loading (trained
solvers from disk) instead of train-at-startup. None of it touches `game-core`,
`solvers`, or any game.

## Current algorithm/game matrix

|                | chess | othello | connect4 | go | liars-dice | twentyone | kuhn (test) |
|----------------|:-----:|:-------:|:--------:|:--:|:----------:|:---------:|:-----------:|
| `Cfr` (+ exact exploitability) | — | — | — | — | tiny configs | — | ✓ → Nash |
| `Mccfr` / `OsMccfr` | — | — | — | — | OS handles the deep ladder | — | ✓ |
| `AlphaBeta` | ✓ (the bot) | ✓ (the bot) | ✓ (the bot) | — (no eval) | — (imperfect info) | — | — |
| `Mcts` | possible | possible | possible | ✓ (the bot) | — | — | — |
| `azero` (PUCT + self-play net) | ✓ (training) | possible | possible | possible | — | — | — |
| `Rollout` | possible | possible | possible | possible | ✓ (the bot) | possible | — |
| bespoke | — | — | — | — | belief policy | decomposed CFR+ (the bot) | — |

The dashes are honest: tabular CFR can't fit big games, search can't see hidden
information, Go has no hand-written eval. Notable measured facts: outcome-sampling
MCCFR runs a 200-deep ladder in milliseconds/iteration where external sampling
would need ~1e41 nodes; CFR+ regret flooring provably stalls outcome sampling
(documented in `solvers/src/os_mccfr.rs`); the azero loop's checkpoint beats
random at chess within minutes of CPU self-play, while real chess *strength*
remains a GPU-scale endeavor — which is what `azt/` is for: a deliberately
*standalone* crate (not a workspace member, so libtorch never touches the
main build) that trains an AlphaZero resnet on Apple-GPU via tch-rs, batching
leaf evaluations across hundreds of concurrent games. It consumes the chess
crate's `encode_planes`/`az_move_index` knowledge and the same run-dir
contract (metrics.jsonl + dashboard + STOP) as the CPU harness.
