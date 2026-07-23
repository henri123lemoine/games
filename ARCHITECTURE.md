# Architecture

The lab exists to answer one question well: **how do game-playing algorithms generalize across games?** Everything below follows from refusing to let either side own the other — algorithms must not be rewritten per game, and games must not import algorithm internals.

## The layers

```
   ┌──────────────────────┐        ┌──────────────────────────────────┐
   │ web/  arcade          │        │ ml/  training (standalone)        │
   │ wasm engine over the  │        │ AlphaZero + PPO self-play; their  │
   │ registry; torch-free  │◀─nets─▶│ exported nets are read by the     │
   │ net inference         │        │ torch-free inference, kept off    │
   └───────────┬───────────┘        │ the main build (own [workspace])  │
               │                    └──────────────────────────────────┘
               ▼
   ┌─────────────────────────────────────────────┐
   │ lab            registry · type-erased match │  ← terminal client
   │                · generic serving surface     │    + the wasm engine
   └───────┬─────────────────────────┬───────────┘
           │                         │
  ┌────────┴──────────────┐   ┌──────┴──────────────────────┐
  │ solvers               │   │ games/*                     │
  │ generic algorithms:   │   │ rules + game knowledge:     │
  │ cfr · mccfr ·         │   │ Game/SimultaneousGame ·     │
  │ alpha-beta · rollout  │   │ Determinizer · SearchSpec · │
  │ · mcts · puct/azero   │   │ UI + game knowledge         │
  │ · exploitability      │   │                             │
  └───────────┬───────────┘   └──────────────┬──────────────┘
              │                              │
           ┌──┴──────────────────────────────┴──┐
           │ game-core    game/agent contracts ·  │
           │ capability traits · Rng             │
           └─────────────────────────────────────┘
```

Dependency rule: `game-core` depends on nothing; `solvers` and `games/*` depend only on `game-core` (games may use `solvers` in dev-dependencies for tests and experiments); `lab` binds everything. Games never depend on solvers at the library level, so adding an algorithm never recompiles a game and vice versa. The two outer tiers attach without leaking inward: the wasm engine wraps `lab`'s serving surface, and the training crates are *standalone* (their own `[workspace]`) so the tensor backend never enters the lab's build — the only thing crossing the boundary is an exported net file, which a torch-free forward reads (see *Learning: the ml tree*).

## The contract: capability traits

A turn-based game implements `Game` (chance/decision nodes, legal actions, terminal returns, information-set keys). A game whose players choose before seeing one another's current choices implements `SimultaneousGame` instead; its arena collects a complete joint action from one immutable state and applies it atomically. Encoding that game as sequential nodes is forbidden because it changes the information structure. Every further power is unlocked by declaring *knowledge*:

| the game declares | in trait | which unlocks |
|---|---|---|
| a static value estimate | `Eval` | `solvers::AlphaBeta` (negamax, alpha-beta, quiescence, iterative deepening) |
| noisy actions + move ordering | `SearchSpec` | sharp pruning & horizon extension in the same search |
| how to sample hidden info | `Determinizer` | `solvers::Rollout` (determinized Monte-Carlo with common random numbers) |
| per-player view, action labels/parsing, transition narration | `GameUi` | the universal client in `lab` — no game writes a play loop |
| simultaneous rules and UI | `SimultaneousGame`, `SimultaneousGameUi` | the joint-action arena and the same type-erased terminal/browser clients |

Concretely: chess ships piece-square tables (`MaterialEval`, plus the tapered `RichEval` it grew later) and "captures are noisy, MVV-LVA first" (`ChessSpec`) — a few hundred lines of evaluation knowledge and zero lines of search — and receives a full tournament-shaped engine. Liar's Dice ships "bidders plausibly hold the face they bid" (`BidConditioned`) and receives parallel determinized rollouts. Neither contains a line of search machinery.

**Bespoke algorithms are allowed but live with their game.** Twenty-One's round-decomposed CFR+ solver exploits structure (rounds linked only by public hearts) that no generic interface should pretend to capture; it stays in `games/twentyone` and is exposed to the rest of the lab as an ordinary `Agent`.

## Adding things

**A game** (the acid test of the design): implement `Game` + `GameUi`, or the simultaneous pair when choices are concurrent, then register one entry in `lab/src/registry.rs`. It immediately gets the matching arena plus the CLI/browser serving surface. Never force a concurrent ruleset through `Game` merely to reuse an alternating solver.

**An algorithm**: write it once in `solvers` against `Game` plus whatever capability traits it needs. It immediately runs on every game that has them. If it needs knowledge no trait captures yet, add a trait to `game-core` with a sane default — never reach into a specific game.

## Identity choices worth knowing

- **Actions are indices.** Agents return an index into `legal_actions(state)`, which must be stably ordered per information set. This keeps `Action` types fully game-private, makes tabular methods line up, and gives serving a wire-format for free (index + label). For cross-state identity (killer/history/RAVE tables), `Game::action_id` gives every action a stable u64 — defaulted via its `Debug` form, overridden cheaply by games search cares about.
- **Information sets are u64 keys** (hashes of sufficient statistics). Collision odds at tens of millions of infosets are negligible (~2⁻²⁵); the payoff is flat, fast tables.
- **One randomness contract.** `Agent::act` receives `&mut Rng` — a private, seeded stream for mixed strategies and stochastic search; deterministic agents ignore it. Matches are reproducible from the arena seed, and agents stay `&self` so they can be shared across seats and parallel games.
- **Current simultaneous moves stay hidden.** `SimultaneousAgent::act` receives the common pre-state. The arena does not mutate it until every active player has chosen; searches and neural backups operate on joint actions for the same reason.
- **Draws are first-class.** `play` returns the actual utility, `win_rate` scores draws ½, and N-player ties split `win_share` so an all-draw field reads exactly the fair `1/players` — never a phantom win for seat 0.
- **Returns are bounded by `Game::max_return`** (default 1.0). Anything that mixes static evaluations with returns or detects proven wins (MCTS-Solver) keys on that bound instead of assuming the win/loss convention.
- **Measure one change at a time.** Evaluation is win share against a *field* with the hero rotated through seats (fair = 1/players); single runs can be ~2σ lucky (it happened — see `games/liars-dice/examples/ab.rs`).

## Serving: terminal and web

`lab` exposes exactly two serving interfaces, deliberately separated from the terminal client:

1. **The registry** (`lab/src/registry.rs`): `game id + options + bot id → Box<dyn AnyMatch>` — the catalog of what can be played.
2. **`AnyMatch`** (`lab/src/runner.rs`): a type-erased match with a uniform, string/index-based surface — `advance()`, `view()`, `legal_labels()`, `apply_human()`, `result_text()`. `TypedMatch` applies one alternating action; `SimultaneousTypedMatch` applies one complete joint turn and emits one atomic event.

The CLI and the browser arcade are two thin frontends over exactly these calls; hidden information is respected everywhere because `view`/narration are viewer-scoped. The full client-side web design (wasm engine, per-game frontends, in-browser tournaments) is in [web/DESIGN.md](web/DESIGN.md).

## Current algorithm/game matrix

|                | chess | 4P chess | othello | connect4 | pente | go | Battlesnake | liars-dice | poker | twentyone | kuhn (test) |
|----------------|:-----:|:--------:|:-------:|:--------:|:-----:|:--:|:-----------:|:----------:|:-----:|:---------:|:-----------:|
| `Cfr` (+ exact exploitability) | — | — | — | — | — | — | — | tiny configs | — | — | ✓ → Nash |
| `Mccfr` / `OsMccfr` | — | — | — | — | — | — | — | OS handles the deep ladder | — | — | ✓ |
| `AlphaBeta` | ✓ | — | ✓ | ✓ | ✓ | — | joint BNS/min-response | — | — | — | — |
| `Mcts` | possible | 4-seat PUCT | possible | possible | possible | ✓ | invalid if sequential | — | — | — | — |
| neural self-play | ✓ PUCT | ✓ 4-seat PUCT | possible | possible | possible | ✓ PUCT | joint logit/maximin | — | — | — | — |
| `Rollout` | possible | possible | possible | possible | possible | possible | possible joint rollout | ✓ | ✓ | possible | — |
| bespoke | — | — | — | — | — | — | MCS/BRS+ bitboard search | belief policy | equity bot | decomposed CFR+ | — |

(The matrix is the two-player-search story. The real-time games also live in the lab; their bots are trained as described below.)

The dashes are honest: tabular CFR can't fit big games, search can't see hidden information, and Go's `GoEval` feeds MCTS/azero rather than alpha-beta. Two durable facts the matrix doesn't show: outcome-sampling MCCFR runs the deep liar's-dice ladder in milliseconds/iteration where external sampling would need astronomically many nodes (~1e41), and CFR+ regret flooring empirically stalls outcome sampling (documented in `solvers/src/os_mccfr.rs`) — which is why liar's dice uses outcome sampling, not the CFR+ variant the perfect-recall games would. Where a learned net is the bot (the `azero` row), the net is trained out-of-tree; see *Learning: the ml tree* below.

## Learning: the ml tree

Some bots are neural nets rather than hand-written evaluators. Training them needs heavyweight, churn-prone machinery (a tensor library, GPUs, long-running self-play); inference needs none of that. The `ml/` tree keeps those two concerns apart, and keeps both off the main workspace's critical path:

- **Training crates are standalone** (their own `[workspace]`, not members of the root one), so the tensor backend (`tch`/libtorch) never touches the lab's `cargo test` or wasm builds. Chess and Go use scalar-value PUCT self-play; four-player chess uses the same search with an absolute four-seat value distribution and a past-checkpoint league. Battlesnake uses fixed-depth joint-action backups (logit equilibrium, duel maximin, or a policy-only ablation) so training preserves simultaneous information. A separate PPO stack trains the real-time slither bot.
- **Inference is torch-free.** A net is exported to a small versioned weight file read by a reference fp32 forward — plain loops, built for correctness and wasm portability. `nn-infer` reads `AZNET1`; `ml/slitherinfer` reads `SLNET1`. Export verification compares the torch and reference forwards before a checkpoint is eligible for deployment.

Alternating neural games share one batched park/resume PUCT implementation. Battlesnake deliberately does not reuse it: its trainer evaluates every root joint action and backs values up through a simultaneous equilibrium solver. Sharing code is subordinate to preserving the game's information structure.
