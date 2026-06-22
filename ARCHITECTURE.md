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
  │ cfr · mccfr ·         │   │ Game impl · Eval ·          │
  │ alpha-beta · rollout  │   │ Determinizer · SearchSpec · │
  │ · mcts · puct/azero   │   │ GameUi (+ bespoke solvers)  │
  │ · exploitability      │   │                             │
  └───────────┬───────────┘   └──────────────┬──────────────┘
              │                              │
           ┌──┴──────────────────────────────┴──┐
           │ game-core    Game · Agent · arena · │
           │ capability traits · Rng             │
           └─────────────────────────────────────┘
```

Dependency rule: `game-core` depends on nothing; `solvers` and `games/*` depend only on `game-core` (games may use `solvers` in dev-dependencies for tests and experiments); `lab` binds everything. Games never depend on solvers at the library level, so adding an algorithm never recompiles a game and vice versa. The two outer tiers attach without leaking inward: the wasm engine wraps `lab`'s serving surface, and the training crates are *standalone* (their own `[workspace]`) so the tensor backend never enters the lab's build — the only thing crossing the boundary is an exported net file, which a torch-free forward reads (see *Learning: the ml tree*).

## The contract: capability traits

A game implements `Game` (rules: chance/decision nodes, legal actions, terminal returns, information-set keys). That alone earns it the arena, CFR/MCCFR (if small enough), and exploitability. Every further power is unlocked by declaring *knowledge*, never by writing an algorithm:

| the game declares | in trait | which unlocks |
|---|---|---|
| a static value estimate | `Eval` | `solvers::AlphaBeta` (negamax, alpha-beta, quiescence, iterative deepening) |
| noisy actions + move ordering | `SearchSpec` | sharp pruning & horizon extension in the same search |
| how to sample hidden info | `Determinizer` | `solvers::Rollout` (determinized Monte-Carlo with common random numbers) |
| per-player view, action labels/parsing, transition narration | `GameUi` | the universal client in `lab` — no game writes a play loop |

Concretely: chess ships piece-square tables (`MaterialEval`, plus the tapered `RichEval` it grew later) and "captures are noisy, MVV-LVA first" (`ChessSpec`) — a few hundred lines of evaluation knowledge and zero lines of search — and receives a full tournament-shaped engine. Liar's Dice ships "bidders plausibly hold the face they bid" (`BidConditioned`) and receives parallel determinized rollouts. Neither contains a line of search machinery.

**Bespoke algorithms are allowed but live with their game.** Twenty-One's round-decomposed CFR+ solver exploits structure (rounds linked only by public hearts) that no generic interface should pretend to capture; it stays in `games/twentyone` and is exposed to the rest of the lab as an ordinary `Agent`.

## Adding things

**A game** (the acid test of the design): implement `Game` + `GameUi`, register one entry in `lab/src/registry.rs`. It immediately gets the arena, the CLI, and (for perfect-information games) alpha-beta the moment you write a ~30-line `Eval`. Nothing else in the repo changes.

**An algorithm**: write it once in `solvers` against `Game` plus whatever capability traits it needs. It immediately runs on every game that has them. If it needs knowledge no trait captures yet, add a trait to `game-core` with a sane default — never reach into a specific game.

## Identity choices worth knowing

- **Actions are indices.** Agents return an index into `legal_actions(state)`, which must be stably ordered per information set. This keeps `Action` types fully game-private, makes tabular methods line up, and gives serving a wire-format for free (index + label). For cross-state identity (killer/history/RAVE tables), `Game::action_id` gives every action a stable u64 — defaulted via its `Debug` form, overridden cheaply by games search cares about.
- **Information sets are u64 keys** (hashes of sufficient statistics). Collision odds at tens of millions of infosets are negligible (~2⁻²⁵); the payoff is flat, fast tables.
- **One randomness contract.** `Agent::act` receives `&mut Rng` — a private, seeded stream for mixed strategies and stochastic search; deterministic agents ignore it. Matches are reproducible from the arena seed, and agents stay `&self` so they can be shared across seats and parallel games.
- **Draws are first-class.** `play` returns the actual utility, `win_rate` scores draws ½, and N-player ties split `win_share` so an all-draw field reads exactly the fair `1/players` — never a phantom win for seat 0.
- **Returns are bounded by `Game::max_return`** (default 1.0). Anything that mixes static evaluations with returns or detects proven wins (MCTS-Solver) keys on that bound instead of assuming the win/loss convention.
- **Measure one change at a time.** Evaluation is win share against a *field* with the hero rotated through seats (fair = 1/players); single runs can be ~2σ lucky (it happened — see `games/liars-dice/examples/ab.rs`).

## Serving: terminal and web

`lab` exposes exactly two serving interfaces, deliberately separated from the terminal client:

1. **The registry** (`lab/src/registry.rs`): `game id + options + bot id → Box<dyn AnyMatch>` — the catalog of what can be played.
2. **`AnyMatch`** (`lab/src/runner.rs`): a type-erased match with a uniform, string/index-based surface — `advance()` (chance + bot moves, narrated), `view()` (the human's information only), `legal_labels()`, `apply_human()`, `result_text()`.

The CLI and the browser arcade are two thin frontends over exactly these calls; hidden information is respected everywhere because `view`/narration are viewer-scoped. The full client-side web design (wasm engine, per-game frontends, in-browser tournaments) is in [web/DESIGN.md](web/DESIGN.md).

## Current algorithm/game matrix

|                | chess | othello | connect4 | pente | go | liars-dice | poker | twentyone | kuhn (test) |
|----------------|:-----:|:-------:|:--------:|:-----:|:--:|:----------:|:-----:|:---------:|:-----------:|
| `Cfr` (+ exact exploitability) | — | — | — | — | — | tiny configs | — | — | ✓ → Nash |
| `Mccfr` / `OsMccfr` | — | — | — | — | — | OS handles the deep ladder | — | — | ✓ |
| `AlphaBeta` | ✓ (the bot) | ✓ (the bot) | ✓ (the bot) | ✓ (the bot) | — (MCTS/azero instead) | — (imperfect info) | — | — | — |
| `Mcts` | possible | possible | possible | possible | ✓ (the bot) | — | — | — | — |
| `azero` (PUCT + self-play net) | ✓ | possible | possible | possible | ✓ | — | — | — | — |
| `Rollout` | possible | possible | possible | possible | possible | ✓ (the bot) | ✓ (a bot) | possible | — |
| bespoke | — | — | — | — | — | belief policy | equity bot (the bot) | decomposed CFR+ (the bot) | — |

(The matrix is the two-player-search story. The single-player and real-time games — 2048, snake, slither — also live in the lab; their bots are MCTS/eval truncation or a learned net, trained as described below.)

The dashes are honest: tabular CFR can't fit big games, search can't see hidden information, and Go's `GoEval` feeds MCTS/azero rather than alpha-beta. Two durable facts the matrix doesn't show: outcome-sampling MCCFR runs the deep liar's-dice ladder in milliseconds/iteration where external sampling would need astronomically many nodes (~1e41), and CFR+ regret flooring empirically stalls outcome sampling (documented in `solvers/src/os_mccfr.rs`) — which is why liar's dice uses outcome sampling, not the CFR+ variant the perfect-recall games would. Where a learned net is the bot (the `azero` row), the net is trained out-of-tree; see *Learning: the ml tree* below.

## Learning: the ml tree

Some bots are neural nets rather than hand-written evaluators. Training them needs heavyweight, churn-prone machinery (a tensor library, GPUs, long-running self-play); inference needs none of that. The `ml/` tree keeps those two concerns apart, and keeps both off the main workspace's critical path:

- **Training crates are standalone** (their own `[workspace]`, not members of the root one), so the tensor backend (`tch`/libtorch) never touches the lab's `cargo test` or wasm builds. One AlphaZero trainer covers the perfect-information net games (chess, go, snake) as per-game binaries over a shared self-play/replay/optimizer/run-dir core; a separate PPO stack trains the real-time slither bot. Both run long, write a run directory, and are driven from the CLI.
- **Inference is torch-free.** A net is exported to a small versioned weight file read by a reference fp32 forward — plain loops, built for correctness and wasm portability. There are two such formats: `nn-infer`'s `AZNET1` for the AlphaZero net games (chess/go/snake) and `ml/slitherinfer`'s `SLNET1` for the PPO slither bot. Each reference forward is the ground truth its exported weights are validated against *and* the path the browser runs through (the WebGPU bots and the CPU fallback both check against it), so a deployed net plays exactly what training measured.

The search itself is not duplicated: there is one PUCT implementation, the batched park/resume `solvers::azero::Search` (generic over `Game` + a policy/value encoder). The CPU harness drives it synchronously, the GPU trainer drives it with batched net forwards, and the browser drives it with WebGPU — same search, three evaluators.
