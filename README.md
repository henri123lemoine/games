# Games lab

Game-playing algorithms (CFR variants, alpha-beta search, MCTS, determinized Monte-Carlo rollouts, PUCT/AlphaZero) written **once** against a shared `Game` trait, applied to many games — the OpenSpiel idea, scoped to a personal lab. Games contribute only their rules and knowledge (an evaluator, a determinizer, a UI surface); they never contain algorithm code. See [ARCHITECTURE.md](ARCHITECTURE.md); the in-browser arcade (everything compiled to wasm, per-game frontends) is designed in [web/DESIGN.md](web/DESIGN.md).

```
game-core/           foundations: Game trait, Agent, capability traits
                     (Eval, Determinizer, SearchSpec, GameUi), match arena
solvers/             the algorithms, generic over any game with the right
                     capabilities: cfr, mccfr, os-mccfr, exploitability,
                     alpha-beta, MCTS, determinized rollout, and the
                     PUCT/AlphaZero self-play search
games/               one crate per game — perfect-information board games
                     (chess, othello, connect4, pente, go), N-player imperfect
                     info (liars-dice, poker, twentyone), and single-player /
                     real-time (2048, snake). Each ships only rules + knowledge;
                     its bot is a generic solver (alpha-beta / MCTS / rollout /
                     azero) or, where the structure demands it, a bespoke
                     in-crate solver exposed as an ordinary Agent
lab/                 registry of games & bots, type-erased matches, and the
                     one terminal client for every game
web/                 the same lab compiled to WebAssembly: engine bindings +
                     a browser arcade with per-game frontends, plus standalone
                     wasm games (slither, DOOM) that share the trained nets
                     (see web/README.md)
ml/                  the training side, deliberately standalone (own workspace,
                     libtorch off the main build): an AlphaZero trainer for the
                     net games and a PPO self-play stack for the real-time bot,
                     plus the torch-free inference both the export-verify and
                     the browser run through
```

## Play anything

```bash
cargo run --release -p lab -- list
cargo run --release -p lab -- play chess depth=6
cargo run --release -p lab -- play chess bot=azero            # the self-play net
cargo run --release -p lab -- play go size=9 sims=6000
cargo run --release -p lab -- play othello
cargo run --release -p lab -- play connect4
cargo run --release -p lab -- play pente size=13 depth=4
cargo run --release -p lab -- play liars-dice players=5 dice=5 rollouts=1000
cargo run --release -p lab -- play poker players=6 samples=2000
cargo run --release -p lab -- play twentyone hearts=6 iters=100000
cargo run --release -p lab -- play 2048                       # or watch: bot=mcts-eval
cargo run --release -p lab -- play snake                      # 1v1; the AlphaZero seat is browser-only
```

One client drives every game: menus by number, or game-native input (`e2e4`, `open 2x4`, `d`/`s`). Hidden information is viewer-scoped throughout.

Or in a browser — the whole lab compiles to wasm:

```bash
wasm-pack build web/engine --target web --out-dir pkg
cd web/app && npm install && npm run dev
```

## Develop

```bash
cargo test --release        # perft suite, Kuhn→Nash, rules invariants, search
cargo clippy --release --all-targets
```

Research harnesses live as examples in each game crate (`liars-dice`: `evaluate`, `league`, `rollout_eval`, `ab`, `exploitability`; `twentyone`: `solve`).

## Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — the design: layering, the capability-trait contract, identity choices, the algorithm/game matrix. Read this before restructuring anything.
- [BENCHMARKS.md](BENCHMARKS.md) — cross-game strength results (SPRT/Elo); a dated snapshot of one benchmark round.
- [web/README.md](web/README.md) — build, run, and deploy the browser arcade. [web/DESIGN.md](web/DESIGN.md) — its design contract.
- Per-game knowledge: [game-core/README.md](game-core/README.md), [games/liars-dice/README.md](games/liars-dice/README.md), [games/twentyone/README.md](games/twentyone/README.md) (+ [BAKEOFF.md](games/twentyone/BAKEOFF.md), the Twenty-One technique shoot-out).
