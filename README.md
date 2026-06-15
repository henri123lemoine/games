# Games lab

Game-playing algorithms (CFR variants, alpha-beta search, determinized Monte-Carlo rollouts) written **once** against a shared `Game` trait, applied to many games — the OpenSpiel idea, scoped to a personal lab. Games contribute only their rules and knowledge (an evaluator, a determinizer, a UI surface); they never contain algorithm code. See [ARCHITECTURE.md](ARCHITECTURE.md); the in-browser arcade (everything compiled to wasm, per-game frontends) is designed in [web/DESIGN.md](web/DESIGN.md).

```
game-core/           foundations: Game trait, Agent, capability traits
                     (Eval, Determinizer, SearchSpec, GameUi), match arena
solvers/             the algorithms, generic over any game with the right
                     capabilities: cfr, mccfr, os-mccfr, exploitability,
                     alpha-beta, MCTS, determinized rollout, and an
                     AlphaZero-style self-play learner (PUCT + pure-Rust net)
games/
  chess/             perft-validated rules + eval/search knowledge + net encoder
  othello/           weighted-square eval; bot = generic alpha-beta
  connect4/          windowed eval; bot = generic alpha-beta
  go/                9x9+ area scoring; bot = generic MCTS
  liars-dice/        N-player Liar's Dice + belief policy + determinization
  twentyone/         Twenty-One + its bespoke decomposed CFR+ solver
lab/                 registry of games & bots, type-erased matches, and the
                     one terminal client for every game
web/                 the same lab compiled to WebAssembly: engine bindings +
                     a browser arcade with per-game frontends and live bot
                     tournaments, all client-side (see web/README.md)
```

## Play anything

```bash
cargo run --release -p lab -- list
cargo run --release -p lab -- play chess depth=6
cargo run --release -p lab -- play chess bot=azero            # the self-play net
cargo run --release -p lab -- play go size=9 sims=6000
cargo run --release -p lab -- play othello
cargo run --release -p lab -- play connect4
cargo run --release -p lab -- play liars-dice players=5 dice=5 rollouts=1000
cargo run --release -p lab -- play twentyone hearts=6 iters=100000
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
