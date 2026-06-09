# Games lab

Algorithms for playing games — CFR variants, belief agents, Monte-Carlo rollout
search, alpha-beta — applied to multiple games through one shared `Game` trait.

```
cfr-core/            the algorithms — a Game trait, CFR+/MCCFR, exact
                     exploitability, a game-agnostic arena (see cfr-core/README)
games/
  liars-dice/        N-player × D-dice × F-face Liar's Dice + strong belief and
                     Monte-Carlo-rollout agents (see games/liars-dice/README)
  chess/             chess: perft-validated move generation + alpha-beta agent
  twentyone/         Twenty-One: engine + fast decomposed CFR+ solver + Game
                     adapter (the repo's original project; see its README)
```

```bash
cargo test --release                                      # everything
cargo run --release -p liars-dice --example play 5 5 6    # play Liar's Dice
cargo run --release -p chess --example play               # play chess
cargo run --release -p twentyone --example play           # play Twenty-One
```

Each game's README has its rules, agents, and measured results; `cfr-core`'s has
the algorithm details.
