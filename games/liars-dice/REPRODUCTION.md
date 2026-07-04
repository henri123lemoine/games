# Liar's Dice Reproduction Standard

This file defines the bar for claiming a faithful reproduction of a neural
imperfect-information-game paper in this crate. Exploratory bakeoffs, local
controls, and project-game improvements are useful, but they are not paper
reproductions unless they pass this checklist.

## Claim Levels

- `paper-faithful`: the benchmark game, algorithm, run manifest, checkpoint
  selection, and evaluation match the primary paper or official code.
- `paper-adapted`: the implementation intentionally changes the game, player
  count, solution concept, architecture, budget, or evaluation. The adaptation
  must be stated in the manifest and report.
- `local baseline`: useful project evidence only. PPO, deploy ReBeL,
  round-subgame DeepCFR, and actor-critic/RNAD-style controls are in this
  bucket until a paper-specific protocol says otherwise.

## Non-Negotiable Gates

1. Primary sources and official code, when available, are recorded in a
   paper-specific spec.
2. The benchmark environment is locked and separate from the project game when
   the project game differs from the paper. The current project game has
   multi-round dice loss, `CallExact`, relative raises, forced opening state,
   no wild-face rule, and a `max_rounds` cap; those are not the ReBeL/Deep CFR
   paper benchmark rules.
3. Configs fail closed. Unknown arguments, malformed `key=value` pairs, invalid
   numbers, invalid booleans, and unsupported devices abort before training.
4. Long runs write a manifest with git commit, dirty diff hash, untracked input
   list, command, env, features, toolchain, host/thread settings, seeds, config
   hash, artifact hashes, and command DAG.
5. Resume restores model, optimizer, replay/reservoir buffers, RNG state,
   counters, best state, metric cursor, and config hash. Weight-only resume is
   a fresh run seeded from old weights, not a true resume.
6. Evaluation is split-aware: training monitor, selection, and final held-out
   evidence are distinct. Final-test data cannot select checkpoints.
7. Exact exploitability or NashConv is used where tractable. Larger games need
   predeclared sampled best-response estimates with confidence intervals.
   Multiplayer project-game claims need an explicit multiplayer solution
   concept or empirical-game-analysis protocol.

## Current Local Status

- `rebel::standard::StandardLiarsDice` is the closest paper-rules substrate:
  two-player, single-round, strictly increasing bids, liar call only, top face
  wild. It still needs paper-config specs, official-code parity checks, and a
  strict runner before long ReBeL reproduction compute.
- `rebel::{cfr,selfplay,value_net}` contains recognizable ReBeL primitives.
  `rebel::{adapter,deploy,deploy_train,agent}` is project-specific and cannot
  be used for faithful ReBeL claims without an explicit adaptation label.
- `solvers/src/deepcfr.rs` is a credible Deep CFR primitive. The
  `games/liars-dice/src/deepcfr.rs` wrapper is a local round-subgame method, not
  a faithful Deep CFR reproduction.
- `pg_train.rs`, `examples/rnad_train.rs`, and `ml/ld-rnad` are local
  actor-critic controls. They are not faithful R-NaD, DeepNash, NeuRD, or MMD.
- `ml/ppo-core` implements PPO mechanics, but `ml/ld-ppo` is only a local PPO
  baseline until the selected policy-gradient benchmark protocol is reproduced.

## Initial Paper Set

- ReBeL: https://arxiv.org/abs/2007.13544
- Deep CFR: https://arxiv.org/abs/1811.00164
- Single Deep CFR: https://arxiv.org/abs/1901.07621
- NFSP: https://arxiv.org/abs/1603.01121
- DREAM: https://arxiv.org/abs/2006.10410
- R-NaD: https://arxiv.org/abs/2002.08456
- DeepNash: https://www.science.org/doi/10.1126/science.add4679
- Student of Games: https://arxiv.org/abs/2112.03178
- Policy-gradient benchmark suite: https://arxiv.org/abs/2502.08938
- Predictive/discounted Deep CFR family: https://arxiv.org/html/2511.08174v1

This list is a starting matrix, not a claim that the repo implements those
papers. Each paper needs its own spec and reproduction harness before compute.
