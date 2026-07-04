# ld-ppo

Standalone PPO self-play trainer for the Liar's Dice bake-off roster.

The PPO math uses `ml/ppo-core` (GAE, clipped surrogate, value clipping,
entropy, minibatches). This crate owns only the Liar's Dice rollout adapter and
exports checkpoints in the same `solvers::azero::Mlp` format used by Deep CFR,
distillation, and R-NaD, so tournament can load them without a torch dependency.

Quick smoke:

```bash
cargo run --release --manifest-path ml/ld-ppo/Cargo.toml -- \
  device=cpu players=2 dice=1 faces=2 mixed=0 iters=1 actors=2 steps=4 \
  hidden=16 epochs=1 minibatches=1 eval_games=0 eval_exploitability=0 \
  keep_checkpoints=0 outdir=runs/ld_ppo_smoke
```

5p5d6f target with mixed-family training:

```bash
cargo run --release --manifest-path ml/ld-ppo/Cargo.toml -- \
  players=5 dice=5 faces=6 mixed=1 max_players=5 max_dice=8 max_faces=6 \
  iters=400 actors=64 steps=64 hidden=256 eval_games=200 outdir=runs/ld_ppo
```

Evaluate the exported checkpoint:

```bash
cargo run --release -p liars-dice --example tournament -- \
  agents=rollout,ppo ppo=runs/ld_ppo/best.bin
```

The run writes `metrics.jsonl`, `train.log`, `ckpt.bin`, durable `ckpt_N.bin`
curve points, and `best.bin`. It logs wall time, PPO health metrics, optional
field win-shares, and a default-on tiny exact exploitability canary; pass
`eval_exploitability=0` for plumbing-only smokes.
