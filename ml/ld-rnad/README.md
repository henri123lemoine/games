# ld-rnad

Standalone wrapper for the Liar's Dice regularized actor-critic contender.

The training implementation lives in `liars_dice::pg_train`; this crate gives
the roadmap's `ml/ld-rnad` trainer a dedicated command without adding a new
member to the root workspace. Checkpoints are the same `solvers::azero::Mlp`
artifact consumed by:

```bash
cargo run --release -p liars-dice --example tournament -- \
  rnads=rnad:runs/ld_rnad/best.bin agents=rollout,rnads
```

Quick smoke:

```bash
cargo run --release --manifest-path ml/ld-rnad/Cargo.toml -- \
  players=2 dice=1 faces=2 mixed=1 min_players=2 max_players=3 \
  min_dice=1 max_dice=2 min_faces=2 max_faces=3 \
  iters=2 episodes_per_iter=16 hidden=16 eval_games=0 eval_exploitability=0 \
  outdir=runs/ld_rnad_smoke
```

5p5d6f target with mixed-family training:

```bash
cargo run --release --manifest-path ml/ld-rnad/Cargo.toml -- \
  players=5 dice=5 faces=6 mixed=1 max_players=5 max_dice=8 max_faces=6 \
  iters=400 episodes_per_iter=512 hidden=256 eval_games=200 outdir=runs/ld_rnad
```

The run writes `metrics.jsonl`, `train.log`, `ckpt.bin`, durable `ckpt_N.bin`
curve points, and `best.bin`. Use `games/liars-dice/examples/curve_report.rs`
to emit tournament roster fragments from the metrics. It also logs a default-on
tiny exact exploitability probe as a sanity diagnostic; pass
`eval_exploitability=0` for plumbing-only smokes.
