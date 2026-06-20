# slither-ppo

PPO self-play league trainer for the slither encircle bot. It learns to **cut off
and encircle** prey — the strategic behavior the project is about — on top of the
headless [`slither-rl`](../slither-rl) substrate (fast vectorized dynamics, an
egocentric viewport-only semantic-grid observation, discrete turn + boost actions,
and a kill + annealed-encircle-shaping reward).

Standalone crate (empty `[workspace]` table, like `azgo/`): keeps libtorch out of
the repo's main cargo workspace. Build from this directory.

## What's here

- **`net.rs`** — the policy/value net (tch): a 3-conv CNN over the 10×32×32
  semantic grid → flatten + concat the 3 scalars → three heads: turn-bucket logits
  (categorical), a boost logit (Bernoulli), and a scalar value. The two action
  heads factor, so joint log-prob/entropy are sums (agar.io evidence: a small CNN
  over a low-res semantic grid beats MLP/pixels).
- **`ppo.rs`** — PPO following cleanrl's reference: GAE(λ), clipped surrogate,
  value clipping, entropy bonus, per-minibatch advantage normalization, several
  minibatch epochs. Unit-tested against a hand-rolled GAE.
- **`rollout.rs`** — vectorized self-play collector over many parallel arenas
  (rayon-parallel env stepping), learner in seat 0, opponents from the pool, dead
  learners auto-reset mid-rollout into a fixed `T×N` block (cleanrl-style).
- **`opponent.rs`** — the PFSP-lite pool: scripted teachers (the hand-coded
  encircle heuristic, fleeing prey, random) + frozen learner snapshots, sampled by
  near-even-matchup weight. Scripted foes act CPU-side; neural snapshots are
  batched through their net like the learner.
- **`curriculum.rs`** — the stage ladder: learner **oversized vs small prey** →
  **mixed** → **even self-play** (encircling only pays when you're bigger; ramp to
  symmetric league play).
- **`eval.rs`** — the eval panel: greedy learner vs random and vs the heuristic,
  measuring winrate + kills (kills come via cut-off — a foe head hitting the
  learner's body).
- **`main.rs`** — the training loop: rollout → PPO update → anneal the encircle
  shaping (decays by iteration, faster once real kills appear) → advance the
  curriculum → checkpoint (`.ot` + `model.json` sidecar) + `metrics.jsonl`.

## Run

```bash
cargo build --release            # downloads libtorch on first build
export DYLD_LIBRARY_PATH=$(find target -name libtorch_cpu.dylib | head -1 | xargs dirname)

# defaults: 300 iters, 256 arenas, 64 steps, auto device (MPS/CUDA/CPU)
./target/release/slither-ppo iters=300 arenas=256 steps=64 device=mps out=runs/r1

cargo test --release             # GAE, factored log-prob/entropy, obs packing, checkpoint round-trip
```

Args: `iters` `arenas` `steps` `device=cpu|mps|cuda` `out=DIR` `eval-every`
`snapshot-every` `lr` `seed`.

## Does it learn?

Yes. On a 300-iter MPS run (`arenas=256 steps=64`):

- **vs random: winrate → ~0.96** (crushes the floor).
- **vs the heuristic teacher: winrate rises from ~0.05 to ~0.6–0.7** — the learner
  starts *beating* the competent hand-coded encircler it was seeded against.
- entropy falls steadily (≈2.87 → ≈1.5: the policy sharpens), explained variance
  climbs to ≈0.85 (the value head fits), KL and clip-fraction stay in the healthy
  PPO band. Kills come via cut-off (a foe head dying on the learner's body).

The encircle-shaping prior holds at full strength early (so the behavior can
emerge), then anneals as the learner kills on its own — the final policy is learned,
not scripted.
