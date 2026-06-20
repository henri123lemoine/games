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
`eval-games` `snapshot-every` `lr` `seed`.

`compare net=A.ot [net2=B.ot] [games=N] [seed=S]` runs the greedy eval panel vs
the heuristic on common seeds — the honest A/B for "is this net actually better".

## Does it learn?

Yes, and it now **stays** learned. On a 600-iter MPS run (`arenas=256 steps=64`):

- **vs random: winrate → ~0.96** (crushes the floor).
- **vs the heuristic teacher: winrate rises to ~0.88 and plateaus there** through
  deep even-self-play — the learner *beats* the competent hand-coded encircler it
  was seeded against, and holds the level instead of regressing.
- entropy falls but is held off the floor (the adaptive entropy coef), explained
  variance ≈0.9, KL/clip in the healthy band, LR decays linearly to a 10% floor.
  Kills come via cut-off (a foe head dying on the learner's body).

The encircle-shaping prior holds at full strength early (so the behavior can
emerge), then anneals as the learner kills on its own — the final policy is learned,
not scripted.

### Keeping it from regressing

An earlier long run *peaked* at ~0.88 vs the heuristic then **collapsed to ~0.66**:
once even-self-play started, pure PFSP dropped the heuristic from the pool (its
`p*(1-p)` weight vanishes as the learner beats it), so the learner trained only
against its own snapshots and forgot how to beat the teacher. Four changes fix it:

1. **Eval-gated keep-best** — winrate-vs-heuristic is tracked every eval and the
   peak weights are persisted to `best.ot` automatically (no manual grab).
2. **Heuristic seat floor** — a fixed fraction of even-self-play seats is always the
   heuristic (`HEURISTIC_FLOOR`), so the gating opponent never leaves the pool.
3. **LR decay** to a small floor and an **adaptive entropy coef** with a floor —
   the policy settles at the plateau instead of over-sharpening into the brittle
   self-play-only mode.
4. **Reward** weights a kill more decisively (`KILL_FLAT` + a larger length bonus)
   so conversion is positive-EV, not only food-farming.

Result: stable ~0.88 (best 0.91) all the way to iter 600 — no regression. Equal-
footing *kill conversion* improved only marginally (≈0.12→0.14 kills/game): turning
a closed-in prey into a kill on equal top speed is the genuinely hard part the
blueprint flags, and survival/growth dominance is still the easier path to the win.
