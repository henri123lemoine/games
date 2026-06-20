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
`eval-games` `snapshot-every` `lr` `seed` `init=CKPT.ot`.

`init=CKPT.ot` warm-starts the learner from a checkpoint and skips the early
oversized/mixed curriculum (it already plays even-size) — a fine-tune leg, e.g.
retuning a strong net after a dynamics change.

`compare net=A.ot [net2=B.ot] [games=N] [seed=S]` runs the greedy eval panel vs
the heuristic on common seeds — the honest A/B for "is this net actually better".
It reports each net at the `[SYMMETRIC/deploy]` config (every worm `START_LENGTH`,
no learner head-start — the real win-share a human faces) AND the old `[favorable]`
config (learner oversized), so the gap between "looks good" and "real" is visible.
Per config: overall `win`, the `kill-win` subset (games with ≥1 kill — winning by
cutting a foe off, not out-growing), and kills/deaths per game. Keep-best gates on
the symmetric number.

### Deploy: export + the parity gate

`export [net=CKPT.ot] [out=PATH]` dumps the checkpoint to the browser `SLNET1`
format. It is a **hard release gate**: after writing, it runs the `slitherinfer`
torch-free reference forward against the tch forward over 64 random inputs and, if
any head deviates by ≥ `PARITY_TOL` (1e-3), it **deletes the blob and exits
non-zero** — so `export` can never leave an unverified blob deployed (the
train↔deploy parity contract). `verify-export [net=][export=]` re-checks an
existing blob the same way. (Actual deviation on a good export is ~1e-5.)

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

Result: stable ~0.88 (best 0.91) all the way to iter 600 — no regression. (Those
numbers are on the *old* dynamics, before the mass-conservation fixes below.)

### Conservative dynamics + pushing kill conversion

The dynamics were then made conservative (boost shed-rate tied to its drain so a
worm can't self-feed; pellet density lowered to 250 + a slow trickle so a grazed
spot stays depleted). This makes the game much harder — you can no longer out-grow
the field by circling in regenerating food — so the heuristic became far tougher:
the old ~0.88 net drops to ~0.24 winrate on the conservative game. Crucially, on
these dynamics **winning *is* killing**: with food sparse, the out-grow path is
gone, so `win ≈ kill-win` and the only way to beat the heuristic is to cut it off.

To make the net a decisive encircler rather than a survivor, two levers (the user
wants it to *encircle*, not win on reflexes):

- **Close-encounter curriculum** — `CLOSE_ENCOUNTER_FRAC` of even-self-play arenas
  spawn as a tight equal-size cluster pinned against a wall (a cut-off is on the
  table and there's a wall to pin against), the rest scattered to avoid overfitting.
- **Kill-aware keep-best** — eval tracks `kill_winrate`; keep-best gates on
  `winrate + KILL_WIN_WEIGHT*kill_winrate` with a `WINRATE_SLACK` floor, so the
  kept net is a better killer *and* not worse overall.

Training fresh on the conservative game stalls (learning to kill from random init
is hard when food is too sparse to coast on), so each leg is a **warm-start
fine-tune** of the previous net (`init=`).

### Make defeating an opponent the headline reward

The first fine-tune still mostly *out-competed* — it out-grew a naive player rather
than relentlessly hunting it. The user wants the kill to be the fun part. So the
reward was re-weighted to make a kill the unambiguous top event: `KILL_FLAT`
1.5→5.0 (a kill now pays ≈7.7 — well above a whole life's growth and above the
death penalty), `LENGTH_DELTA_SCALE` 0.08→0.04 (farming clearly secondary), and
`BOOST_COST` 0.008→0.02 (the bugged mass-creating dynamics had trained in a
boost-spam habit; this breaks it), with the death penalty held at −6 so it stays
fear of dying, not recklessness.

The kill-reward fine-tune does hunt more — but a measurement bug was inflating its
headline. The eval gave the *learner* a head-start (`seat0_length START_LENGTH+30`,
small opponents), so "winrate ≈0.40" was a favorable setup, ~2.7× the real
deployment number. The browser is **symmetric** (every worm `START_LENGTH`, no
head-start, vs a human), where the true strength is ~0.14. The fix: eval (and the
keep-best gate) now run at the symmetric deployment config; `compare` reports both.

Gating on the honest symmetric number, successive warm-start fine-tunes lift the
real deployment strength. Two more changes pushed it further: the even-self-play
training world was made **symmetric too** (`START_LENGTH`, no head-start — the
training distribution should *be* the deployment distribution, not a softer one),
and the close-encounter practice cluster was made **equal-size** (worms packed
against a wall at `START_LENGTH`, so the cut-off is on the table without a size
advantage). The lineage on the honest symmetric/deploy metric (512 games × 4 seeds
vs the heuristic): ft1 ≈0.12 → kill1 ≈0.15 → sym1 ≈0.19 → **sym2 ≈0.22 win,
≈0.22 kills/game** — each leg beats the prior on every seed, now solidly above fair
share (1/6 ≈ 0.167). Absolute numbers are modest because the symmetric game against
a competent encircler is genuinely hard, but this is the real win-share a human
faces, optimized directly rather than flattered by a head-start — and the symmetric-
matched + shorter (200-iter) run no longer regresses in the back half (the earlier
legs peaked then degraded; this one holds its plateau). (Value loss briefly spikes
on the bigger kill rewards at warm-start, then settles — watched, no anneal needed.)
