"""Training hyperparameters and the dynamic-damping schedules (ATARAXOS_SPEC §4).

Every number traces to the reference `RLConfig` defaults (`pyengine/core/rl.py:47-107`)
and the spec table in §4.1/§4.2. The dynamic-damping power-law schedule is
`power_schedule` (`rl.py:756-760`): `clip(coef / (step+1)**decay, floor, ceil)`.
"""

from dataclasses import dataclass

import stratego_nets as S


def power_schedule(coef: float, step: int, decay: float, ceil: float, floor: float) -> float:
    """The reference power-law damping schedule (`rl.py:756-760`).

    `clip(coef / (step+1)**decay, floor, ceil)`. Used for the move LR, the
    move magnet-KL coefficient, and the setup max-entropy temperature.
    """
    x = coef / ((step + 1) ** decay)
    return min(max(x, floor), ceil)


@dataclass
class TrainConfig:
    # ---- self-play ----
    # 64 GB shared box -> stay conservative. The full ~1-day run scales this up
    # (spec §4.1 targets ~1024-2048); the smoke default is small and steady.
    num_envs: int = 512
    # Per-game ply cap before a force-reset (a truncation, not a rules draw).
    # Reference parity: `final_run/train.log`'s `max_num_moves: 4000`. A cap this
    # much shorter (we ran 400 through valrun1) throttles decisive outcomes —
    # cautious self-play can't be resolved by attrition, self-play converges into
    # near-universal draws/timeouts, and the value head collapses toward 0
    # everywhere (see `draw_frac`/`capped_frac` in the training record).
    move_cap: int = 4000
    seed: int = 0
    # Collect this many decision steps before a train pass.
    collect_steps: int = 120
    # Ring capacity per env. CRITICAL: the bridge's `drain_training_batch`
    # re-encodes and returns ALL resident move transitions every call (it never
    # resets the ring), so an over-large capacity re-trains the same stale
    # transitions for many iterations — the value head overfits them, its
    # predictions collapse onto the stored targets, advantages vanish, and
    # advantage filtering then keeps zero rows (move training silently stops).
    # Sizing capacity near `collect_steps` rolls the ring forward ~one fresh
    # iteration per drain (≈ one pass per transition, matching the reference's
    # collect-window-then-reset; at capacity 128 vs collect_steps 120 the
    # overlap re-trains ~8/128 ≈ 6% of transitions a second time — small,
    # known, accepted). It must still exceed a deploy phase (80
    # placements) so move-phase λ-returns within a game stay resident; setup
    # trajectories are accumulated independently of the ring.
    buffer_capacity: int = 128

    # ---- move-RL loss (§4.1, rl.py:547-579) ----
    clip_range: float = 0.2  # PPO epsilon
    vf_coef: float = 1.0  # spec. (vf_coef=0.5 experiment did NOT break the value-collapse stall — fullrun5 still stalled ~iter549 and peaked lower; reverted. The fix needs a target-network value or more data, not this knob.)
    policy_coef: float = 1.0
    kl_coef: float = 0.1  # rev-KL to data policy (constant)
    td_lambda: float = 0.8  # value lambda
    gae_lambda: float = 0.5  # advantage lambda (distinct)
    adv_filt_rate: float = 0.75  # keep top-quantile |adv|
    adv_filt_thresh: float = 0.01  # reference rl.py:64: threshold = max(quantile(|adv|,0.75), 0.01). When nothing exceeds it the move pass legitimately idles that iter (nothing to learn) instead of training on low-|adv| noise.
    # No anti-starve floor (the reference has none). The real fix for "advantages
    # vanish" is training hard enough per iter (move_num_epoch over the full kept set)
    # to keep them alive, not a floor that force-feeds noise and masks the collapse.
    adv_filt_min_keep: int = 0
    max_grad_norm: float = 0.267
    uniform_magnet: bool = True  # flat legal/legal.sum magnet (shipped default)

    # ---- move dynamic damping (§4.1 power schedules) ----
    lr_coef: float = 0.5
    lr_decay: float = 1.1
    lr_ceil: float = 1e-4
    lr_floor: float = 5e-6
    temperature_coef: float = 0.05  # magnet-KL coefficient
    temperature_decay: float = 0.3
    temperature_ceil: float = 0.1
    temperature_floor: float = 0.001  # spec value. (raising to 0.003 over-regularized: entropy pinned ~2.46, weak policy; the adv_filt_thresh=0.001 relative filter is what actually prevents the iter-540 starvation halt, so the magnet floor can stay low and let the policy sharpen)

    # ---- optimizer (AdamW; betas/eps PyTorch defaults) ----
    # MUST stay 0.0 until per-param groups exist: the reference
    # (train_container.py:22-33) forces weight_decay=0 on BIAS params via a
    # second param group; our single-group AdamW would decay biases too.
    # Inert while 0.0; train.py asserts this so tuning it nonzero fails loudly.
    weight_decay: float = 0.0
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    ema_decay: float = 0.999  # move + setup
    bucket: int = 256  # pad effective batch to a multiple of this (MPS shape cache)
    # Move pass minibatches the WHOLE advantage-filtered set (reference rl.py:516-585:
    # many gradient steps/iter, not one over a capped sample). move_batch_size ~ the
    # reference's per-step minibatch (num_envs * adv_filt_rate ~ 400); obs is encoded
    # per minibatch so peak Metal memory stays bounded regardless of the kept-set size.
    move_batch_size: int = 512
    move_num_epoch: int = 1
    # Run the training forward/backward matmuls in bf16 (the loss math still promotes
    # to fp32 where it meets the fp32 advantages/returns, and the optimizer keeps fp32
    # master weights — grads are cast bf16->fp32 before the update). ~1.75x on the
    # matmuls. Off by default until validated past the stability danger zones.
    bf16_train: bool = False

    # ---- setup loop (§4.2, rl.py:616-702) ----
    arr_clip_range: float = 0.2
    arr_policy_coef: float = 1.0
    arr_ent_pred_coef: float = 1.0
    arr_vf_coef: float = 0.5
    arr_kl_coef: float = 0.1
    arr_reg_norm: float = 10.0
    arr_max_grad_norm: float = 0.5
    arr_lr: float = 5e-5  # constant
    arr_batch_size: int = 1024
    arr_num_epoch_per_train: int = 5
    arr_temperature_coef: float = 0.1  # setup max-entropy temperature
    arr_temperature_decay: float = 0.3
    arr_temperature_ceil: float = 1.0
    arr_temperature_floor: float = 0.001

    # ---- stability guard (self-heal + watchdog) ----
    # On a non-finite loss/grad (or a net that goes non-finite after its pass) we
    # revert that net to its last-good in-memory snapshot and scale its LR by
    # `lr_backoff`; a healthy iter nudges the scale back toward 1.0 by `lr_recover`
    # (floor `lr_scale_min`). The watchdog HALTS the run (after checkpointing) if
    # any net stays non-finite or `move/n_kept == 0` for `watchdog_patience`
    # consecutive iters — so a silent multi-thousand-iter freeze is impossible.
    watchdog_patience: int = 5
    lr_backoff: float = 0.5
    lr_recover: float = 1.1
    lr_scale_min: float = 1.0 / 64.0

    # ---- run / infra ----
    iters: int = 1000
    save_every: int = 100
    eval_every: int = 50
    # This is the PERIODIC in-run telemetry eval (train.py spawns eval_ckpt.py as a
    # blocking subprocess every `eval_every` iters — the loop stalls until it
    # returns). It exists to plot a learning curve while training runs, NOT to
    # make a keep/gate decision, so it stays cheap: far fewer games and a much
    # shorter move_cap than a real game needs to reach a decisive result. The
    # GATE decision (is this checkpoint actually good?) is a separate, deliberate
    # offline call to `python -m stratego_trainer.eval_ckpt`, which defaults to
    # the full `move_cap` (see eval_ckpt.py's `--move-cap` default) and should use
    # more games than this to keep gate-decision noise down (~0.04-0.08 win-share
    # spread was measured at n=200, seed-to-seed).
    eval_games: int = 50
    eval_move_cap: int = 1000
    # Sharpen the hero's sampling at eval so a still-exploratory (high-entropy)
    # policy's learned preferences show through against random (1.0 = as-trained).
    eval_temperature: float = 0.25
    work_seconds: float = 0.0  # 0 -> no time budget
    run_name: str = "run"
    runs_root: str = "runs"
    # Optional MLX allocator caps; 0 disables (the default — memory is not the
    # constraint here). Set on a shared box only if you want a hard ceiling.
    mlx_memory_limit_gb: float = 0.0
    mlx_cache_limit_gb: float = 0.0
    # Move/setup net size preset, one of `S.NET_SIZES` ("default" | "mid" | "ref";
    # see stratego_nets/config.py). Chosen once at run-start and never mixed mid-run.
    net_size: str = "default"
    # Optional warm-start checkpoint (e.g. a BC run's output). Loads move+setup
    # net params and EMA shadows only — optimizer state is always fresh, since a
    # BC warm-start uses a different loss/optimizer regime than RL. "" -> from
    # scratch (random init), the reference's own starting point.
    resume_from: str = ""

    # ---- attack-clock curriculum ----
    # The no-attack draw clock the simulator enforces linearly anneals from
    # `clock_start` at iter 0 to `clock_end` (reference parity, 100) by
    # `clock_anneal_iters`, then holds at `clock_end`. A from-scratch net attacks
    # so rarely (see the 2026-07-04 measurement: HeuristicBot 24.5 attacks/100
    # plies vs an iter-100 net's 0.6) that the reference's tight 100-ply clock
    # draws nearly every game before any exploratory attack can be rewarded or
    # punished — a longer early clock gives inexperienced self-play more room to
    # reach a decisive result while it's still learning to attack at all.
    # `clock_start == clock_end` (the default) disables the curriculum outright.
    clock_start: int = 100
    clock_end: int = 100
    clock_anneal_iters: int = 1

    # ---- BC-anchor trust region ----
    # A decaying reverse-KL penalty from the current move policy to a FROZEN copy
    # of the warm-start (BC) checkpoint's policy — a trust region around the known-
    # good attacking behavior, not a pull toward exact teacher play. Only active
    # when `resume_from` is set (no teacher, no anchor). Motivated by valrun3
    # (2026-07-04): BC-init RL hit ws_heur=0.780 by iter100 (already past the BC
    # checkpoint's own 0.549) then collapsed to 0.121 by iter400 as draw_frac
    # climbed 0->0.94 — the policy drifted far enough from the attacking prior to
    # re-enter the passivity basin with no anchor pulling it back. Same
    # power-schedule shape as the magnet (`temperature_*`): coefficient is largest
    # early (when a KL-to-data term of similar scale, 0.1, already regularizes
    # every iter per valrun3's own metrics) and decays toward `anchor_floor` so
    # late training is unconstrained RL — the policy must be free to blow past the
    # teacher (iter100 already did, at 0.780 vs 0.549) not just match it.
    #
    # `anchor_floor` was 0.0 (fully unconstrained late-training) through valrun4;
    # that run improved to ws_heur=0.788 by iter100 (coef 0.1->0.026 over that
    # window) then eroded — a head-to-head confirmed genuine erosion, not just
    # meta-drift: iter100 beat iter684 decisively (0.295 ws) and even tied the
    # raw BC init (0.524 ws), i.e. 684 iters of RL bought nothing. Erosion was
    # already locked in by iter400, where the schedule had decayed to ~0.017 —
    # below the floor here. 0.025 sits at the bottom of the coefficient band
    # that produced iter100's real gain, so the trust region never fully
    # vanishes even arbitrarily late in training (valrun5).
    anchor_coef: float = 0.1
    anchor_decay: float = 0.3
    anchor_ceil: float = 0.1
    anchor_floor: float = 0.025

    def attack_clock(self, step: int) -> int:
        if self.clock_anneal_iters <= 0:
            return self.clock_end
        frac = min(1.0, step / self.clock_anneal_iters)
        return round(self.clock_start + frac * (self.clock_end - self.clock_start))

    def anchor_coef_at(self, step: int) -> float:
        if not self.resume_from:
            return 0.0
        return power_schedule(self.anchor_coef, step, self.anchor_decay,
                              self.anchor_ceil, self.anchor_floor)

    def move_net_config(self) -> S.MoveConfig:
        return S.NET_SIZES[self.net_size][0]

    def setup_net_config(self) -> S.SetupConfig:
        return S.NET_SIZES[self.net_size][1]

    def lr(self, step: int) -> float:
        return power_schedule(self.lr_coef, step, self.lr_decay, self.lr_ceil, self.lr_floor)

    def magnet_coef(self, step: int) -> float:
        return power_schedule(
            self.temperature_coef,
            step,
            self.temperature_decay,
            self.temperature_ceil,
            self.temperature_floor,
        )

    def setup_temperature(self, step: int) -> float:
        return power_schedule(
            self.arr_temperature_coef,
            step,
            self.arr_temperature_decay,
            self.arr_temperature_ceil,
            self.arr_temperature_floor,
        )
