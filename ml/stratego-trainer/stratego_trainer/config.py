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
    move_cap: int = 400  # defensive per-game ply cap before force-reset
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
    # collect-window-then-reset). It must still exceed a deploy phase (80
    # placements) so move-phase λ-returns within a game stay resident; setup
    # trajectories are accumulated independently of the ring.
    buffer_capacity: int = 128

    # ---- move-RL loss (§4.1, rl.py:547-579) ----
    clip_range: float = 0.2  # PPO epsilon
    vf_coef: float = 1.0
    policy_coef: float = 1.0
    kl_coef: float = 0.1  # rev-KL to data policy (constant)
    td_lambda: float = 0.8  # value lambda
    gae_lambda: float = 0.5  # advantage lambda (distinct)
    adv_filt_rate: float = 0.75  # keep top-quantile |adv|
    adv_filt_thresh: float = 0.01  # abs floor
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
    temperature_floor: float = 0.001

    # ---- optimizer (AdamW; betas/eps PyTorch defaults) ----
    weight_decay: float = 0.0
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    ema_decay: float = 0.999  # move + setup
    bucket: int = 256  # pad effective batch to a multiple of this (MPS shape cache)
    max_train_batch: int = 2048  # cap kept rows per move pass (bounds peak Metal mem)

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

    # ---- run / infra ----
    iters: int = 1000
    save_every: int = 100
    eval_every: int = 50
    eval_games: int = 200  # win share has ~2sigma noise; average enough games
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

    def move_net_config(self) -> S.MoveConfig:
        return S.MoveConfig()

    def setup_net_config(self) -> S.SetupConfig:
        return S.SetupConfig()

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
