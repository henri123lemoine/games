"""The Ataraxos move-RL + co-trained setup self-play training loop in MLX.

Faithful to ATARAXOS_SPEC §4.1 (move-RL) and §4.2 (setup), driven by the verified
Rust sim via the `stratego_sim` bridge. Each iteration:

  1. Self-play: collect ragged per-phase decisions, run the move net on move-phase
     envs and the setup net on deploy-phase envs (reducing the value head
     softmax(W/L/D)@[-1,0,1] to a scalar), commit (the verified Rust sampler).
  2. Drain the move-RL arrays (λ-returns / GAE done sim-side) and the completed
     setup trajectories (full 40 placements + the game's MC outcome).
  3. Train ONE move pass (advantage-filtered PPO-clip + magnet-KL + value-CE +
     data-KL, dynamic-damping LR + magnet schedule) and 5 setup epochs (pure-MC
     PPO-clip + value-CE + conditional-entropy MSE + data-KL).
  4. EMA update both nets; periodically checkpoint, eval (win share vs random and
     vs a frozen earlier EMA), and log to metrics.jsonl.

CLI:
  python -m stratego_trainer.train --envs 1024 --iters 400 --run-name smoke
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

import stratego_nets as S
import stratego_sim as sim

from .config import TrainConfig
from .move_loss import advantage_filter_mask, move_loss_and_stats
from .rundir import RunDir
from .setup_loss import PIECE_COUNTS as PIECE_COUNTS_T
from .setup_loss import setup_baseline, setup_loss_and_stats

CATS = np.array(S.spec.CATEGORICAL_AGGREGATION, dtype=np.float32)  # [-1,0,1]


def _all_finite(tree):
    """True iff every leaf array in the tree is all-finite (no NaN/inf)."""
    return all(bool(mx.all(mx.isfinite(v))) for _, v in tree_flatten(tree))


def _isfin(x):
    mx.eval(x)
    return bool(mx.all(mx.isfinite(x)))


def _first_bad(cands):
    """Name of the first (stage, tensor) whose tensor is non-finite, else '?'."""
    for name, v in cands:
        if not _isfin(v):
            return name
    return "?"


def _move_nan_stage(net, batch, stats):
    """Localize the first non-finite stage of a move update (forward -> inputs ->
    loss terms). Only called when a pass already detected a non-finite update, so the
    freeze cause is named in the metrics instead of being a silent skip."""
    out = net(batch["obs"], legal_mask=batch["legal"])
    return _first_bad([
        ("fwd/move_logits", out["move_logits"]), ("fwd/value_logp", out["value_logp"]),
        ("in/advantage", batch["advantage"]), ("in/old_log_prob", batch["old_log_prob"]),
        ("in/data_log_prob", batch["data_log_prob"]), ("in/ret", batch["ret"]),
        ("loss/policy", stats["policy_loss"]), ("loss/value", stats["value_loss"]),
        ("loss/kl", stats["kl_loss"]), ("loss/magnet", stats["magnet_kl"]),
        ("loss/entropy", stats["entropy"]),
    ])


def _setup_nan_stage(net, batch, stats):
    """Localize the first non-finite stage of a setup update (forward -> inputs ->
    loss terms)."""
    out = net(batch["seq"], PIECE_COUNTS_T)
    return _first_bad([
        ("fwd/logits", out["logits"]), ("fwd/value", out["value"]), ("fwd/ent_pred", out["ent_pred"]),
        ("in/reg_adv", batch["reg_adv"]), ("in/reg_returns", batch["reg_returns"]),
        ("in/old_value_scalar", batch["old_value_scalar"]), ("in/outcome", batch["outcome"]),
        ("loss/policy", stats["policy_loss"]), ("loss/value", stats["value_loss"]),
        ("loss/entropy", stats["entropy_loss"]), ("loss/kl", stats["kl_loss"]),
    ])


def _snapshot(net, opt):
    """Materialized copy of a net's params + optimizer state (the last-good state)."""
    snap = {
        "params": tree_map(lambda p: mx.array(p), net.parameters()),
        "opt": tree_map(lambda p: mx.array(p), opt.state),
    }
    mx.eval(snap)
    return snap


def _restore(net, opt, snap):
    """Roll a net + optimizer back to a snapshot (in place)."""
    net.update(tree_map(lambda p: mx.array(p), snap["params"]))
    opt.state = tree_map(lambda p: mx.array(p), snap["opt"])
    mx.eval(net.parameters(), opt.state)


def _first_nonfinite(tree):
    """Name of the first non-finite leaf in a param/opt tree, else ''."""
    for nm, v in tree_flatten(tree):
        if not _isfin(v):
            return nm
    return ""


def _self_heal(net, opt, ema, snap, pass_nan, applied, lr, lr_scale, cfg):
    """Keep-best-style recovery for one net after its train pass.

    If the pass hit a non-finite update or the net went non-finite, revert it to
    its last-good snapshot and scale its LR down by `lr_backoff` (so it retries on
    fresh data at a gentler step). Otherwise fold the EMA, refresh the snapshot, and
    nudge the LR scale back toward 1.0. Returns (snapshot, lr_scale, nan, stage) where
    `stage` localizes a params/opt-state blow-up that survived the finite loss+grad
    check (e.g. `param:setup.ent_out.weight` or `opt:move.value_head...`)."""
    param_bad = _first_nonfinite(net.parameters())
    nan = pass_nan or bool(param_bad)
    if nan:
        stage = ""
        if not pass_nan:
            opt_bad = _first_nonfinite(opt.state)
            stage = f"param:{param_bad}" if param_bad else f"opt:{opt_bad}"
        _restore(net, opt, snap)
        opt.learning_rate = lr
        return snap, max(lr_scale * cfg.lr_backoff, cfg.lr_scale_min), True, stage
    if applied:
        ema.update(net)
        mx.eval(ema.shadow)
    return _snapshot(net, opt), min(lr_scale * cfg.lr_recover, 1.0), False, ""


def _bucket(n, mult):
    """Smallest multiple of `mult` that is >= n (the MPS shape-cache fix)."""
    if mult <= 1 or n == 0:
        return n
    return ((n + mult - 1) // mult) * mult


# Finite floors for the logits/values fed back into the verified sim. The net's
# illegal slots are finfo.min (a large *finite* negative the sim's softmax reads as
# 0); we only need to scrub genuine inf/NaN so a single pathological forward can
# never poison the sim's stored transitions (which both nets then drain).
LOGIT_NEG = -1e30
LOGIT_POS = 1e4


def _sanitize_logits(a):
    return np.nan_to_num(a, nan=LOGIT_NEG, posinf=LOGIT_POS, neginf=LOGIT_NEG).astype(np.float32)


def _stable_log_softmax(logits_np, axis=-1):
    """Numerically stable log-softmax in numpy (subtract max before exp so a large
    value logit can't overflow `exp` to inf and then inf-inf -> NaN)."""
    m = logits_np.max(axis=axis, keepdims=True)
    shifted = logits_np - m
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def _nonfinite(a):
    """Count of non-finite (NaN/inf) entries in `a` (the illegal finfo.min fill is a
    finite value, so it is never counted -- only genuine corruption is)."""
    return int(a.size - int(np.isfinite(a).sum())) if a.size else 0


def _move_forward(net, obs_np, legal_np):
    """Net forward on a move-phase batch -> (logits_np, scalar_values_np, n_scrub)."""
    if obs_np.shape[0] == 0:
        return (np.zeros((0, sim.N_ACTION), np.float32), np.zeros(0, np.float32), 0)
    out = net(mx.array(obs_np), legal_mask=mx.array(legal_np))
    raw_logits = np.array(out["move_logits"].astype(mx.float32))
    vlogp = np.array(out["value_logp"])  # already a stable log-softmax (net side)
    raw_vals = (np.exp(vlogp) * CATS).sum(-1)
    scrub = _nonfinite(raw_logits) + _nonfinite(raw_vals)
    logits = _sanitize_logits(raw_logits)
    vals = np.clip(np.nan_to_num(raw_vals, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0).astype(np.float32)
    return logits, vals, scrub


def _setup_forward(net, obs_np):
    """Setup net forward on a deploy-phase batch -> (logits_np(B,14), scalar_values, n_scrub)."""
    if obs_np.shape[0] == 0:
        return (np.zeros((0, sim.DEPLOY_WIDTH), np.float32), np.zeros(0, np.float32), 0)
    pc = mx.array(list(S.spec.CLASSIC_PIECE_COUNTS), dtype=mx.float32)
    out = net(mx.array(obs_np), pc)
    n_placed = obs_np.reshape(obs_np.shape[0], 40, 14).sum(axis=(1, 2)).astype(int)
    slot = np.clip(n_placed, 0, 39)
    raw_logits = np.array(out["logits"].astype(mx.float32))  # (B,40,14)
    scrub = _nonfinite(raw_logits)
    all_logits = _sanitize_logits(raw_logits)
    logits = all_logits[np.arange(obs_np.shape[0]), slot]
    vlogp = _stable_log_softmax(np.array(out["value"].astype(mx.float32)), axis=-1)
    vals = np.clip((np.exp(vlogp[np.arange(obs_np.shape[0]), slot]) * CATS).sum(-1), -1.0, 1.0)
    return logits, vals.astype(np.float32), scrub


def collect_iter(s, move_net, setup_net, steps):
    """Run `steps` self-play decision steps.

    Returns (env_steps, n_terminals, reward_pl0_sum, n_scrub) where n_scrub counts the
    non-finite net outputs scrubbed before they could poison the sim's stored transitions.
    """
    n_term = 0
    rsum = 0.0
    scrub = 0
    for _ in range(steps):
        b = s.collect()
        m_logits, m_vals, m_scrub = _move_forward(move_net, b["move_obs"], b["move_legal"])
        d_logits, d_vals, d_scrub = _setup_forward(setup_net, b["deploy_obs"])
        scrub += m_scrub + d_scrub
        out = s.commit(m_logits, m_vals, d_logits, d_vals)
        n_term += int(out["terminal"].sum())
        rsum += float(out["reward_pl0"].sum())
    return steps * s.num_envs, n_term, rsum, scrub


def train_move_pass(move_net, opt, data, magnet_coef, lr, cfg):
    """One advantage-filtered PPO pass over the drained move-RL transitions.

    Returns `(stats, nan)`; `nan` is True iff the update was non-finite (skipped).
    """
    n = data["action"].shape[0]
    keep, n_threshold = advantage_filter_mask(data["advantage"], cfg.adv_filt_rate,
                                              cfg.adv_filt_thresh, cfg.adv_filt_min_keep)
    # `n_threshold` is the REAL spec-filter keep (pre anti-starve floor): the watchdog
    # watches it, not the floored `n_kept`, so advantage starvation (full1's silent
    # freeze, no NaN) still trips the alarm.
    # Always surface the attempt's counts so move training is visible every
    # iteration even when nothing passed the advantage filter (would otherwise
    # look like move training silently stopped).
    if n == 0 or keep.sum() == 0:
        return {"move/n_kept": int(keep.sum()), "move/n_threshold_keep": n_threshold,
                "move/n_total": int(n), "move/skipped": True}, False

    opt.learning_rate = lr
    idx = np.nonzero(keep)[0]
    # Bound the per-pass batch: a large kept set would blow up peak Metal memory
    # (obs is 92*F floats/row) and can trigger a GPU page-fault on a shared box.
    # Subsample the kept rows to at most `max_train_batch` before bucketing.
    if idx.size > cfg.max_train_batch:
        idx = np.random.choice(idx, cfg.max_train_batch, replace=False)
    # Pad to a bucket multiple by repeating filtered rows (keeps MPS shape stable;
    # repeated rows only rescale the mean, which advantage filtering already does).
    target = _bucket(idx.size, cfg.bucket)
    if target > idx.size:
        pad = np.random.choice(idx, target - idx.size, replace=True)
        idx = np.concatenate([idx, pad])

    # Sanitize every drained float fed to the loss. If a poisoned sim transition ever
    # carried a NaN/inf (e.g. a value bootstrap that overflowed), an un-scrubbed input
    # turns the whole loss non-finite. Scrubbing here makes the loss provably finite on
    # any input, so a net can never be driven NaN by the data. The data-policy log-prob
    # floors illegal/zero-prob actions at a finite (not 1e30) value so the rev-KL term
    # stays bounded.
    def fin(a, neg=-1e9, pos=1e9):
        return np.nan_to_num(a, nan=0.0, posinf=pos, neginf=neg).astype(np.float32)

    # Count genuine corruption scrubbed from the drained inputs (data_log_prob's -inf
    # for illegal actions is the expected legal-mask artifact, so only its NaNs count).
    move_scrub = (_nonfinite(data["advantage"][idx]) + _nonfinite(data["ret"][idx])
                  + _nonfinite(data["old_log_prob"][idx])
                  + int(np.isnan(data["data_log_prob"][idx]).sum()))

    batch = {
        "obs": mx.array(fin(data["obs"][idx])),
        "legal": mx.array(data["legal_mask"][idx]),
        "action": mx.array(data["action"][idx].astype(np.int32)),
        "old_log_prob": mx.array(fin(data["old_log_prob"][idx], neg=-100.0, pos=0.0)),
        "data_log_prob": mx.array(fin(data["data_log_prob"][idx], neg=-100.0, pos=0.0)),
        "advantage": mx.array(fin(data["advantage"][idx])),
        "ret": mx.array(np.clip(fin(data["ret"][idx]), -1.0, 1.0)),
    }

    def loss_fn(net):
        loss, stats = move_loss_and_stats(net, batch, magnet_coef, cfg)
        return loss, stats

    (loss, stats), grads = nn.value_and_grad(move_net, loss_fn)(move_net)
    grads, gnorm = optim.clip_grad_norm(grads, cfg.max_grad_norm)
    loss_v = float(loss)
    gnorm_v = float(gnorm)
    # Signal a non-finite update so the caller self-heals (revert + LR backoff)
    # rather than corrupting the net or silently spinning.
    if not (np.isfinite(loss_v) and np.isfinite(gnorm_v)):
        return {"move/n_kept": int(keep.sum()), "move/n_threshold_keep": n_threshold,
                "move/n_total": int(n), "move/scrub": move_scrub,
                "move/nan_stage": _move_nan_stage(move_net, batch, stats),
                "move/skipped": True}, True
    opt.update(move_net, grads)
    mx.eval(move_net.parameters(), opt.state)
    return {
        "move/policy_loss": float(stats["policy_loss"]),
        "move/value_loss": float(stats["value_loss"]),
        "move/kl_loss": float(stats["kl_loss"]),
        "move/magnet_kl": float(stats["magnet_kl"]),
        "move/entropy": float(stats["entropy"]),
        "move/loss": float(loss),
        "move/grad_norm": float(gnorm),
        "move/n_kept": int(keep.sum()),
        "move/n_threshold_keep": n_threshold,
        "move/n_total": int(n),
        "move/scrub": move_scrub,
    }, False


def train_setup_pass(setup_net, opt, setup_data, reg_temp, lr, cfg):
    """5 epochs of the pure-MC setup loss over the drained arrangements.

    Returns `(stats, nan)`; `nan` is True iff every minibatch was non-finite (the
    pass made no progress), so the caller reverts the net instead of folding it in.
    """
    m = setup_data["seq"].shape[0]
    if m == 0:
        return {}, False
    opt.learning_rate = lr
    # Scrub the drained inputs (seq is a one-hot, outcome in {-1,0,1}); a stray
    # non-finite would otherwise flow through the baseline into every term. Count any
    # scrubbed corruption so it surfaces in metrics instead of being swallowed.
    setup_scrub = (_nonfinite(setup_data["seq"]) + _nonfinite(setup_data["outcome"])
                   + _nonfinite(setup_data["old_log_prob"]))
    seq_all = mx.array(np.nan_to_num(setup_data["seq"], nan=0.0).astype(np.float32))
    outcome_all = mx.array(np.clip(np.nan_to_num(setup_data["outcome"], nan=0.0), -1.0, 1.0).astype(np.float32))
    old_action_lp_all = mx.array(np.nan_to_num(
        setup_data["old_log_prob"], nan=0.0, posinf=0.0, neginf=-100.0
    ).astype(np.float32))
    # Freeze the behavior baseline once (the generation-time actor).
    base = setup_baseline(setup_net, seq_all, cfg.arr_reg_norm)
    mx.eval(base["log_probs"], base["value_scalar"], base["reg_returns"], base["reg_adv"])

    last = {}
    n_skipped = 0
    n_applied = 0
    nan_stage = ""
    bs = cfg.arr_batch_size
    for _ in range(cfg.arr_num_epoch_per_train):
        perm = np.random.permutation(m)
        for i in range(0, m, bs):
            bi = perm[i:i + bs]
            bidx = mx.array(bi)
            batch = {
                "seq": seq_all[bidx],
                "outcome": outcome_all[bidx],
                "reg_temp": reg_temp,
                "old_action_log_prob": old_action_lp_all[bidx],
                "old_log_probs": base["log_probs"][bidx],
                "old_value_scalar": base["value_scalar"][bidx],
                "reg_returns": base["reg_returns"][bidx],
                "reg_adv": base["reg_adv"][bidx],
            }

            def loss_fn(net):
                loss, stats = setup_loss_and_stats(net, batch, cfg)
                return loss, stats

            (loss, stats), grads = nn.value_and_grad(setup_net, loss_fn)(setup_net)
            # The setup net reuses `start_token` inside its legal-mask cumsum, so
            # it must stay the fixed-zero start it is documented to be — training
            # it would corrupt the per-slot legality. Freeze its gradient.
            if "start_token" in grads:
                grads["start_token"] = mx.zeros_like(grads["start_token"])
            grads, gnorm = optim.clip_grad_norm(grads, cfg.arr_max_grad_norm)
            loss_v = float(loss)
            gnorm_v = float(gnorm)
            # A rare degenerate minibatch (e.g. a tiny all-draw setup batch once the
            # move policy plays both sides well) can still produce a non-finite
            # gradient. Skip that single update rather than corrupt the net or
            # crash the run — the next minibatch resumes cleanly.
            if not (np.isfinite(loss_v) and np.isfinite(gnorm_v)):
                n_skipped += 1
                if not nan_stage:
                    nan_stage = _setup_nan_stage(setup_net, batch, stats)
                continue
            opt.update(setup_net, grads)
            mx.eval(setup_net.parameters(), opt.state)
            n_applied += 1
            last = {
                "setup/policy_loss": float(stats["policy_loss"]),
                "setup/value_loss": float(stats["value_loss"]),
                "setup/entropy_loss": float(stats["entropy_loss"]),
                "setup/kl_loss": float(stats["kl_loss"]),
                "setup/loss": float(loss),
                "setup/grad_norm": float(gnorm),
            }
    last["setup/n_games"] = int(m)
    last["setup/n_skipped"] = n_skipped
    last["setup/n_applied"] = n_applied
    last["setup/scrub"] = setup_scrub
    last["setup/nan_stage"] = nan_stage
    return last, n_applied == 0


def build(cfg):
    move = S.MoveTransformer.from_config(cfg.move_net_config())
    setup = S.ArrangementTransformer.from_config(cfg.setup_net_config())
    move_opt = optim.AdamW(learning_rate=cfg.lr(0), betas=[cfg.adam_b1, cfg.adam_b2],
                           eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
    setup_opt = optim.AdamW(learning_rate=cfg.arr_lr, betas=[cfg.adam_b1, cfg.adam_b2],
                            eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
    move_ema = S.EMA(move, decay=cfg.ema_decay)
    setup_ema = S.EMA(setup, decay=cfg.ema_decay)
    mx.eval(move.parameters(), setup.parameters())
    return move, setup, move_opt, setup_opt, move_ema, setup_ema


def train(cfg: TrainConfig):
    mx.random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    # Be a good citizen on the shared 64 GB box: cap MLX's working set and its
    # allocator cache so a transient spike can't page-fault the GPU or starve
    # other processes. `set_memory_limit` makes over-allocation raise (a catchable
    # Python error) instead of a fatal Metal command-buffer fault.
    if cfg.mlx_memory_limit_gb > 0:
        mx.set_memory_limit(int(cfg.mlx_memory_limit_gb * 1024**3))
    if cfg.mlx_cache_limit_gb > 0:
        mx.set_cache_limit(int(cfg.mlx_cache_limit_gb * 1024**3))
    run = RunDir(cfg.runs_root, cfg.run_name, cfg.work_seconds)
    move, setup, move_opt, setup_opt, move_ema, setup_ema = build(cfg)

    s = sim.BatchSim(num_envs=cfg.num_envs, move_cap=cfg.move_cap, seed=cfg.seed,
                     buffer_capacity=cfg.buffer_capacity)

    total_env_steps = 0
    # Self-heal state: last-good in-memory snapshots, per-net LR scales, and the
    # watchdog's consecutive-unhealthy-iter counter.
    move_snap = _snapshot(move, move_opt)
    setup_snap = _snapshot(setup, setup_opt)
    move_lr_scale = 1.0
    setup_lr_scale = 1.0
    bad_streak = 0
    print(f"[run] {run.path}  envs={cfg.num_envs} iters={cfg.iters} "
          f"move={cfg.move_net_config()} setup={cfg.setup_net_config()}")

    for t in range(cfg.iters):
        if run.should_stop():
            print(f"[stop] STOP/work-budget at iter {t}")
            break
        it_start = time.time()

        # 1-2: self-play + drain
        env_steps, n_term, rsum, collect_scrub = collect_iter(s, move, setup, cfg.collect_steps)
        total_env_steps += env_steps
        move_data = s.drain_training_batch(cfg.td_lambda, cfg.gae_lambda)
        setup_data = s.drain_setup_batch()

        # 3: train, with per-net self-heal (revert + LR backoff on a non-finite
        # update so a transient blow-up can't corrupt a net or freeze the run).
        magnet_coef = cfg.magnet_coef(t)
        reg_temp = cfg.setup_temperature(t)
        move_lr = cfg.lr(t) * move_lr_scale
        setup_lr = cfg.arr_lr * setup_lr_scale

        m_stats, m_nan = train_move_pass(move, move_opt, move_data, magnet_coef, move_lr, cfg)
        move_applied = "move/skipped" not in m_stats
        move_snap, move_lr_scale, move_nan, move_stage = _self_heal(
            move, move_opt, move_ema, move_snap, m_nan, move_applied, move_lr, move_lr_scale, cfg)

        a_stats, a_nan = train_setup_pass(setup, setup_opt, setup_data, reg_temp, setup_lr, cfg)
        setup_applied = a_stats.get("setup/n_applied", 0) > 0
        setup_snap, setup_lr_scale, setup_nan, setup_stage = _self_heal(
            setup, setup_opt, setup_ema, setup_snap, a_nan, setup_applied, setup_lr, setup_lr_scale, cfg)

        # Drop the large drained numpy batches and flush the MLX buffer cache each
        # iteration — without this the Metal allocator's cache grows unbounded
        # (iter time balloons, then OOM on a shared 64 GB box).
        del move_data, setup_data
        mx.clear_cache()

        rec = {
            "iter": t,
            "env_steps": total_env_steps,
            "work_seconds": round(run.elapsed(), 1),
            "iter_seconds": round(time.time() - it_start, 2),
            "lr": move_lr,
            "magnet_coef": magnet_coef,
            "setup_temp": reg_temp,
            "n_terminals": n_term,
            "reward_pl0_mean": (rsum / n_term) if n_term else 0.0,
            "move/nan": move_nan,
            "setup/nan": setup_nan,
            "move/nan_where": (m_stats.get("move/nan_stage", "") or move_stage),
            "setup/nan_where": (a_stats.get("setup/nan_stage", "") or setup_stage),
            "move_lr_scale": round(move_lr_scale, 4),
            "setup_lr_scale": round(setup_lr_scale, 4),
            **m_stats,
            **a_stats,
        }
        # Total non-finite net/sim values scrubbed this iter — 0 in a healthy run; a
        # NaN entering from the verified sim must surface here, never be swallowed.
        scrubbed = collect_scrub + m_stats.get("move/scrub", 0) + a_stats.get("setup/scrub", 0)
        rec["train/scrubbed"] = scrubbed

        # Watchdog: a healthy iter resets the streak; otherwise count it. HALT after
        # `watchdog_patience` straight bad iters. "Bad" = a net stayed non-finite, OR
        # the REAL spec advantage filter kept ~nothing (`move/n_threshold_keep`, the
        # pre-floor count — full1's silent starvation freeze had NO NaN), OR a reported
        # loss is non-finite, OR genuine corruption was scrubbed. The min_keep floor
        # keeps the batch size stable but must NOT hide the starvation alarm, so the
        # alarm watches the pre-floor count, not the floored `move/n_kept`.
        move_threshold = m_stats.get("move/n_threshold_keep", 1)
        reported_finite = (np.isfinite(rec.get("move/loss", 0.0))
                           and np.isfinite(rec.get("setup/loss", 0.0)))
        iter_bad = (move_nan or setup_nan or (move_threshold == 0)
                    or (scrubbed > 0) or not reported_finite)
        bad_streak = bad_streak + 1 if iter_bad else 0
        rec["bad_streak"] = bad_streak

        # 5: eval — win share vs uniform-random (the milestone proof) and vs a
        # frozen earlier policy (self-play improvement). We log BOTH the live
        # working policy (`winrate_vs_random`, the direct learning signal) and the
        # EMA/"magnet" policy (`ema_winrate_vs_random`, the spec-faithful deployed
        # net). Over a short smoke the 0.999 EMA lags the working net heavily, so
        # the working-net curve is the legible learning proof; the EMA catches up
        # over the full run. Hero sampling is sharpened (eval_temperature) so a
        # still-exploratory policy's learned preferences show through.
        do_eval = (t > 0 and t % cfg.eval_every == 0) or t == cfg.iters - 1
        if do_eval:
            # Run the eval in a SEPARATE process (its own MLX runtime). The
            # in-process eval (win_share's heavy self-play forwards) corrupts the
            # trainer's MLX/Metal runtime and NaNs the learner ~1 iter later —
            # proven by a controlled eval-off-vs-on pair with byte-identical
            # trajectories that diverge only at the eval; no in-process isolation
            # (input snapshots, learner snapshot+restore) stops it. Checkpoint,
            # eval out of process, read the winrate back. See eval_ckpt.py.
            import json as _json
            import os as _os
            import subprocess as _sp
            import sys as _sys
            ck = run.save_latest(move, move_opt, move_ema, setup, setup_opt,
                                 setup_ema, step=t)
            _pkg_parent = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            _env = {**_os.environ,
                    "PYTHONPATH": _pkg_parent + _os.pathsep + _os.environ.get("PYTHONPATH", "")}
            _proc = _sp.run(
                [_sys.executable, "-m", "stratego_trainer.eval_ckpt", "--ckpt", ck,
                 "--num-envs", str(min(128, cfg.num_envs)), "--games", str(cfg.eval_games),
                 "--move-cap", str(cfg.move_cap), "--seed", str(cfg.seed + 999),
                 "--temperature", str(cfg.eval_temperature)],
                capture_output=True, text=True, env=_env,
            )
            try:
                _res = _json.loads(_proc.stdout.strip().splitlines()[-1])
                rec["eval/winrate_vs_random"] = round(_res["ws_rand"], 4)
                rec["eval/ema_winrate_vs_random"] = round(_res["ema_ws_rand"], 4)
                run.maybe_save_best(_res["ws_rand"], move, move_opt, move_ema,
                                    setup, setup_opt, setup_ema, step=t)
            except Exception as _exc:
                rec["eval/error"] = f"{_exc} :: {_proc.stderr.strip()[-200:]}"

        run.log(rec)
        if t % 10 == 0 or "eval/winrate_vs_random" in rec or iter_bad:
            msg = (f"it {t:4d}  Lp={rec.get('move/policy_loss', 0):+.3f} "
                   f"Lv={rec.get('move/value_loss', 0):.3f} "
                   f"H={rec.get('move/entropy', 0):.3f} "
                   f"mag={rec.get('move/magnet_kl', 0):+.3f} lr={move_lr:.1e} "
                   f"setupL={rec.get('setup/loss', 0):.3f} "
                   f"nkept={m_stats.get('move/n_kept', 0)}(thr={move_threshold}) "
                   f"scrub={scrubbed} term={n_term} {rec.get('iter_seconds', 0):.1f}s")
            if iter_bad:
                msg += (f"  [BAD move_nan={move_nan} setup_nan={setup_nan} thr={move_threshold} "
                        f"scrub={scrubbed} streak={bad_streak} mscale={move_lr_scale:.3g} "
                        f"sscale={setup_lr_scale:.3g}]")
            if "eval/winrate_vs_random" in rec:
                msg += (f"  WSrand={rec['eval/winrate_vs_random']:.3f}"
                        f" (ema {rec.get('eval/ema_winrate_vs_random', 0):.3f})")
            if "eval/winrate_vs_frozen" in rec:
                msg += f" WSfrozen={rec['eval/winrate_vs_frozen']:.3f}"
            print(msg)

        if bad_streak >= cfg.watchdog_patience:
            run.save_periodic(move, move_opt, move_ema, setup, setup_opt, setup_ema, step=t)
            raise RuntimeError(
                f"[watchdog] {bad_streak} consecutive unhealthy iters through iter {t} "
                f"(move/n_threshold_keep={move_threshold} move_nan={move_nan} "
                f"setup_nan={setup_nan} scrubbed={scrubbed} reported_finite={reported_finite}). "
                f"Saved ckpt_{t}; halting so the run can't spin frozen — investigate before relaunching.")

        if t > 0 and t % cfg.save_every == 0:
            run.save_periodic(move, move_opt, move_ema, setup, setup_opt, setup_ema, step=t)

    run.save_latest(move, move_opt, move_ema, setup, setup_opt, setup_ema, step=cfg.iters)
    print(f"[done] {total_env_steps} env-steps, best WSrand={run.best_eval:.3f}, "
          f"metrics -> {run.metrics_path}")
    return run


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ataraxos Stratego self-play trainer (MLX)")
    p.add_argument("--envs", type=int, default=TrainConfig.num_envs)
    p.add_argument("--iters", type=int, default=TrainConfig.iters)
    p.add_argument("--collect-steps", type=int, default=TrainConfig.collect_steps)
    p.add_argument("--buffer-capacity", type=int, default=TrainConfig.buffer_capacity)
    p.add_argument("--move-cap", type=int, default=TrainConfig.move_cap)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--run-name", type=str, default="run")
    p.add_argument("--runs-root", type=str, default=TrainConfig.runs_root)
    p.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    p.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    p.add_argument("--eval-games", type=int, default=TrainConfig.eval_games)
    p.add_argument("--eval-temperature", type=float, default=TrainConfig.eval_temperature)
    p.add_argument("--work-seconds", type=float, default=TrainConfig.work_seconds)
    # The magnet-KL strength is the shipped-default 0.05 (spec §4.1) but "make it
    # configurable" — at smoke scale the magnet otherwise overwhelms the weaker
    # advantage signal and pins the policy at uniform, so the smoke lowers it.
    p.add_argument("--magnet-coef", type=float, default=TrainConfig.temperature_coef)
    p.add_argument("--mlx-memory-limit-gb", type=float, default=TrainConfig.mlx_memory_limit_gb)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    cfg = TrainConfig(
        num_envs=a.envs,
        iters=a.iters,
        collect_steps=a.collect_steps,
        buffer_capacity=a.buffer_capacity,
        move_cap=a.move_cap,
        seed=a.seed,
        run_name=a.run_name,
        runs_root=a.runs_root,
        save_every=a.save_every,
        eval_every=a.eval_every,
        eval_games=a.eval_games,
        eval_temperature=a.eval_temperature,
        temperature_coef=a.magnet_coef,
        work_seconds=a.work_seconds,
        mlx_memory_limit_gb=a.mlx_memory_limit_gb,
    )
    train(cfg)


if __name__ == "__main__":
    main()
