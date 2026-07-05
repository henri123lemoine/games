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
from .rundir import RunDir, load_checkpoint
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
        ("loss/anchor", stats["anchor_kl"]), ("loss/entropy", stats["entropy"]),
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


PIECE_COUNTS_MX = mx.array(list(S.spec.CLASSIC_PIECE_COUNTS), dtype=mx.float32)
PIECE_COUNTS_MX_BF16 = PIECE_COUNTS_MX.astype(mx.bfloat16)


def _move_forward_raw(net, obs_np, legal_np):
    """Build the move-net forward graph (lazy, un-evaluated). Returns the float32
    (move_logits, value_logp) tuple, or None for an empty batch. The collect loop
    evaluates move + setup graphs together in ONE mx.eval to cut per-step syncs."""
    if obs_np.shape[0] == 0:
        return None
    out = net(mx.array(obs_np).astype(mx.bfloat16), legal_mask=mx.array(legal_np))
    return (out["move_logits"].astype(mx.float32), out["value_logp"].astype(mx.float32))


def _sanitize_probs(raw_probs):
    """Repair a categorical distribution against non-finite corruption: fill
    NaN/inf with a uniform fallback, then renormalize so it stays a valid
    distribution for the sim's stored transitions (the buffer's categorical
    λ-return bootstraps directly from this per §4.1)."""
    probs = np.nan_to_num(raw_probs, nan=1.0 / 3.0, posinf=1.0, neginf=0.0)
    probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-6, None)
    return probs.astype(np.float32)


def _move_forward_finish(raw, obs_np):
    if raw is None:
        return (np.zeros((0, sim.N_ACTION), np.float32), np.zeros(0, np.float32),
                np.zeros((0, 3), np.float32), 0)
    raw_logits = np.array(raw[0])  # already float32 + evaluated
    vlogp = np.array(raw[1])
    raw_probs = np.exp(vlogp)
    raw_vals = (raw_probs * CATS).sum(-1)
    scrub = _nonfinite(raw_logits) + _nonfinite(raw_vals) + _nonfinite(raw_probs)
    logits = _sanitize_logits(raw_logits)
    vals = np.clip(np.nan_to_num(raw_vals, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0).astype(np.float32)
    probs = _sanitize_probs(raw_probs)
    return logits, vals, probs, scrub


def _setup_forward_raw(net, obs_np):
    """Build the setup-net forward graph (lazy). Returns (logits(B,40,14), value),
    or None for an empty batch."""
    if obs_np.shape[0] == 0:
        return None
    out = net(mx.array(obs_np).astype(mx.bfloat16), PIECE_COUNTS_MX_BF16)
    return (out["logits"].astype(mx.float32), out["value"].astype(mx.float32))


def _setup_forward_finish(raw, obs_np):
    if raw is None:
        return (np.zeros((0, sim.DEPLOY_WIDTH), np.float32), np.zeros(0, np.float32),
                np.zeros((0, 3), np.float32), 0)
    n_placed = obs_np.reshape(obs_np.shape[0], 40, 14).sum(axis=(1, 2)).astype(int)
    slot = np.clip(n_placed, 0, 39)
    raw_logits = np.array(raw[0])  # (B,40,14), float32 + evaluated
    scrub = _nonfinite(raw_logits)
    all_logits = _sanitize_logits(raw_logits)
    logits = all_logits[np.arange(obs_np.shape[0]), slot]
    vlogp = _stable_log_softmax(np.array(raw[1]), axis=-1)
    slot_probs = np.exp(vlogp[np.arange(obs_np.shape[0]), slot])
    scrub += _nonfinite(slot_probs)
    vals = np.clip((slot_probs * CATS).sum(-1), -1.0, 1.0)
    probs = _sanitize_probs(slot_probs)
    return logits, vals.astype(np.float32), probs, scrub


def collect_iter(s, move_net, setup_net, steps):
    """Run `steps` self-play decision steps.

    Returns (env_steps, n_terminals, reward_pl0_sum, n_decisive, n_capped,
    n_scrub, move_decisions, move_attacks, t_coll, t_fwd, t_comm) where
    n_decisive counts terminals with a nonzero reward (a real win/loss — the
    complement, among terminals, is a draw or a ply-cap timeout), n_capped
    counts terminals that were force-reset by the ply cap rather than a genuine
    rules-terminal, n_scrub counts the non-finite net outputs scrubbed before
    they could poison the sim's stored transitions, and move_decisions/
    move_attacks are the attack-rate telemetry (see `stratego::sim::RunStats`).
    """
    n_term = 0
    n_decisive = 0
    n_capped = 0
    move_decisions = 0
    move_attacks = 0
    rsum = 0.0
    scrub = 0
    t_coll = t_fwd = t_comm = 0.0
    for _ in range(steps):
        a = time.time(); b = s.collect(); t_coll += time.time() - a
        a = time.time()
        m_raw = _move_forward_raw(move_net, b["move_obs"], b["move_legal"])
        s_raw = _setup_forward_raw(setup_net, b["deploy_obs"])
        ev = []
        if m_raw is not None:
            ev += list(m_raw)
        if s_raw is not None:
            ev += list(s_raw)
        if ev:
            mx.eval(*ev)
        m_logits, m_vals, m_probs, m_scrub = _move_forward_finish(m_raw, b["move_obs"])
        d_logits, d_vals, d_probs, d_scrub = _setup_forward_finish(s_raw, b["deploy_obs"])
        t_fwd += time.time() - a
        scrub += m_scrub + d_scrub
        a = time.time()
        out = s.commit(m_logits, m_vals, m_probs, d_logits, d_vals, d_probs)
        t_comm += time.time() - a
        n_term += int(out["terminal"].sum())
        rsum += float(out["reward_pl0"].sum())
        n_decisive += int((out["reward_pl0"] != 0.0).sum())
        n_capped += int(out["capped"].sum())
        move_decisions += int(out["move_decisions"])
        move_attacks += int(out["move_attacks"])
    return (steps * s.num_envs, n_term, rsum, n_decisive, n_capped, scrub,
            move_decisions, move_attacks, t_coll, t_fwd, t_comm)


def train_move_pass(move_net, opt, data, encode_fn, magnet_coef, lr, cfg, bf16_net=None,
                    anchor_net=None, anchor_coef=0.0):
    """One advantage-filtered PPO pass over the drained move-RL transitions.

    `anchor_net`, if given, is the FROZEN BC warm-start move net (never updated);
    `anchor_coef` (`TrainConfig.anchor_coef_at(t)`) weights the reverse-KL trust
    region toward it (see `move_loss.move_loss_and_stats`). `anchor_net is None`
    (no `resume_from`) disables the term entirely regardless of `anchor_coef`.

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
    idx_all = np.nonzero(keep)[0]

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
    move_scrub = (_nonfinite(data["advantage"][idx_all]) + _nonfinite(data["ret"][idx_all])
                  + _nonfinite(data["old_log_prob"][idx_all])
                  + int(np.isnan(data["data_log_prob"][idx_all]).sum()))

    # Reference parity (rl.py:516-585): take MANY gradient steps over the WHOLE
    # advantage-filtered set each iteration — NOT one step over a capped sample. We
    # minibatch the kept rows (encoding obs per minibatch to bound Metal memory) and run
    # `move_num_epoch` epochs, mirroring the setup pass below. PPO clipping only does real
    # work because earlier minibatches move the net before later ones are seen; a single
    # step leaves the ratio ~1 and the clip inert (= one tiny vanilla-PG step / iter).
    bs = cfg.move_batch_size
    last: dict = {}
    n_applied = 0
    n_skipped = 0
    nan_stage = ""
    _encode_s = 0.0
    for _ in range(cfg.move_num_epoch):
        perm = np.random.permutation(idx_all)
        for i in range(0, perm.size, bs):
            mb = perm[i:i + bs]
            # Pad the last partial minibatch to a bucket multiple (stable MPS shape;
            # repeated rows only reweight the mean, which the advantage filter already does).
            target = _bucket(mb.size, cfg.bucket)
            if target > mb.size:
                mb = np.concatenate([mb, np.random.choice(mb, target - mb.size, replace=True)])
            _t_enc = time.time()
            obs_mb = encode_fn(data["env"][mb], data["slot"][mb])
            _encode_s += time.time() - _t_enc
            _obs = mx.array(fin(obs_mb))
            if bf16_net is not None:
                _obs = _obs.astype(mx.bfloat16)
            batch = {
                "obs": _obs,
                "legal": mx.array(data["legal_mask"][mb]),
                "action": mx.array(data["action"][mb].astype(np.int32)),
                "old_log_prob": mx.array(fin(data["old_log_prob"][mb], neg=-100.0, pos=0.0)),
                "data_log_prob": mx.array(fin(data["data_log_prob"][mb], neg=-100.0, pos=0.0)),
                "advantage": mx.array(fin(data["advantage"][mb])),
                "ret": mx.array(fin(data["ret"][mb])),  # (bs, 3) categorical value target
            }

            def loss_fn(net):
                anchor_lp = None
                if anchor_net is not None and anchor_coef > 0.0:
                    a_out = anchor_net(batch["obs"], legal_mask=batch["legal"])
                    a_logits = a_out["move_logits"].astype(mx.float32)
                    anchor_lp = mx.stop_gradient(
                        a_logits - mx.logsumexp(a_logits, axis=-1, keepdims=True))
                loss, stats = move_loss_and_stats(net, batch, magnet_coef, cfg,
                                                  anchor_log_probs=anchor_lp,
                                                  anchor_coef=anchor_coef)
                return loss, stats

            if bf16_net is not None:
                bf16_net.update(tree_map(lambda p: p.astype(mx.bfloat16), move_net.parameters()))
                _fwd_net = bf16_net
                (loss, stats), grads = nn.value_and_grad(bf16_net, loss_fn)(bf16_net)
                grads = tree_map(lambda g: g.astype(mx.float32), grads)
            else:
                _fwd_net = move_net
                (loss, stats), grads = nn.value_and_grad(move_net, loss_fn)(move_net)
            grads, gnorm = optim.clip_grad_norm(grads, cfg.max_grad_norm)
            loss_v = float(loss)
            gnorm_v = float(gnorm)
            # Skip a single non-finite minibatch rather than corrupt the net or crash;
            # the next minibatch resumes cleanly (only an all-skipped pass reverts).
            if not (np.isfinite(loss_v) and np.isfinite(gnorm_v)):
                n_skipped += 1
                if not nan_stage:
                    nan_stage = _move_nan_stage(_fwd_net, batch, stats)
                continue
            opt.update(move_net, grads)
            mx.eval(move_net.parameters(), opt.state)
            n_applied += 1
            last = {
                "move/policy_loss": float(stats["policy_loss"]),
                "move/value_loss": float(stats["value_loss"]),
                "move/kl_loss": float(stats["kl_loss"]),
                "move/magnet_kl": float(stats["magnet_kl"]),
                "move/anchor_kl": float(stats["anchor_kl"]),
                "move/entropy": float(stats["entropy"]),
                "move/loss": float(loss),
                "move/grad_norm": float(gnorm),
            }
    last.update({
        "move/n_kept": int(keep.sum()),
        "move/n_threshold_keep": n_threshold,
        "move/n_total": int(n),
        "move/scrub": move_scrub,
        "move/n_applied": n_applied,
        "move/n_skipped": n_skipped,
        "move/nan_stage": nan_stage,
        "move/t_encode": round(_encode_s, 3),
        "move/anchor_coef": anchor_coef,
    })
    return last, n_applied == 0


def train_setup_pass(setup_net, opt, setup_data, reg_temp, lr, cfg, bf16_net=None):
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
                "seq": (seq_all[bidx].astype(mx.bfloat16) if bf16_net is not None else seq_all[bidx]),
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

            if bf16_net is not None:
                bf16_net.update(tree_map(lambda p: p.astype(mx.bfloat16), setup_net.parameters()))
                _fwd_net = bf16_net
                (loss, stats), grads = nn.value_and_grad(bf16_net, loss_fn)(bf16_net)
                grads = tree_map(lambda g: g.astype(mx.float32), grads)
            else:
                _fwd_net = setup_net
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
                    nan_stage = _setup_nan_stage(_fwd_net, batch, stats)
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
    # Investigated 2026-07-05 (valrun3's persistently-elevated setup/grad_norm,
    # ~20-500 from iter~100 on, with healthy small losses and no NaN throughout):
    # `setup/grad_norm` is the PRE-clip norm (`optim.clip_grad_norm` returns "the
    # original gradient norm" per its own docstring) — the norm 20-500 is real but
    # the APPLIED step is always rescaled to <= arr_max_grad_norm=0.5, so this
    # never risked corrupting the net. `n_samples` (not just `n_games`) makes the
    # mean-reduction's real denominator explicit: policy/value/entropy losses mean
    # over (n_games, 40 slots), so grad-norm scale should be read against this, not
    # `n_games` alone. Best explanation, not fully proven: once the BC-warm-started
    # move net produces decisive (not clock-drawn) games, `grounded_adv = outcome -
    # value_baseline` swings toward +/-1 instead of staying near 0 (a from-scratch
    # run's mostly-draw regime) -- a legitimately larger, valid gradient for the
    # clip to bound, not a sign of a broken update.
    last["setup/n_samples"] = int(m) * S.spec.ARRANGEMENT_SIZE
    last["setup/n_skipped"] = n_skipped
    last["setup/n_applied"] = n_applied
    last["setup/scrub"] = setup_scrub
    last["setup/nan_stage"] = nan_stage
    return last, n_applied == 0


def build(cfg):
    # Bias params must not be weight-decayed (the reference splits param
    # groups; we don't yet) — see the config.py note on weight_decay.
    assert cfg.weight_decay == 0.0, "implement bias/non-bias param groups before enabling weight_decay"
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
    run = RunDir(cfg.runs_root, cfg.run_name, cfg.work_seconds, net_size=cfg.net_size)
    move, setup, move_opt, setup_opt, move_ema, setup_ema = build(cfg)
    if cfg.resume_from:
        # Params + EMA only (no opt/=None): a warm start (e.g. BC) uses a
        # different loss/optimizer regime, so RL always begins with the fresh
        # AdamW state `build(cfg)` just constructed.
        load_checkpoint(cfg.resume_from, move=move, setup=setup,
                        move_ema=move_ema, setup_ema=setup_ema)
        print(f"[resume] loaded {cfg.resume_from}")
    # The frozen BC-anchor: a standalone move net loaded from the SAME checkpoint,
    # then never updated again (not part of `move_opt`'s tree, no `.update()` call
    # after this). `None` when there's no warm start — nothing to anchor to.
    anchor_net = None
    if cfg.resume_from:
        anchor_net = S.MoveTransformer.from_config(cfg.move_net_config())
        load_checkpoint(cfg.resume_from, move=anchor_net)
        mx.eval(anchor_net.parameters())
    # bf16 inference copies for the collect (self-play) forward — the collect forward
    # is the per-iter bottleneck and runs ~1.75x faster in bf16. Self-play data is
    # precision-insensitive (argmax/value barely move); TRAINING stays fp32 (the
    # validated loss/optimizer math and stability are untouched).
    move_bf16 = S.MoveTransformer.from_config(cfg.move_net_config())
    setup_bf16 = S.ArrangementTransformer.from_config(cfg.setup_net_config())

    s = sim.BatchSim(num_envs=cfg.num_envs, move_cap=cfg.move_cap, seed=cfg.seed,
                     buffer_capacity=cfg.buffer_capacity, attack_clock=cfg.attack_clock(0))

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

        attack_clock = cfg.attack_clock(t)
        if attack_clock != s.attack_clock:
            s.set_attack_clock(attack_clock)

        # 1-2: self-play + drain
        move_bf16.update(tree_map(lambda p: p.astype(mx.bfloat16), move.parameters()))
        setup_bf16.update(tree_map(lambda p: p.astype(mx.bfloat16), setup.parameters()))
        mx.eval(move_bf16.parameters(), setup_bf16.parameters())
        (env_steps, n_term, rsum, n_decisive, n_capped, collect_scrub,
         move_decisions, move_attacks, _tc, _tf, _tm) = collect_iter(
            s, move_bf16, setup_bf16, cfg.collect_steps)
        total_env_steps += env_steps
        _t_collect = time.time()
        move_data = s.drain_training_batch(cfg.td_lambda, cfg.gae_lambda)
        setup_data = s.drain_setup_batch()
        _t_drain = time.time()

        # 3: train, with per-net self-heal (revert + LR backoff on a non-finite
        # update so a transient blow-up can't corrupt a net or freeze the run).
        magnet_coef = cfg.magnet_coef(t)
        reg_temp = cfg.setup_temperature(t)
        move_lr = cfg.lr(t) * move_lr_scale
        setup_lr = cfg.arr_lr * setup_lr_scale

        # fp32 warmup: from random init the first iters are bf16-fragile (the net can
        # overflow before the self-heal recovers -> early NaN cascade). Train them in
        # fp32, then switch to bf16. Collect stays bf16 throughout (sanitized inference).
        _bf16 = cfg.bf16_train and t >= 15
        anchor_coef = cfg.anchor_coef_at(t)
        m_stats, m_nan = train_move_pass(move, move_opt, move_data, s.encode_move_obs, magnet_coef, move_lr, cfg,
                                         bf16_net=move_bf16 if _bf16 else None,
                                         anchor_net=anchor_net, anchor_coef=anchor_coef)
        move_applied = "move/skipped" not in m_stats
        move_snap, move_lr_scale, move_nan, move_stage = _self_heal(
            move, move_opt, move_ema, move_snap, m_nan, move_applied, move_lr, move_lr_scale, cfg)

        a_stats, a_nan = train_setup_pass(setup, setup_opt, setup_data, reg_temp, setup_lr, cfg,
                                          bf16_net=setup_bf16 if _bf16 else None)
        setup_applied = a_stats.get("setup/n_applied", 0) > 0
        setup_snap, setup_lr_scale, setup_nan, setup_stage = _self_heal(
            setup, setup_opt, setup_ema, setup_snap, a_nan, setup_applied, setup_lr, setup_lr_scale, cfg)

        _t_train = time.time()
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
            "t/collect": round(_t_collect - it_start, 3),
            "t/c_rust_collect": round(_tc, 3),
            "t/c_forward": round(_tf, 3),
            "t/c_rust_commit": round(_tm, 3),
            "t/drain": round(_t_drain - _t_collect, 3),
            "t/train": round(_t_train - _t_drain, 3),
            "lr": move_lr,
            "magnet_coef": magnet_coef,
            "setup_temp": reg_temp,
            "n_terminals": n_term,
            "reward_pl0_mean": (rsum / n_term) if n_term else 0.0,
            # `draw_frac`: share of terminals with zero reward (a genuine rules
            # draw OR a ply-cap timeout — the direct symptom of draw collapse).
            # `capped_frac`: of those, the share that hit the ply cap specifically
            # (as opposed to a real rules-terminal draw, which is rare). A healthy
            # run keeps both low; both climbing toward 1.0 is the iter-540-style
            # passivity stall this pair of fields exists to catch early.
            "draw_frac": ((n_term - n_decisive) / n_term) if n_term else 0.0,
            "capped_frac": (n_capped / n_term) if n_term else 0.0,
            "attack_clock": attack_clock,
            # The direct signature of the passivity trap (2026-07-04): a
            # from-scratch net's attack rate collapsing toward 0 starves the
            # value head long before draw_frac itself climbs. HeuristicBot's
            # own rate is ~24.5/100 plies; watch this trend, not just draw_frac.
            "move_decisions": move_decisions,
            "attacks_per_100_plies": (100.0 * move_attacks / move_decisions) if move_decisions else 0.0,
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
        # `watchdog_patience` straight bad iters. "Bad" = a net stayed non-finite, a
        # reported loss is non-finite, or genuine corruption was scrubbed. NOTE:
        # `move/n_threshold_keep == 0` is NOT bad — with the spec filter (thresh=0.01,
        # no min_keep) an iter where no |adv| clears the threshold legitimately trains
        # on nothing (the reference idles too); progress is judged by the eval, not by
        # forcing an update every iter.
        move_threshold = m_stats.get("move/n_threshold_keep", 1)
        reported_finite = (np.isfinite(rec.get("move/loss", 0.0))
                           and np.isfinite(rec.get("setup/loss", 0.0)))
        iter_bad = (move_nan or setup_nan or (scrubbed > 0) or not reported_finite)
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
                 "--move-cap", str(cfg.eval_move_cap), "--seed", str(cfg.seed + 999),
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
                   f"scrub={scrubbed} term={n_term} draw={rec['draw_frac']:.2f}"
                   f"(cap={rec['capped_frac']:.2f}) atk={rec['attacks_per_100_plies']:.1f}/100"
                   f"(clock={attack_clock}) {rec.get('iter_seconds', 0):.1f}s")
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
    p.add_argument("--eval-move-cap", type=int, default=TrainConfig.eval_move_cap)
    p.add_argument("--eval-temperature", type=float, default=TrainConfig.eval_temperature)
    p.add_argument("--work-seconds", type=float, default=TrainConfig.work_seconds)
    # The magnet-KL strength is the shipped-default 0.05 (spec §4.1) but "make it
    # configurable" — at smoke scale the magnet otherwise overwhelms the weaker
    # advantage signal and pins the policy at uniform, so the smoke lowers it.
    p.add_argument("--magnet-coef", type=float, default=TrainConfig.temperature_coef)
    p.add_argument("--magnet-decay", type=float, default=TrainConfig.temperature_decay)
    p.add_argument("--mlx-memory-limit-gb", type=float, default=TrainConfig.mlx_memory_limit_gb)
    p.add_argument("--bf16-train", action="store_true", default=TrainConfig.bf16_train)
    p.add_argument("--net-size", choices=tuple(S.NET_SIZES), default=TrainConfig.net_size)
    p.add_argument("--resume", type=str, default=TrainConfig.resume_from,
                   help="warm-start checkpoint (e.g. a BC run's output); params+EMA only")
    p.add_argument("--clock-start", type=int, default=TrainConfig.clock_start)
    p.add_argument("--clock-end", type=int, default=TrainConfig.clock_end)
    p.add_argument("--clock-anneal-iters", type=int, default=TrainConfig.clock_anneal_iters)
    p.add_argument("--anchor-coef", type=float, default=TrainConfig.anchor_coef,
                   help="BC-anchor trust-region start coefficient (0 disables even with --resume)")
    p.add_argument("--anchor-decay", type=float, default=TrainConfig.anchor_decay)
    p.add_argument("--anchor-floor", type=float, default=TrainConfig.anchor_floor)
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
        eval_move_cap=a.eval_move_cap,
        eval_temperature=a.eval_temperature,
        temperature_coef=a.magnet_coef,
        temperature_decay=a.magnet_decay,
        work_seconds=a.work_seconds,
        mlx_memory_limit_gb=a.mlx_memory_limit_gb,
        bf16_train=a.bf16_train,
        net_size=a.net_size,
        resume_from=a.resume,
        clock_start=a.clock_start,
        clock_end=a.clock_end,
        clock_anneal_iters=a.clock_anneal_iters,
        anchor_coef=a.anchor_coef,
        anchor_decay=a.anchor_decay,
        anchor_floor=a.anchor_floor,
    )
    train(cfg)


if __name__ == "__main__":
    main()
