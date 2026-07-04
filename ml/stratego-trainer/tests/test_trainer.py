"""Tests for the self-play training loop: the loss math, the schedules, the
setup-batch bridge schema, and a short end-to-end iteration that must learn.

These need the `stratego_sim` bridge installed (`maturin develop` from
ml/stratego-py); they skip cleanly if it is absent.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

import stratego_nets as S
from stratego_trainer.config import TrainConfig, power_schedule
from stratego_trainer.move_loss import advantage_filter_mask, move_loss_and_stats, two_hot
from stratego_trainer.setup_loss import setup_baseline, setup_loss_and_stats

sim = pytest.importorskip("stratego_sim")


def test_power_schedule_matches_reference():
    # LR schedule: clip(0.5/(t+1)^1.1, 5e-6, 1e-4); starts pinned at the ceiling.
    assert power_schedule(0.5, 0, 1.1, 1e-4, 5e-6) == pytest.approx(1e-4)
    # far out it anneals toward the floor and clips there.
    assert power_schedule(0.5, 10**9, 1.1, 1e-4, 5e-6) == pytest.approx(5e-6)
    # magnet coef monotonically non-increasing.
    cfg = TrainConfig()
    coefs = [cfg.magnet_coef(t) for t in range(0, 200, 10)]
    assert all(b <= a + 1e-12 for a, b in zip(coefs, coefs[1:]))


def test_two_hot_reconstructs_scalar():
    cats = mx.array([-1.0, 0.0, 1.0])
    v = mx.array([-1.0, -0.3, 0.0, 0.7, 1.0])
    th = two_hot(v, cats)
    recon = (th * cats).sum(-1)
    assert np.allclose(np.array(recon), np.array(v), atol=1e-5)
    # rows are valid distributions
    assert np.allclose(np.array(th.sum(-1)), 1.0, atol=1e-5)


def test_advantage_filter_keeps_top_quantile():
    adv = np.array([0.0, 0.005, 0.5, -0.9, 0.02, -0.001])
    keep, n_threshold = advantage_filter_mask(adv, rate=0.75, thresh=0.01)
    # threshold = max(q75(|adv|), 0.01); the two large-|adv| rows survive.
    assert keep[2] and keep[3]
    assert not keep[0] and not keep[1] and not keep[5]
    assert n_threshold == int(keep.sum())  # no floor applied -> equal


def test_advantage_filter_min_keep_floor_prevents_starve():
    # As the policy sharpens |adv| shrinks below the 0.01 floor for every row, so
    # the threshold keep collapses to 0 (the full1 freeze precursor). The min_keep
    # floor retains the largest-|adv| rows so the pass keeps a stable batch -- but the
    # returned n_threshold still reports the REAL (0) spec keep so the watchdog sees it.
    adv = np.linspace(-0.001, 0.001, 100)
    mask0, n_thr0 = advantage_filter_mask(adv, 0.75, 0.01, min_keep=0)
    assert mask0.sum() == 0 and n_thr0 == 0
    keep, n_thr = advantage_filter_mask(adv, 0.75, 0.01, min_keep=16)
    assert keep.sum() == 16  # floored batch keeps training going
    assert n_thr == 0  # but the real spec keep is reported as 0 (the starvation alarm)
    # the retained rows are exactly the largest |adv|.
    assert np.abs(adv[keep]).min() >= np.abs(adv[~keep]).max()
    # min_keep never shrinks a keep that already clears the threshold.
    big = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    maskb, n_thrb = advantage_filter_mask(big, 0.75, 0.01, min_keep=2)
    assert maskb.sum() == 3 and n_thrb == 3


def _drive(s, steps):
    for _ in range(steps):
        b = s.collect()
        nm = b["move_obs"].shape[0]
        nd = b["deploy_obs"].shape[0]
        s.commit(
            np.zeros((nm, sim.N_ACTION), np.float32), np.zeros(nm, np.float32),
            np.full((nm, 3), 1.0 / 3.0, np.float32),
            np.zeros((nd, sim.DEPLOY_WIDTH), np.float32), np.zeros(nd, np.float32),
            np.full((nd, 3), 1.0 / 3.0, np.float32),
        )


def test_drain_setup_batch_schema():
    s = sim.BatchSim(num_envs=64, move_cap=200, seed=11)
    _drive(s, 700)
    d = s.drain_setup_batch()
    m = d["seq"].shape[0]
    assert m > 0
    assert d["seq"].shape == (m, 40, 14)
    assert d["action"].shape == (m, 40)
    assert d["old_log_prob"].shape == (m, 40)
    assert d["outcome"].shape == (m,)
    assert d["player"].shape == (m,)
    seq = d["seq"]
    # every slot is a one-hot; each game is a full 40-placement classic setup.
    assert np.all(seq.reshape(m, 40, 14).sum(-1) == 1)
    counts = seq.reshape(m, 40, 14).sum(1)
    expected = np.array(S.spec.CLASSIC_PIECE_COUNTS, np.float32)
    assert np.allclose(counts.mean(0), expected, atol=0.01)
    assert set(np.unique(d["outcome"]).tolist()) <= {-1.0, 0.0, 1.0}
    assert set(np.unique(d["player"]).tolist()) <= {0, 1}
    # a second drain returns nothing (the queue was consumed).
    assert s.drain_setup_batch()["seq"].shape[0] == 0


def test_move_loss_reduces_on_real_data():
    s = sim.BatchSim(num_envs=64, move_cap=200, seed=5)
    _drive(s, 400)
    d = s.drain_training_batch(0.8, 0.5)
    keep, _ = advantage_filter_mask(d["advantage"], 0.75, 0.01)
    idx = np.nonzero(keep)[0]
    if idx.size < 8:
        pytest.skip("too few filtered transitions in this short drive")
    idx = idx[:256]
    obs = s.encode_move_obs(d["env"][idx], d["slot"][idx])
    batch = {
        "obs": mx.array(obs),
        "legal": mx.array(d["legal_mask"][idx]),
        "action": mx.array(d["action"][idx].astype(np.int32)),
        "old_log_prob": mx.array(d["old_log_prob"][idx]),
        "data_log_prob": mx.array(np.where(np.isfinite(d["data_log_prob"][idx]),
                                           d["data_log_prob"][idx], -1e30).astype(np.float32)),
        "advantage": mx.array(d["advantage"][idx]),
        "ret": mx.array(d["ret"][idx]),
    }
    net = S.MoveTransformer.from_config(S.MoveConfig())
    opt = optim.AdamW(learning_rate=1e-3)
    cfg = TrainConfig()
    losses = []
    for _ in range(30):
        def lf(n):
            return move_loss_and_stats(n, batch, cfg.magnet_coef(0), cfg)
        (loss, _), grads = nn.value_and_grad(net, lf)(net)
        grads, _ = optim.clip_grad_norm(grads, cfg.max_grad_norm)
        opt.update(net, grads)
        mx.eval(net.parameters(), opt.state)
        losses.append(float(loss))
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]


def test_setup_loss_reduces_and_stays_finite():
    s = sim.BatchSim(num_envs=64, move_cap=200, seed=9)
    _drive(s, 700)
    d = s.drain_setup_batch()
    if d["seq"].shape[0] < 8:
        pytest.skip("too few setup games")
    net = S.ArrangementTransformer.from_config(S.SetupConfig())
    opt = optim.AdamW(learning_rate=5e-5)
    cfg = TrainConfig()
    seq = mx.array(d["seq"])
    outcome = mx.array(d["outcome"])
    base = setup_baseline(net, seq, cfg.arr_reg_norm)
    mx.eval(base)
    losses = []
    for _ in range(10):
        batch = {
            "seq": seq, "outcome": outcome, "reg_temp": cfg.setup_temperature(0),
            "old_log_probs": base["log_probs"], "old_value_scalar": base["value_scalar"],
            "reg_returns": base["reg_returns"], "reg_adv": base["reg_adv"],
        }

        def lf(n):
            return setup_loss_and_stats(n, batch, cfg)
        (loss, _), grads = nn.value_and_grad(net, lf)(net)
        # the setup net's start token must stay fixed-zero (it feeds the legal mask).
        grads["start_token"] = mx.zeros_like(grads["start_token"])
        grads, _ = optim.clip_grad_norm(grads, cfg.arr_max_grad_norm)
        opt.update(net, grads)
        mx.eval(net.parameters(), opt.state)
        losses.append(float(loss))
    assert np.isfinite(losses).all(), losses
    assert losses[-1] < losses[0]


def test_setup_targets_bounded_under_sharpened_policy():
    """Reproduces the iter-51 divergence and proves the cap fixes it.

    A sharpened placement head forces low-probability placements whose per-slot NLL
    suffix-sums to a huge conditional-entropy target/advantage. On the unbounded
    reference reg_returns ran to ~120 and reg_adv to ~1200 (entropy_loss ~6000); the
    max-entropy caps bound them to the trajectory envelope, finite and O(1).
    """
    from stratego_trainer import setup_loss as SL

    counts = list(S.spec.CLASSIC_PIECE_COUNTS)
    flag = S.spec.FLAG_IDX

    def valid_seq():  # a legal classic arrangement (flag on the right half for handedness)
        types = [i for i, c in enumerate(counts) if i != flag for _ in range(c)] + [flag]
        s = np.zeros((40, 14), np.float32)
        s[np.arange(40), np.array(types)] = 1.0
        return s

    net = S.ArrangementTransformer.from_config(S.SetupConfig())
    mx.eval(net.parameters())
    # Sharpen only the placement head -> the policy disfavors most placed types.
    p = net.parameters()
    p["policy_out"]["weight"] = p["policy_out"]["weight"] * 60.0
    p["policy_out"]["bias"] = p["policy_out"]["bias"] * 60.0
    net.update(p)
    mx.eval(net.parameters())

    seq = mx.array(np.stack([valid_seq() for _ in range(32)]))
    outcome = mx.array(np.random.RandomState(0).choice([-1.0, 0.0, 1.0], 32).astype(np.float32))
    cfg = TrainConfig()
    base = setup_baseline(net, seq, cfg.arr_reg_norm)
    mx.eval(base)
    rr = np.array(base["reg_returns"])
    ra = np.array(base["reg_adv"])
    assert np.isfinite(rr).all() and np.isfinite(ra).all()
    # Bounded to the max-entropy envelope (this assertion fails on the unbounded ref).
    assert rr.max() <= SL.MAX_FUTURE_NLL / cfg.arr_reg_norm + 1e-3
    assert np.abs(ra).max() <= SL.MAX_FUTURE_NLL + 1e-3

    batch = {
        "seq": seq, "outcome": outcome, "reg_temp": cfg.setup_temperature(0),
        "old_log_probs": base["log_probs"], "old_value_scalar": base["value_scalar"],
        "reg_returns": base["reg_returns"], "reg_adv": base["reg_adv"],
    }
    (loss, stats), grads = nn.value_and_grad(net, lambda n: setup_loss_and_stats(n, batch, cfg))(net)
    grads["start_token"] = mx.zeros_like(grads["start_token"])
    _, gnorm = optim.clip_grad_norm(grads, cfg.arr_max_grad_norm)
    assert np.isfinite(float(loss)) and np.isfinite(float(gnorm))
    # The entropy MSE target is bounded, so the loss can't reach the thousands it hit pre-fix.
    assert float(stats["entropy_loss"]) < 500.0


def test_setup_loss_robust_to_blown_up_entropy_head():
    """The actual 1024-scale failure: the unregularized ent head extrapolated to ~-80
    against a target <= 3.6, and the plain-MSE entropy term (unbounded in the
    *prediction*) exploded the loss/grad (entropy_loss -> 6322, grad_norm -> 1.4e5).
    The robust (Huber-style) loss bounds both regardless of how far the head strays.
    """
    from stratego_trainer import setup_loss as SL

    net = S.ArrangementTransformer.from_config(S.SetupConfig())
    mx.eval(net.parameters())
    # Drive the ent head to a huge-magnitude output, as it does after ~50 iters.
    p = net.parameters()
    p["ent_out"]["weight"] = p["ent_out"]["weight"] * 1.0 + 5.0  # large weights
    p["ent_out"]["bias"] = p["ent_out"]["bias"] - 80.0  # huge negative offset
    net.update(p)
    mx.eval(net.parameters())

    counts = list(S.spec.CLASSIC_PIECE_COUNTS)
    flag = S.spec.FLAG_IDX
    types = [i for i, c in enumerate(counts) if i != flag for _ in range(c)] + [flag]
    s1 = np.zeros((40, 14), np.float32)
    s1[np.arange(40), np.array(types)] = 1.0
    seq = mx.array(np.stack([s1 for _ in range(16)]))
    outcome = mx.array(np.zeros(16, np.float32))
    cfg = TrainConfig()
    base = setup_baseline(net, seq, cfg.arr_reg_norm)
    mx.eval(base)
    # ent head is way out of range, but the entropy-advantage stays bounded.
    assert np.abs(np.array(base["reg_adv"])).max() <= SL.MAX_FUTURE_NLL + 1e-3

    batch = {
        "seq": seq, "outcome": outcome, "reg_temp": cfg.setup_temperature(50),
        "old_log_probs": base["log_probs"], "old_value_scalar": base["value_scalar"],
        "reg_returns": base["reg_returns"], "reg_adv": base["reg_adv"],
    }
    (loss, stats), grads = nn.value_and_grad(net, lambda n: setup_loss_and_stats(n, batch, cfg))(net)
    grads["start_token"] = mx.zeros_like(grads["start_token"])
    _, gnorm = optim.clip_grad_norm(grads, cfg.arr_max_grad_norm)
    assert np.isfinite(float(loss)) and np.isfinite(float(gnorm))
    # Robust loss: the per-element entropy gradient is bounded to +/-2*delta, so even a
    # head that is ~80 off-target keeps the loss in the hundreds, never the thousands.
    assert float(stats["entropy_loss"]) < 1000.0
    # A plain MSE on this head would be ~ (80 ** 2) = 6400; robust loss is far smaller.
    assert float(stats["entropy_loss"]) < 0.5 * (80.0 ** 2)


def test_self_heal_reverts_corrupted_net_and_backs_off_lr():
    from stratego_trainer.train import _all_finite, _self_heal, _snapshot

    net = nn.Linear(4, 4)
    opt = optim.AdamW(learning_rate=1e-3)

    def lf(n):
        return (n(mx.ones((2, 4))) ** 2).sum()

    _, g = nn.value_and_grad(net, lf)(net)
    opt.update(net, g)
    mx.eval(net.parameters(), opt.state)
    snap = _snapshot(net, opt)
    good_w = np.array(net.parameters()["weight"])
    assert _all_finite(net.parameters())

    # Corrupt the live net, then self-heal with a non-finite pass signal.
    p = net.parameters()
    p["weight"] = p["weight"] * float("nan")
    net.update(p)
    mx.eval(net.parameters())
    assert not _all_finite(net.parameters())

    class _Ema:
        def __init__(self):
            self.shadow = {}
            self.updated = 0

        def update(self, _):
            self.updated += 1

    ema = _Ema()
    cfg = TrainConfig()
    new_snap, new_scale, nan, stage = _self_heal(net, opt, ema, snap, pass_nan=False, applied=False,
                                                 lr=1e-3, lr_scale=1.0, cfg=cfg)
    assert nan is True
    assert stage == "param:weight"  # localizes which tensor went non-finite
    assert ema.updated == 0  # never fold a corrupted net into the EMA
    assert new_scale == pytest.approx(cfg.lr_backoff)  # LR scaled down
    assert new_snap is snap  # snapshot preserved on revert
    assert _all_finite(net.parameters())
    assert np.allclose(np.array(net.parameters()["weight"]), good_w)  # reverted to last good


def test_self_heal_healthy_folds_ema_and_recovers_lr():
    from stratego_trainer.train import _self_heal, _snapshot

    net = nn.Linear(4, 4)
    opt = optim.AdamW(learning_rate=1e-3)

    def lf(n):
        return (n(mx.ones((2, 4))) ** 2).sum()

    _, g = nn.value_and_grad(net, lf)(net)
    opt.update(net, g)
    mx.eval(net.parameters(), opt.state)
    snap0 = _snapshot(net, opt)

    class _Ema:
        def __init__(self):
            self.shadow = {}
            self.updated = 0

        def update(self, _):
            self.updated += 1

    ema = _Ema()
    cfg = TrainConfig()
    new_snap, new_scale, nan, stage = _self_heal(net, opt, ema, snap0, pass_nan=False, applied=True,
                                                 lr=1e-3, lr_scale=0.5, cfg=cfg)
    assert nan is False
    assert stage == ""  # healthy -> no nan stage
    assert ema.updated == 1  # healthy net folds into the EMA
    assert new_scale == pytest.approx(min(0.5 * cfg.lr_recover, 1.0))  # LR recovers toward 1.0
    assert new_snap is not snap0  # snapshot refreshed to the new good state
