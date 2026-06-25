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
    keep = advantage_filter_mask(adv, rate=0.75, thresh=0.01)
    # threshold = max(q75(|adv|), 0.01); the two large-|adv| rows survive.
    assert keep[2] and keep[3]
    assert not keep[0] and not keep[1] and not keep[5]


def _drive(s, steps):
    for _ in range(steps):
        b = s.collect()
        nm = b["move_obs"].shape[0]
        nd = b["deploy_obs"].shape[0]
        s.commit(
            np.zeros((nm, sim.N_ACTION), np.float32), np.zeros(nm, np.float32),
            np.zeros((nd, sim.DEPLOY_WIDTH), np.float32), np.zeros(nd, np.float32),
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
    keep = advantage_filter_mask(d["advantage"], 0.75, 0.01)
    idx = np.nonzero(keep)[0]
    if idx.size < 8:
        pytest.skip("too few filtered transitions in this short drive")
    idx = idx[:256]
    batch = {
        "obs": mx.array(d["obs"][idx]),
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
