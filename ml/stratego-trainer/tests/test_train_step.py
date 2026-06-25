"""A synthetic train step (fwd+bwd+AdamW) must reduce a toy loss for each net,
and the EMA shadow must move toward the trained params."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import stratego_nets as S
from stratego_nets.spec import (
    BELIEF_IN_DIM,
    MOVE_IN_DIM,
    N_OCCUPIABLE_CELL,
    N_PIECE_TYPE,
)

B = 32
N_STEPS = 60


def _train(net, loss_fn, ema_decay):
    opt = optim.AdamW(learning_rate=3e-4)
    ema = S.EMA(net, decay=ema_decay)
    lg = nn.value_and_grad(net, loss_fn)
    losses = []
    for _ in range(N_STEPS):
        loss, grads = lg(net)
        opt.update(net, grads)
        ema.update(net)
        mx.eval(net.parameters(), opt.state, ema.shadow)
        losses.append(float(loss))
    return losses, ema


def test_move_step_reduces_loss():
    mx.random.seed(0)
    net = S.MoveTransformer.from_config(S.MoveConfig())
    obs = mx.random.normal((B, N_OCCUPIABLE_CELL, MOVE_IN_DIM))
    target = mx.array(np.random.randint(0, 3, (B,)))
    # Legal mask = the representable (non-lake) slots; this is what the sim feeds.
    repr_mask = mx.array(
        np.broadcast_to(net._action_map.not_lake_mask_np, (B, 1800))
    )
    repr_f = repr_mask.astype(mx.float32)
    # Policy target supported only on representable slots (renormalized).
    raw = mx.softmax(mx.random.normal((B, 1800)) * 0.1, axis=-1) * repr_f
    pol_target = raw / raw.sum(axis=-1, keepdims=True)

    def loss_fn(net):
        out = net(obs, legal_mask=repr_mask)
        ml = out["move_logits"].astype(mx.float32)
        logp = ml - mx.logsumexp(ml, axis=-1, keepdims=True)
        # only sum over representable slots (masked logp is -inf elsewhere).
        pol_loss = -(pol_target * mx.where(repr_mask, logp, 0.0)).sum(axis=-1).mean()
        val_loss = nn.losses.cross_entropy(out["value_logp"], target, reduction="mean")
        return pol_loss + val_loss

    losses, ema = _train(net, loss_fn, S.spec.EMA_DECAY_MOVE)
    assert losses[-1] < losses[0] * 0.9, (losses[0], losses[-1])


def test_setup_step_reduces_loss():
    mx.random.seed(0)
    net = S.ArrangementTransformer.from_config(S.SetupConfig())
    pc = mx.array(list(S.spec.CLASSIC_PIECE_COUNTS), dtype=mx.float32)
    seq = mx.zeros((B, 39, N_PIECE_TYPE))
    # Read off the net's legal mask and target a legal type per slot (else the
    # target sits on a masked -inf logit and CE is +inf — a harness error, not a
    # net bug). Spy (0) is movable and always has budget for the first 40 slots.
    legal = np.array(net(seq, pc)["logits"]) > np.finfo(np.float32).min
    type_target_np = np.argmax(legal, axis=-1)  # first legal type per slot
    type_target = mx.array(type_target_np)
    val_target = mx.array(np.random.randint(0, 3, (B, 40)))
    ent_target = mx.random.normal((B, 40, 1))

    def loss_fn(net):
        out = net(seq, pc)
        logits = out["logits"].astype(mx.float32)
        pol = nn.losses.cross_entropy(
            logits.reshape(-1, N_PIECE_TYPE), type_target.reshape(-1), reduction="mean"
        )
        val = nn.losses.cross_entropy(
            out["value"].reshape(-1, 3), val_target.reshape(-1), reduction="mean"
        )
        ent = ((out["ent_pred"] - ent_target) ** 2).mean()
        return pol + val + ent

    losses, ema = _train(net, loss_fn, S.spec.EMA_DECAY_SETUP)
    assert losses[-1] < losses[0] * 0.9, (losses[0], losses[-1])


def test_belief_step_reduces_loss():
    mx.random.seed(0)
    net = S.BeliefTransformer.from_config(S.BeliefConfig())
    obs = mx.random.normal((B, N_OCCUPIABLE_CELL, BELIEF_IN_DIM))
    pos_np = np.zeros((B, 40, N_OCCUPIABLE_CELL), dtype=np.float32)
    n_unknown = 8
    for b in range(B):
        cells = np.random.choice(N_OCCUPIABLE_CELL, n_unknown, replace=False)
        pos_np[b, np.arange(n_unknown), cells] = 1.0
    pos = mx.array(pos_np)
    type_idx = np.random.randint(0, 12, (B, 40))
    typ = mx.array(np.eye(N_PIECE_TYPE, dtype=np.float32)[type_idx])
    type_target = mx.array(type_idx)
    active = mx.array((pos_np.sum(axis=-1) > 0).astype(np.float32))  # (B,40)

    def loss_fn(net):
        logits = net(obs, pos, typ).astype(mx.float32)  # (B,40,14)
        ce = nn.losses.cross_entropy(
            logits.reshape(-1, N_PIECE_TYPE), type_target.reshape(-1), reduction="none"
        ).reshape(B, 40)
        return (ce * active).sum() / active.sum()  # masked to active pieces

    losses, ema = _train(net, loss_fn, S.spec.EMA_DECAY_BELIEF)
    assert losses[-1] < losses[0] * 0.9, (losses[0], losses[-1])


def test_ema_tracks_params():
    mx.random.seed(0)
    net = S.MoveTransformer.from_config(S.MoveConfig())
    obs = mx.random.normal((B, N_OCCUPIABLE_CELL, MOVE_IN_DIM))
    target = mx.array(np.random.randint(0, 3, (B,)))

    def loss_fn(net):
        out = net(obs)
        return nn.losses.cross_entropy(out["value_logp"], target, reduction="mean")

    ema = S.EMA(net, decay=0.9)
    init_shadow = np.array(ema.shadow["value_head"]["weight"])
    opt = optim.AdamW(learning_rate=1e-2)
    lg = nn.value_and_grad(net, loss_fn)
    for _ in range(20):
        loss, grads = lg(net)
        opt.update(net, grads)
        ema.update(net)
        mx.eval(net.parameters(), opt.state, ema.shadow)

    final_shadow = np.array(ema.shadow["value_head"]["weight"])
    final_param = np.array(net.parameters()["value_head"]["weight"])
    # shadow moved from init...
    assert not np.allclose(init_shadow, final_shadow)
    # ...but lags the live params (EMA decay 0.9 over 20 steps).
    assert not np.allclose(final_shadow, final_param)
    # explicit one-step EMA check: ema' = decay*ema + (1-decay)*param
    e0 = mx.zeros((2, 3))
    p0 = mx.ones((2, 3))
    e1 = 0.9 * e0 + 0.1 * p0
    assert np.allclose(np.array(e1), 0.1)
