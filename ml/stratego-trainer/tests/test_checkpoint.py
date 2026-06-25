"""safetensors save/load must round-trip weights AND optimizer state (and EMA)."""

import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

import stratego_nets as S
from stratego_nets.spec import MOVE_IN_DIM, N_OCCUPIABLE_CELL

B = 8


def _flat(tree):
    return {k: np.array(v) for k, v in tree_flatten(tree)}


def test_roundtrip_weights_optimizer_ema():
    mx.random.seed(0)
    net = S.MoveTransformer.from_config(S.MoveConfig())
    opt = optim.AdamW(learning_rate=1e-3)
    ema = S.EMA(net, decay=0.999)

    obs = mx.random.normal((B, N_OCCUPIABLE_CELL, MOVE_IN_DIM))
    target = mx.array(np.random.randint(0, 3, (B,)))

    def loss_fn(net):
        return nn.losses.cross_entropy(net(obs)["value_logp"], target, reduction="mean")

    lg = nn.value_and_grad(net, loss_fn)
    # take a few steps so optimizer state (m, v, step) is non-trivial.
    for _ in range(5):
        loss, grads = lg(net)
        opt.update(net, grads)
        ema.update(net)
        mx.eval(net.parameters(), opt.state, ema.shadow)

    saved_model = _flat(net.parameters())
    saved_opt = _flat(opt.state)
    saved_ema = _flat(ema.shadow_params())

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "ckpt.safetensors")
        S.save(path, net, optimizer=opt, ema=ema, metadata={"step": 5})

        # Fresh net/opt/ema, then load.
        net2 = S.MoveTransformer.from_config(S.MoveConfig())
        opt2 = optim.AdamW(learning_rate=1e-3)
        ema2 = S.EMA(net2, decay=0.999)
        # prime opt2 state shape by a dummy step so keys exist (AdamW lazily inits).
        lg2 = nn.value_and_grad(net2, loss_fn)
        l2, g2 = lg2(net2)
        opt2.update(net2, g2)
        mx.eval(net2.parameters(), opt2.state)

        S.load(path, net2, optimizer=opt2, ema=ema2)

    loaded_model = _flat(net2.parameters())
    loaded_opt = _flat(opt2.state)
    loaded_ema = _flat(ema2.shadow_params())

    for k, v in saved_model.items():
        assert np.allclose(v, loaded_model[k]), f"weight mismatch {k}"
    for k, v in saved_ema.items():
        assert np.allclose(v, loaded_ema[k]), f"ema mismatch {k}"
    for k, v in saved_opt.items():
        assert k in loaded_opt, f"missing opt key {k}"
        assert np.allclose(v, loaded_opt[k]), f"opt mismatch {k}"


def test_loaded_model_produces_identical_forward():
    mx.random.seed(1)
    net = S.MoveTransformer.from_config(S.MoveConfig())
    obs = mx.random.normal((B, N_OCCUPIABLE_CELL, MOVE_IN_DIM))
    ref = np.array(net(obs)["move_logits"])

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "w.safetensors")
        S.save(path, net)
        net2 = S.MoveTransformer.from_config(S.MoveConfig())
        # before load: different random init -> different output
        assert not np.allclose(np.array(net2(obs)["move_logits"]), ref, atol=1e-3)
        S.load(path, net2)

    got = np.array(net2(obs)["move_logits"])
    # finite slots match exactly; ignore the -inf lake fills.
    finite = np.isfinite(ref)
    assert np.allclose(got[finite], ref[finite], atol=1e-4)
