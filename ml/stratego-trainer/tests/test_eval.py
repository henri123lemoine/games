"""Tests for the eval harness's net-forward -> sim-logits bridge (`_net_logits`)."""

import numpy as np
import mlx.core as mx
import pytest

import stratego_nets as S
from stratego_nets.spec import MOVE_IN_DIM, N_ACTION, N_OCCUPIABLE_CELL

pytest.importorskip("stratego_sim")
from stratego_trainer.eval import _net_logits  # noqa: E402


def test_move_logits_ceiling_survives_a_deranged_net():
    """A net emitting an absurdly large LEGAL logit (a stand-in for a collapsed
    checkpoint) must not overflow `_net_logits`'s temperature scaling to inf/NaN
    -- the eval harness has to be able to MEASURE a broken checkpoint, not crash
    on it. (2026-07-05: a valrun3 checkpoint deep in its mid-run draw-collapse
    reproducibly crashed eval_ckpt.py's Metal forward here, preceded by an
    "overflow in multiply" RuntimeWarning.)"""
    net = S.MoveTransformer.from_config(S.MoveConfig())
    obs = np.array(mx.random.normal((2, N_OCCUPIABLE_CELL, MOVE_IN_DIM)))
    legal = np.ones((2, N_ACTION), dtype=bool)
    real_out = net(mx.array(obs), legal_mask=mx.array(legal))

    class Deranged:
        """Stands in for a net whose Linear output has blown up: a real
        value head (so the value-prob math stays exercised) but one absurdly
        large *legal* move logit, the shape a collapsed checkpoint produced."""

        def __call__(self, *_args, **_kwargs):
            logits = np.array(real_out["move_logits"])
            # float32 max is ~3.4e38; this value times even a modest inv_t would
            # overflow the OLD code's unclamped `* inv_t` multiply itself.
            logits[0, 0] = 1e35
            return {"move_logits": mx.array(logits), "value_logp": real_out["value_logp"]}

    with np.errstate(over="raise"):
        logits, vals, probs = _net_logits(Deranged(), None, obs, legal, "move", temperature=0.25)
    assert np.isfinite(logits).all()
    assert np.isfinite(vals).all()
    assert np.isfinite(probs).all()
