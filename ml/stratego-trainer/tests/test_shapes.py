"""Forward-shape and param-count checks for all three nets, on synthetic inputs."""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

import stratego_nets as S
from stratego_nets.spec import (
    BELIEF_IN_DIM,
    MOVE_IN_DIM,
    N_ACTION,
    N_OCCUPIABLE_CELL,
    N_PIECE_TYPE,
    N_VF_CAT,
)

B = 4


def nparams(model):
    return sum(p.size for _, p in tree_flatten(model.parameters()))


def test_move_forward_shapes():
    net = S.MoveTransformer.from_config(S.MoveConfig())
    obs = mx.random.normal((B, N_OCCUPIABLE_CELL, MOVE_IN_DIM))
    out = net(obs)
    assert out["move_logits"].shape == (B, N_ACTION)
    assert out["value_logp"].shape == (B, N_VF_CAT)
    # value head is a valid log-softmax (rows sum to ~0 in prob space).
    probs = np.exp(np.array(out["value_logp"]))
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-4)


def test_move_legal_mask_applied():
    net = S.MoveTransformer.from_config(S.MoveConfig())
    obs = mx.random.normal((B, N_OCCUPIABLE_CELL, MOVE_IN_DIM))
    mask = mx.array(np.random.rand(B, N_ACTION) > 0.5)
    out = net(obs, legal_mask=mask)
    logits = np.array(out["move_logits"])
    fill = np.finfo(np.float32).min
    assert np.all(logits[~np.array(mask)] == fill)


def test_setup_forward_shapes():
    net = S.ArrangementTransformer.from_config(S.SetupConfig())
    pc = mx.array(list(S.spec.CLASSIC_PIECE_COUNTS), dtype=mx.float32)
    seq = mx.zeros((B, 39, N_PIECE_TYPE))  # prefix; +1 start token -> 40
    out = net(seq, pc)
    assert out["logits"].shape == (B, 40, N_PIECE_TYPE)
    assert out["value"].shape == (B, 40, N_VF_CAT)
    assert out["ent_pred"].shape == (B, 40, 1)


def test_setup_legal_mask_zeroes_exhausted_types_and_flag_left():
    net = S.ArrangementTransformer.from_config(S.SetupConfig())
    pc = mx.array(list(S.spec.CLASSIC_PIECE_COUNTS), dtype=mx.float32)
    seq = mx.zeros((B, 39, N_PIECE_TYPE))
    out = net(seq, pc)
    logits = np.array(out["logits"])
    fill = np.finfo(np.float32).min
    # lake (12) and empty (13) have 0 budget in classic -> always masked.
    assert np.all(logits[:, :, 12] == fill)
    assert np.all(logits[:, :, 13] == fill)
    # flag (10) is masked on the left half (first 5 slots of each row of 10).
    left_slots = [s for s in range(40) if (s % 10) < 5]
    assert np.all(logits[:, left_slots, 10] == fill)


def test_belief_forward_shapes():
    net = S.BeliefTransformer.from_config(S.BeliefConfig())
    obs = mx.random.normal((B, N_OCCUPIABLE_CELL, BELIEF_IN_DIM))
    pos = np.zeros((B, 40, N_OCCUPIABLE_CELL), dtype=np.float32)
    pos[:, np.arange(8), np.arange(8)] = 1.0  # 8 unknown pieces on distinct cells
    typ = np.eye(N_PIECE_TYPE, dtype=np.float32)[np.random.randint(0, 12, (B, 40))]
    out = net(mx.array(obs) if not isinstance(obs, mx.array) else obs,
              mx.array(pos), mx.array(typ))
    assert out.shape == (B, 40, N_PIECE_TYPE)


def test_param_counts_in_budget():
    mv = nparams(S.MoveTransformer.from_config(S.MoveConfig()))
    st = nparams(S.ArrangementTransformer.from_config(S.SetupConfig()))
    bf = nparams(S.BeliefTransformer.from_config(S.BeliefConfig()))
    assert 5e6 <= mv <= 8e6, mv
    assert 2e6 <= st <= 4e6, st
    assert 8e6 <= bf <= 12e6, bf


def test_ref_param_counts_match_paper():
    mv = nparams(S.MoveTransformer.from_config(S.MOVE_REF))
    st = nparams(S.ArrangementTransformer.from_config(S.SETUP_REF))
    bf = nparams(S.BeliefTransformer.from_config(S.BELIEF_REF))
    # paper: move ~14.7M, setup ~12.6M, belief ~57.1M
    assert abs(mv - 14.7e6) < 0.3e6, mv
    assert abs(st - 12.6e6) < 0.3e6, st
    assert abs(bf - 57.1e6) < 0.5e6, bf
