"""The (92,92) move-head -> 1800-action mapping must match games/stratego/src/action.rs."""

import mlx.core as mx
import numpy as np
import pytest

from stratego_nets.action_map import ActionLogitMap, create_srcdst_to_env_action_index
from stratego_nets.spec import (
    BOARD_LEN,
    LAKE_INDICES,
    N_ACTION,
    N_OCCUPIABLE_CELL,
)

VALID_CELLS = [i for i in range(100) if i not in set(LAKE_INDICES)]
FULL_TO_REDUCED = {pos: idx for idx, pos in enumerate(VALID_CELLS)}
N_VALID = len(VALID_CELLS)


def _from_abs_p0(src, dst):
    """Port of action.rs::from_abs for player 0 (no POV flip): env action index."""
    sr, sc = divmod(src, BOARD_LEN)
    dr, dc = divmod(dst, BOARD_LEN)
    if src == dst:
        return None
    if sc == dc:
        c = dr - (1 if dr > sr else 0)
    elif sr == dr:
        c = (BOARD_LEN - 1) + (dc - (1 if dc > sc else 0))
    else:
        return None
    return 100 * c + src


KNOWN_PAIRS = [
    (0, 10), (0, 1), (0, 90), (0, 9), (11, 15),
    (33, 3), (99, 9), (99, 90), (25, 28), (60, 64),
]


@pytest.mark.parametrize("src,dst", KNOWN_PAIRS)
def test_known_srcdst_pairs_match_action_rs(src, dst):
    srcdst_to_env = create_srcdst_to_env_action_index()
    env = _from_abs_p0(src, dst)
    assert env is not None
    expect_flat = FULL_TO_REDUCED[src] * N_VALID + FULL_TO_REDUCED[dst]
    assert srcdst_to_env[env] == expect_flat


def test_representable_slots_and_ranges():
    srcdst_to_env = create_srcdst_to_env_action_index()
    repr_mask = srcdst_to_env != -1
    # 1544 of 1800 env slots have a src-dst representation (others touch a lake).
    assert repr_mask.sum() == 1544
    vals = srcdst_to_env[repr_mask]
    assert vals.min() >= 0
    assert vals.max() < N_OCCUPIABLE_CELL * N_OCCUPIABLE_CELL


def test_apply_scatters_grid_into_action_vector():
    m = ActionLogitMap()
    # Build a grid whose every entry is a unique, identifiable value.
    grid_np = np.arange(N_OCCUPIABLE_CELL * N_OCCUPIABLE_CELL, dtype=np.float32)
    grid_np = grid_np.reshape(1, N_OCCUPIABLE_CELL, N_OCCUPIABLE_CELL)
    out = np.array(m.apply(mx.array(grid_np)))[0]  # (1800,)

    # Spot-check a known action lands the right grid value.
    src, dst = 25, 28
    env = _from_abs_p0(src, dst)
    grid_flat = FULL_TO_REDUCED[src] * N_OCCUPIABLE_CELL + FULL_TO_REDUCED[dst]
    assert out[env] == grid_np.reshape(-1)[grid_flat]

    # Lake-touching slots are filled to the dtype min (read as -inf downstream).
    fill = np.finfo(np.float32).min
    lake_slots = np.array(create_srcdst_to_env_action_index()) == -1
    assert np.all(out[lake_slots] == fill)


def test_apply_batched_and_lake_actions_masked():
    m = ActionLogitMap()
    grid = mx.random.normal((3, N_OCCUPIABLE_CELL, N_OCCUPIABLE_CELL))
    out = m.apply(grid)
    assert out.shape == (3, N_ACTION)
    fill = np.finfo(np.float32).min
    lake_slots = np.array(create_srcdst_to_env_action_index()) == -1
    out_np = np.array(out)
    assert np.all(out_np[:, lake_slots] == fill)
