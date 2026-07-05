"""Move-head (92x92) -> 1800-action mapping, ported from the reference LogitConverter.

The move net's key-query head emits a (B, 92, 92) grid of "piece at occupiable
cell i -> occupiable cell j" logits (lakes already dropped, so 92 not 100). The
Rust sim (and the reference env) index actions in a *src-displacement* space of
1800 slots: `action = 100*c + src_pov`, where `c in [0,18)` is the displacement
code (`c<9` vertical destination-row, `c>=9` horizontal destination-col, the
source row/col skipped) and `src_pov` is the source cell in the acting player's
POV. See `games/stratego/src/action.rs` (`to_abs`/`from_abs`).

This module builds, once, the gather index that scatters the flattened 92*92
src-dst logits into the 1800-slot env action vector, exactly reproducing
`pyengine/networks/utils.py::create_srcdst_to_env_action_index` +
`LogitConverter.forward`. The result lines up element-for-element with the sim's
legal-action masks so illegal slots can be set to -inf before the categorical.

Mapping derivation (matches action.rs `from_abs`):
  env action index = c * N_BOARD_CELL + src_cell   (the flattened (c, src) grid)
  where for a vertical move to absolute row `new_row != src_row`:
      c = new_row        if new_row < src_row else new_row - 1   (skip src row)
  and for a horizontal move to absolute col `new_col != src_col`:
      c = (BOARD_LEN - 1) + (new_col if new_col < src_col else new_col - 1)
The src-dst grid is over the 92 *occupiable* cells (full board minus the 8 lakes),
so both src and dst are remapped through full->reduced cell indices. Lake-touching
actions have no src-dst slot and stay at the masked-out fill value.
"""

import numpy as np

from .spec import BOARD_LEN, LAKE_INDICES, MAX_N_POSSIBLE_DST, N_ACTION, N_BOARD_CELL


def create_srcdst_to_env_action_index(excluded=LAKE_INDICES) -> np.ndarray:
    """Index from env action space (1800,) into the flattened reduced src-dst grid.

    Returns an int64 array of length N_ACTION. Value -1 means the env action is a
    lake-touching move with no src-dst representation. Otherwise the value is the
    flattened index `reduced_src * n_valid + reduced_dst` into the (92*92,) grid.

    This is a verbatim port of the reference `create_srcdst_to_env_action_index`.
    """
    excluded_set = set(excluded)
    valid_cells = [i for i in range(N_BOARD_CELL) if i not in excluded_set]
    n_valid = len(valid_cells)
    full_to_reduced = {pos: idx for idx, pos in enumerate(valid_cells)}

    idx = np.full((MAX_N_POSSIBLE_DST, N_BOARD_CELL), -1, dtype=np.int64)

    for src in range(N_BOARD_CELL):
        if src in excluded_set:
            continue
        src_row, src_col = src // BOARD_LEN, src % BOARD_LEN

        for new_row in range(BOARD_LEN):
            if new_row == src_row:
                continue
            dst = new_row * BOARD_LEN + src_col
            if dst in excluded_set:
                continue
            c = new_row if new_row < src_row else new_row - 1
            idx[c, src] = full_to_reduced[src] * n_valid + full_to_reduced[dst]

        for new_col in range(BOARD_LEN):
            if new_col == src_col:
                continue
            dst = src_row * BOARD_LEN + new_col
            if dst in excluded_set:
                continue
            c = (BOARD_LEN - 1) + (new_col if new_col < src_col else new_col - 1)
            idx[c, src] = full_to_reduced[src] * n_valid + full_to_reduced[dst]

    return idx.reshape(-1)


class ActionLogitMap:
    """Reusable converter from a (B, 92, 92) src-dst grid to (B, 1800) env logits.

    Precomputes the gather index and the not-lake mask once. ``apply`` works on an
    MLX array; ``not_lake_mask`` exposes which of the 1800 slots are representable.
    """

    def __init__(self):
        srcdst_to_env = create_srcdst_to_env_action_index()
        self.not_lake_mask_np = srcdst_to_env != -1  # (1800,)
        # For each representable env slot, the flat index into the 92*92 grid.
        self.reparam_actions_np = srcdst_to_env[self.not_lake_mask_np]  # (n_repr,)
        # Positions in the 1800 vector that receive a value (the True indices).
        self.target_slots_np = np.nonzero(self.not_lake_mask_np)[0]  # (n_repr,)

    def apply(self, grid):
        """(..., 92, 92) MLX logits -> (..., 1800) MLX logits.

        Non-representable (lake-touching) slots are filled with the smallest
        finite value of the dtype, so they read as -inf to the downstream softmax
        even before a legal mask is applied. The reference's `LogitConverter`
        fills these slots with 0 instead, relying entirely on legal_action_mask
        to blank them -- behaviorally identical here because the real game
        engine's legal mask never marks a lake-touching slot legal in the
        first place; this would only diverge from the reference if this net
        were ever called with a legal_mask that doesn't already exclude lake
        slots (audit-nets, confirmed).
        """
        import mlx.core as mx

        target = mx.array(self.target_slots_np)
        gather = mx.array(self.reparam_actions_np)
        flat = grid.reshape(*grid.shape[:-2], -1)  # (..., 8464)
        gathered = mx.take(flat, gather, axis=-1)  # (..., n_repr)
        fill = mx.finfo(grid.dtype).min
        out = mx.full((*grid.shape[:-2], N_ACTION), fill, dtype=grid.dtype)
        out[..., target] = gathered
        return out
