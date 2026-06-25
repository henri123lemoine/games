"""Shapes, channel widths, and sizing constants for the three Ataraxos nets.

All numbers trace to ATARAXOS_SPEC.md §3 and ENCODING_SPEC.md §0, which in turn
trace to the reference repo (`pyengine/utils/constants.py`, `stratego_board.h`).
Citations are in comments; this module is the single source of truth the nets and
tests import from.
"""

# ---- board / action geometry (ENCODING_SPEC §1, action.rs) ----
BOARD_LEN = 10
N_BOARD_CELL = 100
N_LAKE_CELL = 8
N_OCCUPIABLE_CELL = 92  # 100 - 8 lakes; the move/belief token count
LAKE_INDICES = (42, 43, 46, 47, 52, 53, 56, 57)  # constants.py:53

N_ACTION = 1800  # the src-displacement env action space (action.rs)
MAX_N_POSSIBLE_DST = 2 * (BOARD_LEN - 1)  # 18 displacement codes c in [0,18)

# ---- piece types (stratego_board.h:12; one-hot width for setup + belief) ----
N_PIECE_TYPE = 14  # spy..bomb=12, lake, empty
FLAG_IDX = 10
# CLASSIC_INITIAL_COUNTS (stratego_board.h:165), spy..bomb then lake,empty; sums to 40.
CLASSIC_PIECE_COUNTS = (1, 8, 5, 4, 4, 4, 3, 2, 1, 1, 1, 6, 0, 0)
ARRANGEMENT_SIZE = 40  # 4 rows x 10 cols home grid
N_ARRANGEMENT_ROW = 4
N_ARRANGEMENT_COL = 10

# ---- value head (constants.py:97) ----
CATEGORICAL_AGGREGATION = (-1, 0, 1)  # lose, tie, win
N_VF_CAT = 3

# ---- per-token input widths (ENCODING_SPEC §0) ----
NUM_BOARD_STATE_CHANNELS = 355
N_PIECE_ID = 256
MOVE_PLANE_HISTORY = 32
BELIEF_PLANE_HISTORY = 86
MOVE_IN_DIM = NUM_BOARD_STATE_CHANNELS + MOVE_PLANE_HISTORY + N_PIECE_ID  # 643
BELIEF_IN_DIM = NUM_BOARD_STATE_CHANNELS + BELIEF_PLANE_HISTORY + N_PIECE_ID  # 697

# ---- EMA decays (rl.py:53 move/setup; belief.py:43 belief) ----
EMA_DECAY_MOVE = 0.999
EMA_DECAY_SETUP = 0.999
EMA_DECAY_BELIEF = 0.99


def head_dim_to_embed(n_head: int, head_dim: int) -> int:
    """embed_dim = n_head * head_dim, matching the reference's per-head sizing."""
    return n_head * head_dim
