"""stratego_nets: the three Ataraxos Stratego transformers in MLX.

Public API (imported by the M5 training loop and M6 search):

    from stratego_nets import (
        MoveTransformer, ArrangementTransformer, BeliefTransformer,
        EMA, save, load,
        MoveConfig, SetupConfig, BeliefConfig,
    )

Construct a net from a config, call it on synthetic/sim obs, train with an
mlx.optimizers optimizer, EMA-update each step, and checkpoint with save/load.
See README.md for the obs-array contract and net signatures.
"""

from .action_map import ActionLogitMap, create_srcdst_to_env_action_index
from .checkpoint import load, save
from .config import (
    BELIEF_REF,
    DEFAULT_BATCH,
    MOVE_REF,
    SETUP_REF,
    BeliefConfig,
    MoveConfig,
    SetupConfig,
)
from .ema import EMA
from .nets import ArrangementTransformer, BeliefTransformer, MoveTransformer
from . import spec

__all__ = [
    "MoveTransformer",
    "ArrangementTransformer",
    "BeliefTransformer",
    "EMA",
    "save",
    "load",
    "MoveConfig",
    "SetupConfig",
    "BeliefConfig",
    "MOVE_REF",
    "SETUP_REF",
    "BELIEF_REF",
    "DEFAULT_BATCH",
    "ActionLogitMap",
    "create_srcdst_to_env_action_index",
    "spec",
]
