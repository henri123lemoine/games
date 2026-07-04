"""Net sizing presets, tuned to a ~1-day tabula-rasa run on the M5 Max (64 GB).

These are the depth/width knobs M5 (training loop) can sweep. The "ref" presets
reproduce the paper's full sizes (move 14.7M / setup 12.6M / belief 57M); the
default presets below scale toward the ~1-day budget called for in the milestone
(move ~5-8M, setup ~2-4M, belief ~8-12M). Batch target 1024-2048; b=4096 is
memory-bound on 64 GB (see BENCHMARK.md).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MoveConfig:
    depth: int = 6
    embed_dim: int = 256
    n_head: int = 8


@dataclass(frozen=True)
class SetupConfig:
    depth: int = 4
    embed_dim: int = 256
    n_head: int = 8
    force_handedness: bool = True


@dataclass(frozen=True)
class BeliefConfig:
    n_encoder_layer: int = 4
    n_decoder_block: int = 4
    embed_dim: int = 256
    n_head: int = 8
    n_piece: int = 40
    dropout: float = 0.2  # ~9.7M; the only net with dropout


# Full-size reference presets (paper sizes), available for ablation.
MOVE_REF = MoveConfig(depth=8, embed_dim=384, n_head=8)
SETUP_REF = SetupConfig(depth=4, embed_dim=512, n_head=8)
BELIEF_REF = BeliefConfig(
    n_encoder_layer=6, n_decoder_block=6, embed_dim=512, n_head=8
)

# An intermediate point between the default (smoke-scale) and ref (paper-scale)
# presets, used by the throughput calibration sweep (README/BENCHMARK.md).
MOVE_MID = MoveConfig(depth=8, embed_dim=320, n_head=8)
SETUP_MID = SetupConfig(depth=4, embed_dim=384, n_head=8)

# The named move/setup size presets a run can select (`TrainConfig.net_size` /
# `--net-size`), and the single source of truth for resolving a size name back
# to configs — both the trainer and the standalone eval/play tooling key off
# this so a checkpoint's declared size always reconstructs the matching net.
NET_SIZES = {
    "default": (MoveConfig(), SetupConfig()),
    "mid": (MOVE_MID, SETUP_MID),
    "ref": (MOVE_REF, SETUP_REF),
}

DEFAULT_BATCH = 1024
