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

DEFAULT_BATCH = 1024
