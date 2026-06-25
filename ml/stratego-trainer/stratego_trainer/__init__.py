"""stratego_trainer: the Ataraxos move-RL + co-trained setup self-play loop (MLX).

Drives the verified Rust sim (`stratego_sim` bridge) with the `stratego_nets`
MLX transformers, faithful to ATARAXOS_SPEC §4.1/§4.2 — PPO-clip, magnet-KL,
categorical value-CE, advantage filtering, and the dynamic-damping power-law
schedules. Run it:

    python -m stratego_trainer.train --envs 1024 --iters 400 --run-name smoke
"""

from .config import TrainConfig, power_schedule
from .rundir import RunDir, load_checkpoint

__all__ = ["TrainConfig", "power_schedule", "RunDir", "load_checkpoint", "train"]


def train(cfg):
    """Run the self-play training loop (lazy import keeps `python -m
    stratego_trainer.train` from importing the loop twice)."""
    from .train import train as _train

    return _train(cfg)
