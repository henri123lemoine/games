"""
Twenty-One card game environment with Rust backend.

This package provides a fast implementation of the Twenty-One card game
suitable for reinforcement learning experiments and game simulations.

The core game logic is implemented in Rust for performance, with a clean
Python API for ease of use.

Example:
    >>> import twentyone
    >>> env = twentyone.Env(seed=42)
    >>> env.start_new_round()
    >>> obs = env.observation(0)
    >>> result = env.step(twentyone.Action.Draw)
"""

from ._twentyone import Action, Env, Observation, RoundOutcome, Solver, StepResult

__version__ = "0.1.0"
__all__ = ["Action", "Env", "Observation", "RoundOutcome", "Solver", "StepResult"]
