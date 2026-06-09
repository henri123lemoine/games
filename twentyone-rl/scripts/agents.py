"""Agents for Twenty-One: a uniform `act(env, player)` interface.

The Rust environment is the single source of truth, so agents act directly on a
live `twentyone.Env`. This lets the Nash agent index its strategy by the env's
lossless `sufficient_key`, which an `Observation` alone cannot reconstruct (it
omits the unseen-card set).
"""

from __future__ import annotations

import random
from typing import Protocol

import twentyone


class Agent(Protocol):
    """Anything that can choose an action for `player` in a game state."""

    name: str

    def act(self, env: twentyone.Env, player: int) -> twentyone.Action: ...


class RandomAgent:
    """Draws or stands uniformly at random."""

    def __init__(self, seed: int = 0, name: str = "Random") -> None:
        self.name = name
        self._rng = random.Random(seed)

    def act(self, env: twentyone.Env, player: int) -> twentyone.Action:
        if self._rng.random() < 0.5:
            return twentyone.Action.Draw
        return twentyone.Action.Stand


class ThresholdAgent:
    """Draws while the current total is below `threshold`, else stands."""

    def __init__(self, threshold: int = 17, name: str | None = None) -> None:
        self.threshold = threshold
        self.name = name or f"Threshold({threshold})"

    def act(self, env: twentyone.Env, player: int) -> twentyone.Action:
        obs = env.observation(player)
        if obs.self_total < self.threshold:
            return twentyone.Action.Draw
        return twentyone.Action.Stand


class SolverAgent:
    """Plays the (near-)Nash strategy from a trained Rust `Solver`.

    With ``greedy=True`` (default) it plays the most-probable action of the
    learned average strategy, which is the stronger player against a fixed
    opponent. With ``greedy=False`` it samples the mixed strategy — the
    unexploitable equilibrium policy whose exploitability the solver measures.
    """

    def __init__(
        self,
        solver: twentyone.Solver,
        seed: int = 0,
        name: str = "Solver",
        greedy: bool = True,
    ) -> None:
        self.solver = solver
        self.name = name
        self.greedy = greedy
        self._rng = random.Random(seed)

    @classmethod
    def load(
        cls, path: str, seed: int = 0, name: str = "Solver", greedy: bool = True
    ) -> SolverAgent:
        return cls(twentyone.Solver.load(path), seed=seed, name=name, greedy=greedy)

    def act(self, env: twentyone.Env, player: int) -> twentyone.Action:
        p_draw = self.solver.draw_probability(env, player)
        draw = p_draw >= 0.5 if self.greedy else self._rng.random() < p_draw
        return twentyone.Action.Draw if draw else twentyone.Action.Stand
