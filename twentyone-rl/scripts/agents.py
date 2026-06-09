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
    """Plays a trained Rust `Solver` in one of three modes.

    - ``"greedy"`` (default): the most-probable action of the learned average
      strategy — the stronger table player against a fixed opponent.
    - ``"mixed"``: samples the mixed strategy — the unexploitable equilibrium
      policy whose exploitability the solver measures.
    - ``"search"``: inference-time within-round lookahead on the exact cards
      (perfect-information Monte-Carlo over the hidden card); can outplay the
      table, especially for the abstracted full-game model.
    """

    def __init__(
        self,
        solver: twentyone.Solver,
        seed: int = 0,
        name: str | None = None,
        greedy: bool = True,
        mode: str | None = None,
    ) -> None:
        self.solver = solver
        self.mode = mode if mode is not None else ("greedy" if greedy else "mixed")
        if self.mode not in ("greedy", "mixed", "search"):
            raise ValueError(f"unknown SolverAgent mode: {self.mode!r}")
        self.name = name if name is not None else f"Solver({self.mode})"
        self._rng = random.Random(seed)

    @classmethod
    def load(
        cls,
        path: str,
        seed: int = 0,
        name: str | None = None,
        greedy: bool = True,
        mode: str | None = None,
    ) -> SolverAgent:
        return cls(twentyone.Solver.load(path), seed=seed, name=name, greedy=greedy, mode=mode)

    def act(self, env: twentyone.Env, player: int) -> twentyone.Action:
        if self.mode == "search":
            draw = self.solver.search_draw(env, player)
        else:
            p_draw = self.solver.draw_probability(env, player)
            draw = p_draw >= 0.5 if self.mode == "greedy" else self._rng.random() < p_draw
        return twentyone.Action.Draw if draw else twentyone.Action.Stand
