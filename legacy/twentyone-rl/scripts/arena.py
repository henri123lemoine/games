"""Head-to-head and round-robin evaluation with Wilson confidence intervals."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import twentyone
from agents import Agent


def play_game(agent0: Agent, agent1: Agent, seed: int) -> int | None:
    """Play one full game. Returns the winning player (0 or 1), or None for a draw."""
    env = twentyone.Env(seed=seed)
    agents = (agent0, agent1)
    while True:
        env.start_new_round()
        while True:
            player = env.current_player()
            action = agents[player].act(env, player)
            result = env.step(action)
            if result.game_over:
                h0, h1 = env.hearts(0), env.hearts(1)
                if h0 == h1:
                    return None
                return 0 if h0 > h1 else 1
            if result.round_over:
                break


def _wilson_interval(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (robust near 0/1)."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class MatchResult:
    agent0: str
    agent1: str
    games: int
    wins0: int
    wins1: int
    draws: int

    @property
    def score0(self) -> float:
        """Win rate for agent0 counting draws as half (symmetric, sums to 1)."""
        return (self.wins0 + 0.5 * self.draws) / self.games

    @property
    def ci0(self) -> tuple[float, float]:
        return _wilson_interval(self.wins0 + 0.5 * self.draws, self.games)


def run_match(agent0: Agent, agent1: Agent, games: int, seed: int = 0) -> MatchResult:
    """Play `games`, alternating who is seated as player 0 to cancel seat bias."""
    rng = random.Random(seed)
    wins = [0, 0]
    draws = 0
    for i in range(games):
        game_seed = rng.randint(0, 2**31 - 1)
        if i % 2 == 0:
            winner = play_game(agent0, agent1, game_seed)
            mapped = winner
        else:
            winner = play_game(agent1, agent0, game_seed)
            mapped = None if winner is None else 1 - winner
        if mapped is None:
            draws += 1
        else:
            wins[mapped] += 1
    return MatchResult(agent0.name, agent1.name, games, wins[0], wins[1], draws)
