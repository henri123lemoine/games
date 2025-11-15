"""Proper MCCFR agent implementation following Lanctot et al. 2009."""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import twentyone


class MCCFR:
    """Monte Carlo Counterfactual Regret Minimization agent.

    Implements a simplified MCCFR algorithm for the Twenty-One game.
    This version uses sampling over full games rather than tree traversal
    for simplicity and compatibility with the environment interface.
    """

    def __init__(self, seed: int = 42):
        """Initialize MCCFR agent."""
        self.seed = seed
        self.random = random.Random(seed)

        # Regret and strategy storage keyed by information set
        self.regret_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=np.float64))
        self.strategy_sum: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(2, dtype=np.float64)
        )

        self.iterations = 0
        self.num_actions = 2  # Draw, Stand

    def encode_information_set(self, obs: twentyone.Observation, player: int) -> str:
        """Encode observation into information set string.

        Information set includes all information available to the player:
        - Own face-up card, face-down card, total, hearts, stood status
        - Opponent's face-up card, hearts, stood status
        - Round number, deck count
        """
        return (
            f"p{player}_"
            f"own({obs.self_face_up},{obs.self_face_down},{obs.self_total},{obs.self_hearts},{obs.self_stood})_"
            f"opp({obs.opp_face_up},{obs.opp_hearts},{obs.opp_stood})_"
            f"round{obs.round}_deck{obs.deck_count}"
        )

    def get_strategy(self, infoset: str) -> np.ndarray:
        """Get current strategy using regret matching."""
        regrets = np.maximum(self.regret_sum[infoset], 0.0)
        normalizing_sum = np.sum(regrets)

        if normalizing_sum > 0:
            strategy = regrets / normalizing_sum
        else:
            # Uniform random strategy if no positive regrets
            strategy = np.ones(self.num_actions, dtype=np.float64) / self.num_actions

        return strategy

    def get_average_strategy(self, infoset: str) -> np.ndarray:
        """Get average strategy over all iterations."""
        avg_strategy = self.strategy_sum[infoset].copy()
        normalizing_sum = np.sum(avg_strategy)

        if normalizing_sum > 0:
            avg_strategy /= normalizing_sum
        else:
            # Uniform if no strategy recorded yet
            avg_strategy = np.ones(self.num_actions, dtype=np.float64) / self.num_actions

        return avg_strategy

    def choose_action(
        self, obs: twentyone.Observation, player: int, round_num: int
    ) -> twentyone.Action:
        """Choose action using average strategy (for evaluation)."""
        infoset = self.encode_information_set(obs, player)
        strategy = self.get_average_strategy(infoset)

        # Sample action according to strategy
        if self.random.random() < strategy[0]:
            return twentyone.Action.Draw
        else:
            return twentyone.Action.Stand

    def play_game_with_sampling(self, update_player: int) -> dict[str, float]:
        """Play a single game and collect information for MCCFR updates."""
        env = twentyone.Env(seed=self.random.randint(0, 2**31))

        # Track all decision points for the update player
        decision_points = []

        while True:
            env.start_new_round()

            while True:
                current_player = env.current_player()
                obs = env.observation(current_player)
                infoset = self.encode_information_set(obs, current_player)

                # Get strategy for this information set
                strategy = self.get_strategy(infoset)

                # Sample action according to strategy
                action_idx = 0 if self.random.random() < strategy[0] else 1
                action = twentyone.Action.Draw if action_idx == 0 else twentyone.Action.Stand

                # Record decision point if this is the update player
                if current_player == update_player:
                    decision_points.append(
                        {
                            "infoset": infoset,
                            "strategy": strategy.copy(),
                            "action_taken": action_idx,
                        }
                    )

                # Take action
                result = env.step(action)

                if result.round_over:
                    if result.game_over:
                        # Game ended, calculate utility
                        if env.hearts(update_player) <= 0:
                            game_utility = -1.0  # Lost
                        else:
                            game_utility = 1.0  # Won

                        return {"utility": game_utility, "decision_points": decision_points}
                    break

    def mccfr_iteration(self, update_player: int) -> float:
        """Single MCCFR iteration updating one player."""
        # Play game and collect data
        game_data = self.play_game_with_sampling(update_player)
        utility = game_data["utility"]
        decision_points = game_data["decision_points"]

        # Update regrets for all decision points
        for point in decision_points:
            infoset = point["infoset"]
            strategy = point["strategy"]
            action_taken = point["action_taken"]

            # Outcome sampling MCCFR: estimate value of each action
            # For the action taken, we observed the utility
            # For actions not taken, we use 0 as baseline (conservative)
            value_estimates = np.zeros(self.num_actions, dtype=np.float64)
            value_estimates[action_taken] = utility

            # Expected value under current strategy
            expected_value = np.dot(strategy, value_estimates)

            # Compute regret for each action
            # regret[a] = value_if_took_a - expected_value_under_strategy
            for action_idx in range(self.num_actions):
                action_regret = value_estimates[action_idx] - expected_value
                self.regret_sum[infoset][action_idx] += action_regret

            # Update strategy sum (for average strategy)
            self.strategy_sum[infoset] += strategy

        return utility

    def train(self, iterations: int = 1000) -> dict[str, Any]:
        """Train the agent for given iterations."""
        training_stats = {"utilities": [], "infosets_learned": 0}

        for i in range(iterations):
            # Alternate which player we update
            update_player = i % 2

            # Run MCCFR iteration
            utility = self.mccfr_iteration(update_player)
            training_stats["utilities"].append(utility)

            if i % 100 == 0:
                print(
                    f"MCCFR iteration {i}, utility: {utility:.4f}, infosets: {len(self.regret_sum)}"
                )

        self.iterations += iterations
        training_stats["infosets_learned"] = len(self.regret_sum)
        training_stats["total_iterations"] = self.iterations

        return training_stats

    def average_policy(self) -> dict[str, Any]:
        """Return average policy information for evaluation."""
        return {
            "agent_type": "mccfr",
            "iterations_trained": self.iterations,
            "infosets_learned": len(self.regret_sum),
            "strategy_type": "average",
        }


def save_policy(policy: dict[str, Any], path: Path) -> None:
    """Save policy to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(policy, f, indent=2)
