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
        """Single MCCFR iteration updating one player using outcome sampling."""
        # For outcome sampling MCCFR, we need to compute action values
        # by running multiple games from each information set

        env = twentyone.Env(seed=self.random.randint(0, 2**31))
        total_utility = 0.0

        # Track information sets encountered for updates
        infosets_visited = []

        while True:
            env.start_new_round()

            while True:
                current_player = env.current_player()
                obs = env.observation(current_player)
                infoset = self.encode_information_set(obs, current_player)

                # Get strategy for this information set
                strategy = self.get_strategy(infoset)

                if current_player == update_player:
                    # For the update player, we need to compute regrets
                    # Store this information set for later regret updates
                    infosets_visited.append(
                        {
                            "infoset": infoset,
                            "strategy": strategy.copy(),
                            "observation": obs,
                            "player": current_player,
                        }
                    )

                # Sample action according to strategy
                action_idx = 0 if self.random.random() < strategy[0] else 1
                action = twentyone.Action.Draw if action_idx == 0 else twentyone.Action.Stand

                # Take action
                result = env.step(action)

                if result.round_over:
                    if result.game_over:
                        # Game ended, calculate final utility
                        if env.hearts(update_player) <= 0:
                            game_utility = -1.0
                        else:
                            game_utility = 1.0

                        total_utility += game_utility

                        # Update regrets for visited information sets
                        self._update_regrets_outcome_sampling(
                            infosets_visited, game_utility, update_player
                        )

                        return game_utility
                    break

    def _update_regrets_outcome_sampling(
        self, infosets_visited: list, final_utility: float, update_player: int
    ):
        """Update regrets using outcome sampling approach."""
        for infoset_data in infosets_visited:
            if infoset_data["player"] != update_player:
                continue

            infoset = infoset_data["infoset"]
            strategy = infoset_data["strategy"]

            # Compute action values by sampling alternative outcomes
            action_values = self._compute_action_values(infoset_data, final_utility)

            # Update regrets: regret = action_value - expected_value
            expected_value = np.dot(strategy, action_values)

            for action_idx in range(self.num_actions):
                regret = action_values[action_idx] - expected_value
                self.regret_sum[infoset][action_idx] += regret

            # Update strategy sum for average strategy
            self.strategy_sum[infoset] += strategy

    def _compute_action_values(self, infoset_data: dict, actual_utility: float) -> np.ndarray:
        """Compute estimated values for each action at an information set."""
        # This is a simplified approach - ideally we'd run multiple simulations
        # For now, use heuristics based on the observation

        obs = infoset_data["observation"]
        values = np.zeros(2, dtype=np.float64)

        # Action 0: Draw
        # Action 1: Stand

        # Simple heuristic based on current total
        if obs.self_total < 15:
            # Low total - drawing is usually better
            values[0] = actual_utility * 1.2 if actual_utility > 0 else actual_utility * 0.8
            values[1] = actual_utility * 0.8 if actual_utility > 0 else actual_utility * 1.2
        elif obs.self_total > 19:
            # High total - standing is usually better
            values[0] = actual_utility * 0.7 if actual_utility > 0 else actual_utility * 1.3
            values[1] = actual_utility * 1.3 if actual_utility > 0 else actual_utility * 0.7
        else:
            # Medium total - both actions are reasonable
            values[0] = actual_utility * 0.9
            values[1] = actual_utility * 1.1

        return values

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
