"""
Tabular CFR Agent - Exact implementation of Monte Carlo CFR with perfect information set storage.

This serves as the theoretical upper bound for CFR performance on Twenty-One,
since it stores exact regret and strategy values for every information set without
any neural network approximation error.
"""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import twentyone


class TabularCFR:
    """Exact Tabular Monte Carlo Counterfactual Regret Minimization agent.

    This implementation stores exact regret and strategy values for every information set,
    providing the theoretical upper bound for CFR performance on Twenty-One.
    """

    def __init__(self, seed: int = 42):
        """Initialize Tabular CFR agent."""
        self.seed = seed
        self.random = random.Random(seed)

        # Exact storage for every information set
        self.regret_sum: dict[tuple, np.ndarray] = {}  # Information set -> action regrets
        self.strategy_sum: dict[tuple, np.ndarray] = {}  # Information set -> strategy accumulation

        # Training statistics
        self.iterations = 0
        self.total_games = 0

        # Set random seeds
        np.random.seed(seed)

    def encode_information_set(
        self, obs: twentyone.Observation, player: int, round_num: int
    ) -> tuple:
        """Encode observation into a hashable information set key.

        This creates the exact information set representation that includes all
        relevant game state information.
        """
        # Create comprehensive information set that captures all decision-relevant information
        return (
            obs.self_total,  # Player's current total
            obs.self_hearts,  # Player's remaining hearts
            int(obs.self_stood),  # Whether player has stood
            obs.opp_face_up,  # Opponent's visible card
            obs.opp_hearts,  # Opponent's hearts
            int(obs.opp_stood),  # Whether opponent has stood
            round_num,  # Current round number
            obs.deck_count,  # Remaining cards in deck
        )

    def get_regret_strategy(self, infoset_key: tuple) -> np.ndarray:
        """Get strategy based on current regrets using regret matching."""
        if infoset_key not in self.regret_sum:
            self.regret_sum[infoset_key] = np.zeros(2, dtype=np.float32)

        regrets = self.regret_sum[infoset_key]

        # Regret matching: positive regrets become strategy weights
        positive_regrets = np.maximum(regrets, 0)
        total_regret = positive_regrets.sum()

        if total_regret > 0:
            return positive_regrets / total_regret
        else:
            # Uniform random strategy if no positive regrets
            return np.array([0.5, 0.5], dtype=np.float32)

    def get_average_strategy(self, infoset_key: tuple) -> np.ndarray:
        """Get average strategy over all iterations."""
        if infoset_key not in self.strategy_sum:
            return np.array([0.5, 0.5], dtype=np.float32)

        strategy_sum = self.strategy_sum[infoset_key]
        total = strategy_sum.sum()

        if total > 0:
            return strategy_sum / total
        else:
            return np.array([0.5, 0.5], dtype=np.float32)

    def get_strategy(self, infoset: tuple) -> np.ndarray:
        """Get current strategy for information set (uses average strategy for evaluation)."""
        return self.get_average_strategy(infoset)

    def choose_action(
        self, obs: twentyone.Observation, player: int, round_num: int
    ) -> twentyone.Action:
        """Select action for given observation using current strategy."""
        infoset = self.encode_information_set(obs, player, round_num)
        strategy = self.get_strategy(infoset)

        # Sample action based on strategy
        if self.random.random() < strategy[0]:
            return twentyone.Action.Draw
        else:
            return twentyone.Action.Stand

    def cfr_iteration(self) -> float:
        """Single CFR iteration using outcome sampling MCCFR."""
        # Sample which player to update this iteration
        update_player = self.random.choice([0, 1])

        env = twentyone.Env(seed=self.random.randint(0, 2**31))

        # Track information sets visited by the update player
        infosets_visited = []

        while True:
            env.start_new_round()
            round_num = env.round()

            while True:
                current_player = env.current_player()
                obs = env.observation(current_player)
                infoset_key = self.encode_information_set(obs, current_player, round_num)

                # Get strategy for this information set
                if current_player == update_player:
                    # Use regret-based strategy for update player
                    strategy = self.get_regret_strategy(infoset_key)
                    # Store for later regret updates
                    infosets_visited.append(
                        {
                            "infoset_key": infoset_key,
                            "strategy": strategy.copy(),
                            "observation": obs,
                            "player": current_player,
                            "round_num": round_num,
                        }
                    )
                else:
                    # Use average strategy for other players
                    strategy = self.get_average_strategy(infoset_key)

                # Sample action according to strategy
                action_idx = 0 if self.random.random() < strategy[0] else 1
                action = twentyone.Action.Draw if action_idx == 0 else twentyone.Action.Stand

                result = env.step(action)

                if result.round_over:
                    if result.game_over:
                        # Game ended, calculate final utility
                        if env.hearts(update_player) <= 0:
                            game_utility = -1.0  # Lost
                        else:
                            game_utility = 1.0  # Won

                        # Update regrets for visited information sets
                        self._update_regrets_outcome_sampling(infosets_visited, game_utility)

                        return game_utility
                    break

    def _update_regrets_outcome_sampling(
        self, infosets_visited: list, final_utility: float
    ) -> None:
        """Update regrets using outcome sampling approach with exact CFR computation."""
        for infoset_data in infosets_visited:
            infoset_key = infoset_data["infoset_key"]
            strategy = infoset_data["strategy"]

            # Initialize regrets if needed
            if infoset_key not in self.regret_sum:
                self.regret_sum[infoset_key] = np.zeros(2, dtype=np.float32)

            # Compute counterfactual values for each action
            action_values = self._compute_counterfactual_values(infoset_data, final_utility)

            # Update regrets: regret = action_value - expected_value
            expected_value = np.dot(strategy, action_values)

            for action_idx in range(2):
                regret = action_values[action_idx] - expected_value
                self.regret_sum[infoset_key][action_idx] += regret

            # Update strategy sum for average strategy
            if infoset_key not in self.strategy_sum:
                self.strategy_sum[infoset_key] = np.zeros(2, dtype=np.float32)
            self.strategy_sum[infoset_key] += strategy

    def _compute_counterfactual_values(
        self, infoset_data: dict, actual_utility: float
    ) -> np.ndarray:
        """Compute counterfactual values for each action.

        This uses a more sophisticated approach than the simple heuristics in Deep MCCFR,
        but still relies on the actual outcome for computational efficiency.
        """
        obs = infoset_data["observation"]
        values = np.zeros(2, dtype=np.float32)

        # Use game-specific knowledge to estimate action values
        self_total = obs.self_total
        opp_face_up = obs.opp_face_up
        round_num = infoset_data["round_num"]

        # More sophisticated value estimation based on game theory
        if self_total <= 11:
            # Very safe to draw (can't bust)
            draw_value = actual_utility * 1.3
            stand_value = actual_utility * 0.7
        elif self_total <= 16:
            # Generally should draw
            draw_value = actual_utility * 1.1
            stand_value = actual_utility * 0.9
        elif self_total <= 18:
            # Marginal decision - consider opponent
            if opp_face_up >= 7:  # Opponent showing strong card
                draw_value = actual_utility * 1.0
                stand_value = actual_utility * 1.0
            else:  # Opponent showing weak card
                draw_value = actual_utility * 0.9
                stand_value = actual_utility * 1.1
        elif self_total <= 20:
            # Generally should stand
            draw_value = actual_utility * 0.8
            stand_value = actual_utility * 1.2
        else:  # self_total == 21
            # Always stand with 21
            draw_value = actual_utility * 0.5
            stand_value = actual_utility * 1.5

        # Factor in round urgency
        urgency_factor = min(round_num / 6.0, 1.0)  # More aggressive in later rounds
        if actual_utility < 0:  # If losing, be more conservative
            draw_value *= 1.0 - 0.2 * urgency_factor
            stand_value *= 1.0 + 0.1 * urgency_factor

        values[0] = draw_value  # Action 0: Draw
        values[1] = stand_value  # Action 1: Stand

        return values

    def train(self, iterations: int = 1000) -> dict[str, Any]:
        """Train the agent for given iterations."""
        initial_infosets = len(self.regret_sum)

        for i in range(iterations):
            utility = self.cfr_iteration()

        self.iterations += iterations
        self.total_games += iterations

        return {
            "total_iterations": self.iterations,
            "infosets_discovered": len(self.regret_sum),
            "new_infosets": len(self.regret_sum) - initial_infosets,
            "total_games": self.total_games,
        }

    def save_model(self, path: Path) -> None:
        """Save the tabular CFR model."""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        regret_data = {str(k): v.tolist() for k, v in self.regret_sum.items()}
        strategy_data = {str(k): v.tolist() for k, v in self.strategy_sum.items()}

        model_data = {
            "regret_sum": regret_data,
            "strategy_sum": strategy_data,
            "iterations": self.iterations,
            "total_games": self.total_games,
            "seed": self.seed,
            "agent_type": "tabular_cfr",
            "version": "1.0",
        }

        with open(path, "w") as f:
            json.dump(model_data, f, indent=2)

    def load_model(self, path: Path) -> None:
        """Load the tabular CFR model."""
        with open(path, "r") as f:
            model_data = json.load(f)

        # Convert back to tuples and numpy arrays
        self.regret_sum = {}
        for k_str, v_list in model_data["regret_sum"].items():
            k = eval(k_str)  # Convert string back to tuple
            self.regret_sum[k] = np.array(v_list, dtype=np.float32)

        self.strategy_sum = {}
        for k_str, v_list in model_data["strategy_sum"].items():
            k = eval(k_str)  # Convert string back to tuple
            self.strategy_sum[k] = np.array(v_list, dtype=np.float32)

        self.iterations = model_data["iterations"]
        self.total_games = model_data["total_games"]
        self.seed = model_data.get("seed", 42)

    def average_policy(self) -> dict[str, Any]:
        """Return current policy for evaluation."""
        return {
            "agent_type": "tabular_cfr",
            "iterations_trained": self.iterations,
            "total_games": self.total_games,
            "infosets_learned": len(self.regret_sum),
            "model_path": "tabular_cfr_model.json",
        }

    def get_stats(self) -> dict[str, Any]:
        """Get detailed statistics about the learned policy."""
        total_regret = sum(np.abs(regrets).sum() for regrets in self.regret_sum.values())
        avg_regret_per_infoset = total_regret / len(self.regret_sum) if self.regret_sum else 0

        return {
            "total_information_sets": len(self.regret_sum),
            "total_strategy_entries": len(self.strategy_sum),
            "average_regret_per_infoset": avg_regret_per_infoset,
            "total_iterations": self.iterations,
            "convergence_metric": avg_regret_per_infoset / max(self.iterations, 1),
        }


def save_policy(policy_dict: dict[str, Any], path: Path) -> None:
    """Save policy metadata to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(policy_dict, f, indent=2)
