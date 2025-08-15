"""Deep MCCFR agent implementation using neural networks."""

import json
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import twentyone


class SharedEncoder(nn.Module):
    """Shared encoder for processing information sets."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PolicyNetwork(nn.Module):
    """Policy network for action probability estimation."""

    def __init__(self, encoder_output_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(encoder_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Draw, Stand
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        logits = self.head(encoded)
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """Value network for counterfactual value estimation."""

    def __init__(self, encoder_output_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(encoder_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.head(encoded)


class RegretNetwork(nn.Module):
    """Regret network for regret estimation."""

    def __init__(self, encoder_output_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(encoder_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Regret for Draw, Stand
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.head(encoded)


class ExperienceBuffer:
    """Experience replay buffer for stable learning."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def add(
        self, infoset: np.ndarray, action: int, regret: float, policy: np.ndarray, value: float
    ) -> None:
        """Add experience to buffer."""
        self.buffer.append((infoset, action, regret, policy, value))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Sample batch from buffer."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]

        # Pre-allocate arrays to avoid slow tensor creation warnings
        infosets_np = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions_np = np.array([exp[1] for exp in batch], dtype=np.int64)
        regrets_np = np.array([exp[2] for exp in batch], dtype=np.float32)
        policies_np = np.array([exp[3] for exp in batch], dtype=np.float32)
        values_np = np.array([exp[4] for exp in batch], dtype=np.float32)

        infosets = torch.from_numpy(infosets_np)
        actions = torch.from_numpy(actions_np)
        regrets = torch.from_numpy(regrets_np)
        policies = torch.from_numpy(policies_np)
        values = torch.from_numpy(values_np)

        return infosets, actions, regrets, policies, values

    def __len__(self) -> int:
        return len(self.buffer)


class DeepMCCFR:
    """Deep Monte Carlo Counterfactual Regret Minimization agent."""

    def __init__(self, seed: int = 42, learning_rate: float = 3e-4, device: str = "cpu"):
        """Initialize Deep MCCFR agent."""
        self.seed = seed
        self.device = torch.device(device)
        self.random = random.Random(seed)

        # Neural networks - separate encoders to avoid gradient conflicts
        self.policy_encoder = SharedEncoder().to(self.device)
        self.value_encoder = SharedEncoder().to(self.device)
        self.regret_encoder = SharedEncoder().to(self.device)
        self.policy_net = PolicyNetwork().to(self.device)
        self.value_net = ValueNetwork().to(self.device)
        self.regret_net = RegretNetwork().to(self.device)

        # Optimizers with better parameters for MCCFR
        self.policy_optimizer = optim.Adam(
            list(self.policy_encoder.parameters()) + list(self.policy_net.parameters()),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        self.value_optimizer = optim.Adam(
            list(self.value_encoder.parameters()) + list(self.value_net.parameters()),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        self.regret_optimizer = optim.Adam(
            list(self.regret_encoder.parameters()) + list(self.regret_net.parameters()),
            lr=learning_rate,
            weight_decay=1e-5,
        )

        # Experience buffer
        self.experience_buffer = ExperienceBuffer()

        # MCCFR specific storage
        self.regret_sum = {}  # Information set -> action regrets
        self.strategy_sum = {}  # Information set -> strategy accumulation

        # Training statistics
        self.iterations = 0
        self.total_games = 0

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

    def encode_observation(
        self, obs: twentyone.Observation, player: int, round_num: int
    ) -> np.ndarray:
        """Encode observation into simplified feature vector based on available attributes."""
        # Use a smaller feature vector based on what's actually available
        features = np.zeros(10, dtype=np.float32)
        idx = 0

        # Own State
        features[idx] = obs.self_face_up / 11.0  # Normalize to [0,1]
        idx += 1
        features[idx] = obs.self_face_down / 11.0
        idx += 1
        features[idx] = obs.self_total / 25.0  # Normalize (max possible ~22-25)
        idx += 1
        features[idx] = obs.self_hearts / 6.0  # Normalize hearts
        idx += 1
        features[idx] = float(obs.self_stood)  # Binary
        idx += 1

        # Opponent State
        features[idx] = obs.opp_face_up / 11.0
        idx += 1
        features[idx] = obs.opp_hearts / 6.0  # Opponent hearts
        idx += 1
        features[idx] = float(obs.opp_stood)  # Binary
        idx += 1

        # Game State
        features[idx] = round_num / 6.0  # Normalize round number
        idx += 1
        features[idx] = obs.deck_count / 11.0  # Remaining cards
        idx += 1

        return features

    def get_strategy(self, infoset: np.ndarray) -> np.ndarray:
        """Get current strategy for information set using average strategy."""
        infoset_key = tuple(infoset)
        return self.get_average_strategy(infoset_key)

    def choose_action(
        self, obs: twentyone.Observation, player: int, round_num: int
    ) -> twentyone.Action:
        """Choose action based on current strategy."""
        infoset = self.encode_observation(obs, player, round_num)
        strategy = self.get_strategy(infoset)

        # Sample action based on strategy
        if self.random.random() < strategy[0]:
            return twentyone.Action.Draw
        else:
            return twentyone.Action.Stand

    def train_networks(self, batch_size: int = 64) -> dict[str, float]:
        """Train neural networks on experience buffer."""
        if len(self.experience_buffer) < batch_size:
            return {}

        infosets, actions, regrets, policies, values = self.experience_buffer.sample(batch_size)
        infosets = infosets.to(self.device)
        actions = actions.to(self.device)
        regrets = regrets.to(self.device)
        policies = policies.to(self.device)
        values = values.to(self.device)

        losses = {}

        # Train policy network
        policy_encoded = self.policy_encoder(infosets)
        pred_policies = self.policy_net(policy_encoded)
        policy_loss = F.mse_loss(pred_policies, policies)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        losses["policy_loss"] = policy_loss.item()

        # Train value network
        value_encoded = self.value_encoder(infosets)
        pred_values = self.value_net(value_encoded).squeeze()
        value_loss = F.mse_loss(pred_values, values)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        losses["value_loss"] = value_loss.item()

        # Train regret network
        regret_encoded = self.regret_encoder(infosets)
        pred_regrets = self.regret_net(regret_encoded)
        regret_targets = torch.zeros_like(pred_regrets)
        regret_targets[range(len(actions)), actions] = regrets
        regret_loss = F.mse_loss(pred_regrets, regret_targets)

        self.regret_optimizer.zero_grad()
        regret_loss.backward()
        self.regret_optimizer.step()
        losses["regret_loss"] = regret_loss.item()

        return losses

    def cfr_iteration(self) -> None:
        """Single CFR iteration with proper MCCFR implementation."""
        # Sample a chance outcome for this iteration (player to update)
        update_player = self.random.choice([0, 1])

        env = twentyone.Env(seed=self.random.randint(0, 2**31))

        # Run full game and collect utilities
        game_history = []

        while True:
            env.start_new_round()
            round_history = []
            round_num = env.round()

            while True:
                player = env.current_player()
                obs = env.observation(player)

                infoset = self.encode_observation(obs, player, round_num)
                infoset_key = tuple(infoset)  # Use as dict key

                # Get current strategy for this information set
                if player == update_player:
                    # Use current regret-based strategy for update player
                    strategy = self.get_regret_strategy(infoset_key)
                else:
                    # Use average strategy for other players
                    strategy = self.get_average_strategy(infoset_key)

                # Sample action
                action_idx = 0 if self.random.random() < strategy[0] else 1
                action = twentyone.Action.Draw if action_idx == 0 else twentyone.Action.Stand

                # Store information for regret updates
                round_history.append(
                    {
                        "player": player,
                        "infoset": infoset,
                        "infoset_key": infoset_key,
                        "strategy": strategy.copy(),
                        "action": action_idx,
                    }
                )

                result = env.step(action)

                if result.round_over:
                    # Calculate utilities for this round
                    round_utility = self._calculate_round_utility(result, round_num)

                    # Update regrets for the update player
                    if update_player in [step["player"] for step in round_history]:
                        self._update_cfr_regrets(round_history, round_utility, update_player)

                    # Update strategy sums for all players
                    self._update_strategy_sums(round_history)

                    game_history.extend(round_history)

                    if result.game_over:
                        return
                    break

    def get_regret_strategy(self, infoset_key: tuple) -> np.ndarray:
        """Get strategy based on current regrets."""
        if infoset_key not in self.regret_sum:
            self.regret_sum[infoset_key] = np.zeros(2, dtype=np.float32)

        regrets = np.maximum(self.regret_sum[infoset_key], 0)
        regret_sum = regrets.sum()

        if regret_sum > 0:
            strategy = regrets / regret_sum
        else:
            strategy = np.array([0.5, 0.5], dtype=np.float32)

        return strategy

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

    def _calculate_round_utility(self, result, round_num: int) -> dict[int, float]:
        """Calculate utility for each player from round result."""
        utilities = {0: 0.0, 1: 0.0}

        if result.outcome and result.outcome.winner is not None:
            winner = result.outcome.winner
            loser = 1 - winner
            # Winner gets positive utility, loser gets negative
            utilities[winner] = float(round_num)
            utilities[loser] = -float(round_num)

        return utilities

    def _update_cfr_regrets(
        self, history: list, utilities: dict[int, float], update_player: int
    ) -> None:
        """Update regrets using CFR algorithm."""
        for step in history:
            if step["player"] != update_player:
                continue

            infoset_key = step["infoset_key"]
            action_taken = step["action"]
            strategy = step["strategy"]

            # Initialize if needed
            if infoset_key not in self.regret_sum:
                self.regret_sum[infoset_key] = np.zeros(2, dtype=np.float32)

            # Get utility for this player
            player_utility = utilities[update_player]

            # Calculate regret for each action
            for action in range(2):
                if action == action_taken:
                    # This is the action that was actually taken
                    action_regret = player_utility - (
                        strategy @ np.array([player_utility, player_utility])
                    )
                else:
                    # Counterfactual: what if we had taken this action instead
                    action_regret = player_utility * 0.5  # Simplified counterfactual value

                self.regret_sum[infoset_key][action] += action_regret

            # Store experience for neural network training
            self.experience_buffer.add(
                step["infoset"],
                action_taken,
                self.regret_sum[infoset_key][action_taken],
                strategy,
                player_utility,
            )

    def _update_strategy_sums(self, history: list) -> None:
        """Update strategy sums for average strategy calculation."""
        for step in history:
            infoset_key = step["infoset_key"]
            strategy = step["strategy"]

            if infoset_key not in self.strategy_sum:
                self.strategy_sum[infoset_key] = np.zeros(2, dtype=np.float32)

            self.strategy_sum[infoset_key] += strategy

    def train(self, iterations: int = 1000) -> dict[str, Any]:
        """Train the agent for given iterations."""
        training_stats = {"losses": []}

        for i in range(iterations):
            # CFR iteration
            self.cfr_iteration()

            # Train networks periodically
            if i % 10 == 0 and len(self.experience_buffer) > 64:
                losses = self.train_networks()
                if losses:
                    training_stats["losses"].append(losses)

        self.iterations += iterations
        self.total_games += iterations

        training_stats["total_iterations"] = self.iterations
        training_stats["buffer_size"] = len(self.experience_buffer)

        return training_stats

    def save_model(self, path: Path) -> None:
        """Save model weights."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_encoder": self.policy_encoder.state_dict(),
                "value_encoder": self.value_encoder.state_dict(),
                "regret_encoder": self.regret_encoder.state_dict(),
                "policy_net": self.policy_net.state_dict(),
                "value_net": self.value_net.state_dict(),
                "regret_net": self.regret_net.state_dict(),
                "iterations": self.iterations,
                "total_games": self.total_games,
            },
            path,
        )

    def load_model(self, path: Path) -> None:
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_encoder.load_state_dict(checkpoint["policy_encoder"])
        self.value_encoder.load_state_dict(checkpoint["value_encoder"])
        self.regret_encoder.load_state_dict(checkpoint["regret_encoder"])
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.regret_net.load_state_dict(checkpoint["regret_net"])
        self.iterations = checkpoint.get("iterations", 0)
        self.total_games = checkpoint.get("total_games", 0)

    def average_policy(self) -> dict[str, Any]:
        """Return current policy for evaluation."""
        return {
            "agent_type": "deep_mccfr",
            "iterations_trained": self.iterations,
            "total_games": self.total_games,
            "model_path": "deep_mccfr_model.pth",
        }


def save_policy(policy: dict[str, Any], path: Path) -> None:
    """Save policy to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(policy, f, indent=2)
