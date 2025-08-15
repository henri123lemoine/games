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

    def __init__(self, input_dim: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
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
        expected_shape = (x.shape[0], self.input_dim)
        if x.shape != expected_shape:
            raise ValueError(
                f"Input tensor shape {x.shape} doesn't match expected shape {expected_shape}. "
                f"Expected input_dim={self.input_dim}, got {x.shape[1] if len(x.shape) > 1 else 'unknown'}."
            )
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

    def __init__(
        self,
        seed: int = 42,
        learning_rate: float = 3e-4,
        device: str = "cpu",
        input_dim: int | None = None,
    ):
        """Initialize Deep MCCFR agent."""
        self.seed = seed
        self.device = torch.device(device)
        self.random = random.Random(seed)

        # Compute feature dimensions dynamically if not provided
        self.input_dim = input_dim if input_dim is not None else self._compute_feature_dim()

        # Neural networks - single shared encoder for consistency
        self.shared_encoder = SharedEncoder(input_dim=self.input_dim).to(self.device)
        self.policy_net = PolicyNetwork().to(self.device)
        self.regret_net = RegretNetwork().to(self.device)

        # Single optimizer for all networks
        self.optimizer = optim.Adam(
            list(self.shared_encoder.parameters())
            + list(self.policy_net.parameters())
            + list(self.regret_net.parameters()),
            lr=learning_rate,
            weight_decay=1e-5,
        )

        # MCCFR specific storage
        self.regret_sum: dict[tuple, np.ndarray] = {}  # Information set -> action regrets
        self.strategy_sum: dict[tuple, np.ndarray] = {}  # Information set -> strategy accumulation

        # Training statistics
        self.iterations = 0
        self.total_games = 0

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _compute_feature_dim(self) -> int:
        """Compute the number of features produced by encode_observation."""
        # Create a dummy observation to get feature dimensions
        dummy_env = twentyone.Env(seed=0)
        dummy_env.start_new_round()
        dummy_obs = dummy_env.observation(0)
        dummy_features = self.encode_observation(dummy_obs, 0, 1)
        return len(dummy_features)

    def _validate_input_shape(self, tensor: torch.Tensor, expected_features: str = "input") -> None:
        """Validate input tensor has the correct shape."""
        if len(tensor.shape) != 2:
            raise ValueError(
                f"Expected 2D tensor for {expected_features}, got shape {tensor.shape}"
            )
        if tensor.shape[1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch for {expected_features}: expected {self.input_dim}, "
                f"got {tensor.shape[1]}. This usually means the observation encoding has changed "
                f"since the model was trained. Try retraining the model or check the encode_observation method."
            )

    def encode_observation(
        self, obs: twentyone.Observation, player: int, round_num: int
    ) -> np.ndarray:
        """Encode observation into simplified essential feature vector."""
        # Simplified feature vector with only essential information
        features = np.zeros(8, dtype=np.float32)

        # Core game state features
        features[0] = obs.self_total / 25.0  # Own total (normalized)
        features[1] = obs.self_hearts / 6.0  # Own hearts (normalized)
        features[2] = float(obs.self_stood)  # Whether we've stood (binary)
        features[3] = obs.opp_face_up / 11.0  # Opponent visible card (normalized)
        features[4] = obs.opp_hearts / 6.0  # Opponent hearts (normalized)
        features[5] = float(obs.opp_stood)  # Whether opponent has stood (binary)
        features[6] = round_num / 6.0  # Round number (normalized)
        features[7] = obs.deck_count / 11.0  # Remaining cards (normalized)

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

    def train_networks(
        self, infosets: list, strategies: list, regrets_data: list
    ) -> dict[str, float]:
        """Train neural networks on collected CFR data."""
        if len(infosets) == 0:
            return {}

        try:
            # Convert to tensors with shape validation
            infosets_array = np.array(infosets, dtype=np.float32)
            strategies_array = np.array(strategies, dtype=np.float32)
            regrets_array = np.array(regrets_data, dtype=np.float32)

            # Validate shapes before tensor conversion
            if len(infosets_array.shape) != 2:
                raise ValueError(f"infosets must be 2D, got shape {infosets_array.shape}")
            if infosets_array.shape[1] != self.input_dim:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.input_dim}, got {infosets_array.shape[1]}"
                )
            if len(strategies_array.shape) != 2 or strategies_array.shape[1] != 2:
                raise ValueError(f"strategies must be shape (N, 2), got {strategies_array.shape}")
            if len(regrets_array.shape) != 2 or regrets_array.shape[1] != 2:
                raise ValueError(f"regrets must be shape (N, 2), got {regrets_array.shape}")

            infosets_tensor = torch.from_numpy(infosets_array).to(self.device)
            strategies_tensor = torch.from_numpy(strategies_array).to(self.device)
            regrets_tensor = torch.from_numpy(regrets_array).to(self.device)

            # Get shared encoding
            encoded = self.shared_encoder(infosets_tensor)
        except Exception as e:
            raise RuntimeError(
                f"Error in train_networks tensor conversion: {e}. "
                f"This often indicates shape mismatches between training data and model architecture. "
                f"Expected input_dim: {self.input_dim}, got infosets shape: {getattr(infosets_array, 'shape', 'unknown')}"
            ) from e

        # Train policy network to predict strategies
        pred_policies = self.policy_net(encoded)
        policy_loss = F.mse_loss(pred_policies, strategies_tensor)

        # Train regret network to predict regrets
        pred_regrets = self.regret_net(encoded)
        regret_loss = F.mse_loss(pred_regrets, regrets_tensor)

        # Combined loss
        total_loss = policy_loss + regret_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "regret_loss": regret_loss.item(),
            "total_loss": total_loss.item(),
        }

    def cfr_iteration(self) -> float:
        """Single CFR iteration using proper outcome sampling MCCFR."""
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
                infoset = self.encode_observation(obs, current_player, round_num)
                infoset_key = tuple(infoset)

                # Get strategy for this information set
                if current_player == update_player:
                    # Use regret-based strategy for update player
                    strategy = self.get_regret_strategy(infoset_key)
                    # Store for later regret updates
                    infosets_visited.append(
                        {
                            "infoset_key": infoset_key,
                            "infoset": infoset,
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

    def _update_regrets_outcome_sampling(
        self, infosets_visited: list, final_utility: float
    ) -> None:
        """Update regrets using outcome sampling approach."""
        for infoset_data in infosets_visited:
            infoset_key = infoset_data["infoset_key"]
            strategy = infoset_data["strategy"]

            # Initialize regrets if needed
            if infoset_key not in self.regret_sum:
                self.regret_sum[infoset_key] = np.zeros(2, dtype=np.float32)

            # Compute action values by simulation or heuristics
            action_values = self._compute_action_values(infoset_data, final_utility)

            # Update regrets: regret = action_value - expected_value
            expected_value = np.dot(strategy, action_values)

            for action_idx in range(2):
                regret = action_values[action_idx] - expected_value
                self.regret_sum[infoset_key][action_idx] += regret

            # Update strategy sum for average strategy (only for visited infosets)
            if infoset_key not in self.strategy_sum:
                self.strategy_sum[infoset_key] = np.zeros(2, dtype=np.float32)
            self.strategy_sum[infoset_key] += strategy

    def _compute_action_values(self, infoset_data: dict, actual_utility: float) -> np.ndarray:
        """Compute estimated values for each action at an information set."""
        # Use the observation from the information set data
        obs = infoset_data["observation"]
        values = np.zeros(2, dtype=np.float32)

        # Action 0: Draw, Action 1: Stand

        # Simple heuristic based on current total (similar to working MCCFR)
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
        training_stats: dict[str, list[dict[str, float]]] = {"losses": []}

        # Collect training data during CFR iterations
        training_infosets: list[np.ndarray] = []
        training_strategies: list[np.ndarray] = []
        training_regrets: list[np.ndarray] = []

        for i in range(iterations):
            # CFR iteration
            utility = self.cfr_iteration()

            # Collect current strategies and regrets for neural network training
            if i % 100 == 0:  # Collect training data every 100 iterations
                for infoset_key, regrets in self.regret_sum.items():
                    if infoset_key in self.strategy_sum:
                        # Get current strategy and regrets
                        strategy = self.get_average_strategy(infoset_key)
                        infoset_array = np.array(infoset_key, dtype=np.float32)

                        training_infosets.append(infoset_array)
                        training_strategies.append(strategy)
                        training_regrets.append(regrets)

                # Train networks if we have enough data
                if len(training_infosets) > 32:
                    losses = self.train_networks(
                        training_infosets, training_strategies, training_regrets
                    )
                    if losses:
                        training_stats["losses"].append(losses)

                    # Clear training data
                    training_infosets = []
                    training_strategies = []
                    training_regrets = []

        self.iterations += iterations
        self.total_games += iterations

        training_stats["total_iterations"] = self.iterations
        training_stats["infosets_learned"] = len(self.regret_sum)

        return training_stats

    def save_model(self, path: Path) -> None:
        """Save model weights and configuration."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "shared_encoder": self.shared_encoder.state_dict(),
                "policy_net": self.policy_net.state_dict(),
                "regret_net": self.regret_net.state_dict(),
                "iterations": self.iterations,
                "total_games": self.total_games,
                "regret_sum": dict(self.regret_sum),
                "strategy_sum": dict(self.strategy_sum),
                # Save model configuration for robustness
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.shared_encoder.hidden_dim,
                    "seed": self.seed,
                },
                "version": "2.0",  # Version for backward compatibility
            },
            path,
        )

    def load_model(self, path: Path) -> None:
        """Load model weights with backward compatibility."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            # Check for version compatibility
            version = checkpoint.get("version", "1.0")
            config = checkpoint.get("config", {})

            # Validate input dimensions if config is available
            if "input_dim" in config:
                saved_input_dim = config["input_dim"]
                if saved_input_dim != self.input_dim:
                    current_dim = self._compute_feature_dim()
                    if saved_input_dim != current_dim:
                        raise ValueError(
                            f"Model was trained with input_dim={saved_input_dim}, "
                            f"but current observation encoding produces {current_dim} features. "
                            f"This indicates the observation encoding has changed. "
                            f"Either retrain the model or adjust the encode_observation method."
                        )
                    # Update our input_dim to match the saved model
                    self.input_dim = saved_input_dim
                    # Recreate networks with correct dimensions
                    self._recreate_networks()

            self.shared_encoder.load_state_dict(checkpoint["shared_encoder"])
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.regret_net.load_state_dict(checkpoint["regret_net"])
            self.iterations = checkpoint.get("iterations", 0)
            self.total_games = checkpoint.get("total_games", 0)

            # Load CFR data if available
            if "regret_sum" in checkpoint:
                self.regret_sum = checkpoint["regret_sum"]
            if "strategy_sum" in checkpoint:
                self.strategy_sum = checkpoint["strategy_sum"]

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {path}: {e}. "
                f"This often indicates incompatible model versions or corrupted files. "
                f"Try retraining the model if the error persists."
            ) from e

    def _recreate_networks(self) -> None:
        """Recreate networks with updated input dimensions."""
        hidden_dim = self.shared_encoder.hidden_dim

        # Recreate networks
        self.shared_encoder = SharedEncoder(input_dim=self.input_dim, hidden_dim=hidden_dim).to(
            self.device
        )
        self.policy_net = PolicyNetwork().to(self.device)
        self.regret_net = RegretNetwork().to(self.device)

        # Recreate optimizer
        self.optimizer = optim.Adam(
            list(self.shared_encoder.parameters())
            + list(self.policy_net.parameters())
            + list(self.regret_net.parameters()),
            lr=3e-4,  # Use default learning rate
            weight_decay=1e-5,
        )

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
