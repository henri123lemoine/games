"""Deep MCCFR agent implementation using neural networks."""

import json
import random
from collections import deque
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import twentyone


class SharedEncoder(nn.Module):
    """Shared encoder for processing information sets."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PolicyNetwork(nn.Module):
    """Policy network for action probability estimation."""

    def __init__(self, encoder_output_dim: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(encoder_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Draw, Stand
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        logits = self.head(encoded)
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """Value network for counterfactual value estimation."""

    def __init__(self, encoder_output_dim: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(encoder_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.head(encoded)


class RegretNetwork(nn.Module):
    """Regret network for regret estimation."""

    def __init__(self, encoder_output_dim: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(encoder_output_dim, 32),
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

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch from buffer."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]

        infosets = torch.FloatTensor([exp[0] for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        regrets = torch.FloatTensor([exp[2] for exp in batch])
        policies = torch.FloatTensor([exp[3] for exp in batch])
        values = torch.FloatTensor([exp[4] for exp in batch])

        return infosets, actions, regrets, policies, values

    def __len__(self) -> int:
        return len(self.buffer)


class DeepMCCFR:
    """Deep Monte Carlo Counterfactual Regret Minimization agent."""

    def __init__(self, seed: int = 42, learning_rate: float = 1e-4, device: str = "cpu"):
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

        # Optimizers
        self.policy_optimizer = optim.Adam(
            list(self.policy_encoder.parameters()) + list(self.policy_net.parameters()),
            lr=learning_rate,
        )
        self.value_optimizer = optim.Adam(
            list(self.value_encoder.parameters()) + list(self.value_net.parameters()),
            lr=learning_rate,
        )
        self.regret_optimizer = optim.Adam(
            list(self.regret_encoder.parameters()) + list(self.regret_net.parameters()),
            lr=learning_rate,
        )

        # Experience buffer
        self.experience_buffer = ExperienceBuffer()

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
        """Get current strategy for information set."""
        with torch.no_grad():
            infoset_tensor = torch.FloatTensor(infoset).unsqueeze(0).to(self.device)
            encoded = self.regret_encoder(infoset_tensor)

            # Get regrets and convert to strategy
            regrets = self.regret_net(encoded).squeeze(0).cpu().numpy()
            regrets = np.maximum(regrets, 0)  # Only positive regrets

            if regrets.sum() > 0:
                strategy = regrets / regrets.sum()
            else:
                strategy = np.array([0.5, 0.5])  # Uniform if no regrets

        return strategy

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

    def train_networks(self, batch_size: int = 64) -> Dict[str, float]:
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
        """Single CFR iteration."""
        env = twentyone.Env(seed=self.random.randint(0, 2**31))
        env.start_new_round()

        # Track game history for counterfactual updates
        history = []

        while True:
            player = env.current_player()
            obs = env.observation(player)
            round_num = env.round()

            infoset = self.encode_observation(obs, player, round_num)
            strategy = self.get_strategy(infoset)

            # Choose action based on strategy
            action_idx = 0 if self.random.random() < strategy[0] else 1
            action = twentyone.Action.Draw if action_idx == 0 else twentyone.Action.Stand

            # Store state for later updates
            history.append((infoset, action_idx, strategy, player))

            result = env.step(action)

            if result.round_over or result.game_over:
                # Compute utilities and update regrets
                winner = None
                if result.round_over and result.outcome:
                    winner = result.outcome.winner
                self._update_regrets(history, winner, round_num)

                if result.game_over:
                    break

                if not result.game_over:
                    env.start_new_round()
                    history = []

    def _update_regrets(self, history: List, winner: Optional[int], round_num: int) -> None:
        """Update regrets based on round outcome."""
        if not history:
            return

        # Simple utility: winner gets +1, loser gets -1, tie gets 0
        for infoset, action_idx, strategy, player in history:
            if winner is None:
                utility = 0.0  # Tie
            elif winner == player:
                utility = float(round_num)  # Win
            else:
                utility = -float(round_num)  # Loss

            # Compute regret (simplified)
            regret = utility - (strategy @ np.array([utility, utility]))

            # Store experience
            self.experience_buffer.add(infoset, action_idx, regret, strategy, utility)

    def train(self, iterations: int = 1000) -> Dict[str, Any]:
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

    def average_policy(self) -> Dict[str, Any]:
        """Return current policy for evaluation."""
        return {
            "agent_type": "deep_mccfr",
            "iterations_trained": self.iterations,
            "total_games": self.total_games,
            "model_path": "deep_mccfr_model.pth",
        }


def save_policy(policy: Dict[str, Any], path: Path) -> None:
    """Save policy to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(policy, f, indent=2)
