"""Base classes and shared logic for MCCFR agents."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import twentyone


class MCCFRAgent(ABC):
    """Abstract base class for MCCFR agents."""

    @abstractmethod
    def choose_action(
        self, obs: twentyone.Observation, player: int, round_num: int
    ) -> twentyone.Action:
        """Choose action for given observation."""
        pass

    @abstractmethod
    def train(self, iterations: int) -> dict[str, Any]:
        """Train the agent for given iterations."""
        pass

    @abstractmethod
    def save_model(self, path: Path) -> None:
        """Save model to file."""
        pass

    @abstractmethod
    def load_model(self, path: Path) -> None:
        """Load model from file."""
        pass

    @abstractmethod
    def average_policy(self) -> dict[str, Any]:
        """Return average policy information."""
        pass
