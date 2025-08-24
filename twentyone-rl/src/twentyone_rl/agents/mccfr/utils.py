"""Shared utilities for MCCFR agents."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import twentyone


def save_policy(policy: dict[str, Any], path: Path) -> None:
    """Save policy to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(policy, f, indent=2)


def compute_action_values_heuristic(
    obs: twentyone.Observation, actual_utility: float, round_num: int = 1
) -> np.ndarray:
    """Compute estimated values for each action using game-specific heuristics.

    This is the shared heuristic used across MCCFR implementations.
    """
    values = np.zeros(2, dtype=np.float32)

    # Action 0: Draw, Action 1: Stand
    self_total = obs.self_total

    if self_total < 15:
        # Low total - drawing is usually better
        values[0] = actual_utility * 1.2 if actual_utility > 0 else actual_utility * 0.8
        values[1] = actual_utility * 0.8 if actual_utility > 0 else actual_utility * 1.2
    elif self_total > 19:
        # High total - standing is usually better
        values[0] = actual_utility * 0.7 if actual_utility > 0 else actual_utility * 1.3
        values[1] = actual_utility * 1.3 if actual_utility > 0 else actual_utility * 0.7
    else:
        # Medium total - both actions are reasonable
        values[0] = actual_utility * 0.9
        values[1] = actual_utility * 1.1

    return values
