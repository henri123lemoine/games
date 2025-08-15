"""MCCFR agent implementation."""

import json
import random
from pathlib import Path
from typing import Dict, Any

import twentyone


class MCCFR:
    """Monte Carlo Counterfactual Regret Minimization agent."""
    
    def __init__(self, seed: int = 42):
        """Initialize MCCFR agent."""
        self.seed = seed
        self.random = random.Random(seed)
        self.regret_sum: Dict[str, Dict[str, float]] = {}
        self.strategy_sum: Dict[str, Dict[str, float]] = {}
        self.iterations = 0
    
    def train(self, iterations: int = 1000) -> None:
        """Train the agent for given iterations."""
        for _ in range(iterations):
            # Create environment for training
            env = twentyone.Env(seed=self.random.randint(0, 2**31))
            env.start_new_round()
            
            # Simulate game
            while True:
                player = env.current_player()
                obs = env.observation(player)
                
                # Simple strategy for now - improve this later
                action = twentyone.Action.Draw if obs.self_total < 17 else twentyone.Action.Stand
                
                result = env.step(action)
                if result.round_over or result.game_over:
                    break
        
        self.iterations += iterations
    
    def average_policy(self) -> Dict[str, Dict[str, float]]:
        """Return the average strategy."""
        # For now, return a simple policy
        return {
            "simple_strategy": {
                "draw_threshold": 17.0,
                "iterations_trained": self.iterations
            }
        }


def save_policy(policy: Dict[str, Any], path: Path) -> None:
    """Save policy to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(policy, f, indent=2)