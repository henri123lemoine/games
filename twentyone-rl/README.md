# Twenty-One RL

Reinforcement learning experiments for the Twenty-One card game.

This package provides agents, training algorithms, and utilities for conducting RL experiments with the fast `twentyone` environment.

## Installation

First install the twentyone environment:

```bash
# From the twentyone-py directory
cd ../twentyone-py
maturin develop
```

Then install this RL package:

```bash
uv pip install -e .
```

## Usage

### Basic Gameplay

```python
import twentyone

env = twentyone.Env(seed=42)
env.start_new_round()

while True:
    player = env.current_player()
    obs = env.observation(player)

    # Simple strategy
    action = twentyone.Action.Draw if obs.self_total < 17 else twentyone.Action.Stand

    result = env.step(action)
    if result.round_over:
        break
```

### Training an MCCFR Agent

```python
from twentyone_rl.agents import MCCFR, save_policy

agent = MCCFR(seed=42)
agent.train(iterations=1000000)
policy = agent.average_policy()
save_policy(policy, "policy.json")
```

### Examples

- `examples/basic_play.py` - Basic gameplay demonstration
- `examples/train_agent.py` - Train an MCCFR agent

## Features

- **Fast Environment**: Uses the Rust-backed `twentyone` package for high-performance simulations
- **MCCFR Agent**: Monte Carlo Counterfactual Regret Minimization implementation
- **Clean API**: Direct Python interface, no bridge processes needed
- **Type Safety**: Full type hints throughout

## Performance

The new architecture provides significant performance improvements:

- **No JSON serialization overhead**: Direct memory access between Python and Rust
- **No subprocess management**: Everything runs in-process
- **Parallel training**: Can easily leverage multiple cores

## Migration from Bridge-based Code

The package includes a compatibility `Bridge` class that wraps the new `twentyone` package with the old JSON bridge interface, making migration easier for existing code.
