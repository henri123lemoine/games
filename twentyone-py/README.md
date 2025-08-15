# Twenty-One

A fast implementation of the Twenty-One card game environment with Rust backend and Python interface.

## Installation

```bash
pip install twentyone
# or
uv add twentyone
```

## Quick Start

```python
import twentyone

# Create a new game environment
env = twentyone.Env(seed=42)

# Start a new round
env.start_new_round()

# Play the game
while True:
    player = env.current_player()
    obs = env.observation(player)

    # Simple strategy: draw if total < 17, otherwise stand
    action = twentyone.Action.Draw if obs.self_total < 17 else twentyone.Action.Stand

    result = env.step(action)

    if result.round_over:
        print(f"Round over! Winner: {result.outcome.winner if result.outcome else 'Tie'}")
        if result.game_over:
            break
        env.start_new_round()
```

## Features

- Fast Rust implementation for high-performance simulations
- Clean Python API with full type hints
- Support for deterministic testing with preset decks
- Comprehensive observation space for RL experiments
