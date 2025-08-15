# Twenty-One

A high-performance implementation of the Twenty-One card game designed for reinforcement learning research.

## Project Structure

**twentyone-core/**: Pure Rust game engine library
**twentyone-py/**: Python bindings using PyO3
**twentyone-rl/**: RL agents, training algorithms, and experiments

## Quick Start

```bash
# Build Python bindings
cd twentyone-py && maturin develop

# Install RL package
cd ../twentyone-rl

# Run examples
uv run examples/basic_play.py
uv run examples/train_agent.py
```

## Game Rules

Twenty-One is a 2-player card game with a hearts system:

- Each player starts with 6 hearts
- Each round uses cards 1-11 (one of each)
- Players get 2 cards: one face-up (visible), one face-down (hidden)
- Goal: Get closest to 21 without going over
- Round winner deals damage equal to round number
- Game ends when a player reaches 0 hearts

## Architecture

**Performance**: Direct memory access between Python and Rust with zero serialization overhead. Efficient bitmask-based deck representation.

**Modularity**: Clean separation between game logic, bindings, and RL code. Each component can be used independently.

**Developer Experience**: Full type safety with Python type hints and Rust's type system. Deterministic testing with preset deck orders.
