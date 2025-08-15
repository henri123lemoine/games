# CLAUDE.md

This project explores reinforcement learning in Twenty-One, a simple hidden-information card game. The implementation uses a modular architecture with a Rust environment core and Python bindings for ML experiments.

## Project Structure

- **twentyone-core/**: Pure Rust game engine library
- **twentyone-py/**: Python bindings using PyO3 for zero-overhead Rust integration
- **twentyone-rl/**: RL agents, training algorithms, and experiments

## Development Workflow

### Building and Testing

```bash
# Build Python bindings
cd twentyone-py && maturin develop

# Install RL package
cd ../twentyone-rl && uv pip install -e .

# Run examples
uv run examples/basic_play.py
uv run examples/train_agent.py
```

### Code Standards

- **Rust code**: Follow guidelines in `twentyone-core/CLAUDE.md`
- **Python code**: Follow guidelines in `twentyone-rl/CLAUDE.md`
- **Dependencies**: Use `uv` for Python package management
- **Type safety**: Maintain full type annotations in Python, leverage Rust's type system

## Game Rules

Twenty-One is a 2-player card game with hearts system:

- 2 players, each starting with 6 hearts
- Each round uses cards 1-11 (one of each)
- Goal: Get closest to 21 without going over
- Round winner deals damage equal to round number
- Game ends when a player reaches 0 hearts

**Each Round:**

1. Both players receive 2 cards: one face-up (visible to opponent), one face-down (hidden)
2. Players alternate turns choosing to draw (face-up) or stand
3. Round ends when both stand or no cards remain
4. Winner determined by closest to 21 without busting
