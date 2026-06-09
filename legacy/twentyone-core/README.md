# Twenty-One Core

Pure Rust implementation of the Twenty-One card game environment.

This library provides a fast, allocation-conscious game engine for the Twenty-One card game, suitable for use in other Rust projects or as a backend for language bindings.

## Features

- **Fast**: Efficient implementation using bitmasks for deck state
- **Memory-conscious**: Minimal allocations during gameplay
- **Deterministic**: Support for preset deck orders for testing
- **Well-tested**: Comprehensive test suite
- **Documentation**: Full API documentation with examples

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
twentyone-core = "0.1"
```

Basic usage:

```rust
use twentyone_core::{Env, Action};

let mut env = Env::new(42);  // seed for reproducibility
env.start_new_round()?;

loop {
    let player = env.current_player();
    let obs = env.observation(player);

    // Simple strategy: draw if total < 17, otherwise stand
    let action = if obs.self_total < 17 {
        Action::Draw
    } else {
        Action::Stand
    };

    let result = env.step(action)?;

    if result.round_over {
        if let Some(outcome) = result.outcome {
            println!("Round over! Winner: {:?}", outcome.winner);
        }
        if result.game_over {
            break;
        }
        env.start_new_round()?;
    }
}
```

## Game Rules

Twenty-One is a 2-player card game:

- Each player starts with 6 hearts
- Each round uses cards 1-11 (one of each)
- Players get 2 cards: one face-up (visible), one face-down (hidden)
- Players take turns drawing face-up cards or standing
- Goal: get closest to 21 without going over
- Round winner deals damage equal to round number
- Game ends when a player reaches 0 hearts

## API

### Core Types

- `Env` - Main game environment
- `Action` - Player action (Draw or Stand)
- `Observation` - Player's view of the game state
- `StepResult` - Result of taking an action
- `RoundOutcome` - Result of a completed round

### Key Methods

- `Env::new(seed)` - Create environment with random seed
- `Env::start_new_round()` - Deal cards and start a new round
- `Env::step(action)` - Take an action for the current player
- `Env::observation(player)` - Get observation for a player
- `Env::current_player()` - Get the current player index
