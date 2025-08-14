# CLAUDE.md

This project's purpose is exploring RL in a simple hidden-information game. The environment itself is written in Rust, and the agent is written in Python.

## Rules

Twenty-One is a 2-player card game with hearts system. Each round uses cards 1-11, players get 2 cards (1 face-up, 1 face-down) and take turns drawing face-up cards or standing. Goal is closest to 21 without going over. Round winner deals damage equal to round number. Game ends when a player reaches 0 hearts.

Setup:

- 2 players, each starting with **6 hearts**
- Each round uses cards numbered 1-11 (one of each)

Gameplay:

**Each Round:**

1. Both players receive 2 cards: one face-up (visible to opponent), one face-down (hidden)
2. Players take turns choosing to:
   - **Draw** a card from the remaining deck. This card is face-up.
   - **Stand** (keep current total)
3. Round ends when both players stand or no cards remain

Winning Rounds:

- **Goal:** Get as close to 21 as possible without going over
- **Winner:** Closest to 21 without exceeding it
- **Over 21:** Loss, unless both players go over, then closest to 21 wins
- **Tie:** No hearts lost

Losing Hearts:

- Round winner deals damage equal to the round number
- Round 1 = 1 heart lost, Round 2 = 2 hearts lost, etc.
- **Game ends** when a player reaches 0 hearts

## Architecture Overview

This is a Twenty-One card game implementation with two main components:

### Rust Core (`src/`)

- **Environment**: Fast, allocation-conscious game engine in `src/env.rs`
- **Bridge Binary**: JSON stdin/stdout bridge in `src/bin/bridge.rs` for Python interop
- Uses bitmasks for efficient deck state management
- Deterministic testing support with preset deck orders
- XorShift64 PRNG for reproducible randomness

See `src/CLAUDE.md` for more details.

### Python Scripts (`scripts/`)

- **Bridge Interface**: `common.py` provides `Bridge` class for Rust binary communication
- **Bot Training**: MCCFR agent training and simple bot implementations
- **Interactive Play**: Human vs agent gameplay scripts
- Auto-builds Rust bridge binary when needed via `find_or_build_bridge()`

The bridge protocol uses JSON commands like `{"cmd":"new","seed":42}`, `{"cmd":"step","action":"draw"}` for game interactions.

See `scripts/CLAUDE.md` for more details.

## RL

### Action Space

- Draw a card
- Stand

### Observation Space

*This section is subject to change and *not* final.*

**Own State (15 dimensions):**

- Own visible card value (1-11) - 1 dimension
- Own hidden card value (1-11) - 1 dimension
- Own additional drawn cards (up to 9 cards, padded with 0s) - 9 dimensions
- Own current total - 1 dimension
- Own hearts remaining - 1 dimension
- Own has stood this round (binary) - 1 dimension
- Round number - 1 dimension

**Opponent State (13 dimensions):**

- Opponent visible card value (1-11) - 1 dimension
- Opponent additional drawn cards (up to 9 cards, sequence matters) - 9 dimensions
- Opponent minimum possible total (visible + drawn) - 1 dimension
- Opponent hearts remaining - 1 dimension
- Opponent has stood this round (binary) - 1 dimension

**Deck State (11 dimensions):**

- Cards 1-11: Binary availability flags - 11 dimensions

**Total: 39 dimensions**

### Reward Function

The only goal is to win the game.
