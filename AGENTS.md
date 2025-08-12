# AGENTS.md

## Rules of 21

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

## RL

### Action Space

- Draw a card
- Stand

### Observation Space

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

## Agents Guidelines

- Use test-driven development: write failing tests, then code, then refactor.
- Follow Rust 2021 edition and rustfmt defaults.
- Run cargo clippy -- -D warnings before every commit.
- Keep functions short (<20 lines) and modules focused.
- Prefer Result over panic! for recoverable errors.
- Document all public items with /// and an example.
- No global mutable state; pass dependencies explicitly.
- Use #[derive(Debug)] and thiserror for error types.
