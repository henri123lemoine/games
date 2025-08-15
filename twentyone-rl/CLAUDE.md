# Claude.md

## Python Guidelines

- Runtime: target Python **3.13+** using `uv`.
- Style: autoformat, autosort and lint with **Ruff**; don't fight the formatter. Commands: `uvx ruff format .` for formatting, `uvx ruff check --select I,F401 --fix .` for linting.
- Typing: add **PEP 585 and 604+** type hints; run **mypy** (at least on changed code) and fix errors. Command: `uvx mypy .`
- Tests: use **pytest** when it would be useful; tests must be fast, deterministic, and isolated. Do not over-test, trust your instincts, but implement them when it seems wise and run tests often. Commands: `uvx pytest .`
- Config: keep tool config in **`pyproject.toml`** (formatter, linter, type checker, test settings). Commands: `uv add <package>` (for new packages)
- Errors: raise specific exceptions; no bare `except:`; don't silently swallow exceptions—attach context. Don't spam exceptions, use them to signal errors, and raise them at the appropriate level (often, directly in the function that caused the error).
- Logging: use `loguru`.
- Write clean, idiomatic Python code.
- I/O boundaries: separate pure logic from I/O; prefer `pathlib` for filesystem code.
- Performance: profile before optimizing; readability > cleverness.
- CI gate: code must be Ruff/mypy/pytest clean before merge.
- For now, all code should be in `scripts/`. If it grows too large, it should be refactored and moved as deemed appropriate.

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
