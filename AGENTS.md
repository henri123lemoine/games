# Agents

## Guidelines

### Python

- Runtime: target Python **3.13+** using `uv`.
- Style: autoformat, autosort and lint with **Ruff**; don't fight the formatter. Command: `uvx ruff format .` as well as `uvx ruff check --select I,F401 --fix .`
- Typing: add **PEP 484+** type hints; run **mypy** (at least on changed code) and fix errors. Command: `uvx mypy .`
- Tests: use **pytest** when it would be useful; tests must be fast, deterministic, and isolated. Do not over-test, trust your instincts, but implement them when it seems wise and run tests often. Command: `uvx pytest .`
- Config: keep tool config in **`pyproject.toml`** (formatter, linter, type checker, test settings). Command: `uv add <package>` (for new packages)
- Errors: raise specific exceptions; no bare `except:`; don't silently swallow exceptions—attach context. Don't spam exceptions, use them to signal errors, and raise them at the appropriate level (often, directly in the function that caused the error).
- Logging: use `loguru`.
- Write clean, idiomatic Python code.
- I/O boundaries: separate pure logic from I/O; prefer `pathlib` for filesystem code.
- Performance: profile before optimizing; readability > cleverness.
- CI gate: code must be Ruff/mypy/pytest clean before merge.
- For now, all code should be in `scripts/`. If it grows too large, it should be refactored and moved as deemed appropriate.

### Rust

- Toolchain: use **stable** with **Edition 2024**.
- Formatting: run **rustfmt** (set `style_edition = "2024"`); no manual formatting.
- Lints: run **clippy** and fix warnings; treat warnings as errors in CI when feasible.
- Layout: follow **Cargo** conventions—`src/lib.rs`, `src/main.rs`, optional `src/bin/`, `tests/`, `benches/`, `examples/`; use **workspaces** for multi-crate repos.
- Public API: keep the surface small and consistent; follow the **Rust API Guidelines**; respect SemVer and document MSRV.
- Errors: return `Result<T, E>`; use **`thiserror`** for library error types; **`anyhow`** is fine for binaries; avoid `unwrap`/`expect` outside tests/bin, though occasionally acceptable during development.
- Ownership: prefer borrowing (`&str`, slices) and avoid unnecessary clones; minimize `Arc/Mutex` to true sharing needs.
- Concurrency/async: don’t block in async contexts; prefer structured concurrency and graceful shutdown. Use async rust only when it is strictly necessary, or helps performance tremendously or when it does not add significant complexity.
- Docs: write `///` rustdoc for public items; include examples that compile as doctests.
- Testing: unit tests inline (`#[cfg(test)]`); integration tests in `tests/`; `cargo test` must pass locally and in CI.
- Features: keep defaults minimal; features should be additive and orthogonal.
- Logging: use a logging/tracing facade in libraries; avoid `println!` in library code, though here it's acceptable.
- Unsafe: isolate and document safety invariants; prefer safe abstractions first.
- CI gate: `cargo fmt --check`, `cargo clippy -- -D warnings`, and `cargo test` must be clean before merge.

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

## Agents Guidelines

- Use test-driven development: write failing tests, then code, then refactor.
- Follow Rust 2021 edition and rustfmt defaults.
- Run cargo clippy -- -D warnings before every commit.
- Keep functions short (<20 lines) and modules focused.
- Prefer Result over panic! for recoverable errors.
- Document all public items with /// and an example.
- No global mutable state; pass dependencies explicitly.
- Use #[derive(Debug)] and thiserror for error types.
