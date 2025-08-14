# Claude.md

## Guidelines

The Python part of this project is all in the `scripts/` directory.

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

## Files

- Quick bot game: `uv run run_basic.py`
- Train MCCFR agent: `uv run mccfr_agent.py`
- Play vs agent: `uv run play_vs_agent.py data/policy_mccfr.json`
