"""End-to-end tests for the path a user actually exercises: train a solver,
save it, load it back, and play full games with the agents.

This is the path that previously failed only at runtime (a stale binary loading
a model file), so it is covered here against the real bindings."""

from __future__ import annotations

from pathlib import Path

import pytest
import twentyone
from agents import RandomAgent, SolverAgent, ThresholdAgent
from arena import play_game, run_match


def _tiny_solver() -> twentyone.Solver:
    solver = twentyone.Solver.abstracted(123, 2)
    solver.solve(300)
    return solver


def test_train_save_load_play(tmp_path: Path) -> None:
    solver = _tiny_solver()
    path = tmp_path / "solver.bin"
    solver.save(str(path))

    reloaded = twentyone.Solver.load(str(path))
    assert reloaded.iterations() == solver.iterations()
    assert reloaded.num_infosets() == solver.num_infosets()

    # The reloaded solver plays full games end to end without error.
    agent = SolverAgent(reloaded, seed=1)
    winner = play_game(agent, ThresholdAgent(17), seed=42)
    assert winner in (0, 1, None)


def test_solver_agent_load_helper(tmp_path: Path) -> None:
    path = tmp_path / "solver.bin"
    _tiny_solver().save(str(path))
    agent = SolverAgent.load(str(path), seed=7)
    # draw_probability is a valid probability for an in-progress state.
    env = twentyone.Env(seed=3)
    env.start_new_round()
    p = env.current_player()
    assert 0.0 <= agent.solver.draw_probability(env, p) <= 1.0


def test_load_rejects_garbage(tmp_path: Path) -> None:
    bad = tmp_path / "garbage.bin"
    bad.write_bytes(b"not a solver file" * 64)
    # Must raise a clean Python exception, not crash the interpreter.
    with pytest.raises(Exception):
        twentyone.Solver.load(str(bad))


def test_baseline_match_runs() -> None:
    result = run_match(RandomAgent(seed=0), ThresholdAgent(17), games=40, seed=1)
    assert result.games == 40
    assert result.wins0 + result.wins1 + result.draws == 40
    lo, hi = result.ci0
    assert 0.0 <= lo <= hi <= 1.0
