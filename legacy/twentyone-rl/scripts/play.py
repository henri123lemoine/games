"""Play Twenty-One against a trained solver from the terminal.

    uv run scripts/play.py --solver data/solver_6h.bin

You control one seat; the solver plays the other with its learned strategy.
At each of your turns, enter ``d`` to draw or ``s`` to stand.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import twentyone
from agents import SolverAgent

from twentyone_rl.display import (
    check_game_over,
    show_action_choice,
    show_game_start,
    show_hearts_status,
    show_round_result,
    show_round_start,
    show_turn_info,
)


def human_action(env: twentyone.Env, player: int) -> twentyone.Action:
    obs = env.observation(player)
    while True:
        choice = (
            input(f"  Your move (total={obs.self_total}) — [d]raw or [s]tand? ").strip().lower()
        )
        if choice in ("d", "draw"):
            return twentyone.Action.Draw
        if choice in ("s", "stand"):
            return twentyone.Action.Stand
        print("  Please enter 'd' or 's'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, default=Path("data/solver_6h.bin"))
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0, help="seat you control")
    parser.add_argument("--seed", type=int, default=None, help="game seed (random if omitted)")
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="solver samples its strategy instead of playing greedily",
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    solver_agent = SolverAgent.load(str(args.solver), seed=seed, greedy=not args.mixed)
    human_seat = args.seat
    names = ["You", "Solver"] if human_seat == 0 else ["Solver", "You"]
    names_t = (names[0], names[1])

    env = twentyone.Env(seed=seed)
    show_game_start(env, seed, names_t)

    while not check_game_over(env):
        env.start_new_round()
        show_round_start(env.round())
        p0_up: list[int] = []
        p1_up: list[int] = []
        while True:
            player = env.current_player()
            obs = env.observation(player)
            p0_up, p1_up = show_turn_info(env, player, obs, names_t)
            if player == human_seat:
                action = human_action(env, player)
            else:
                action = solver_agent.act(env, player)
            show_action_choice(names[player], action)
            result = env.step(action)
            if result.round_over or result.game_over:
                show_round_result(env, result, p0_up, p1_up, names_t)
                show_hearts_status(env, names_t)
                break

    winner = 0 if env.hearts(0) > env.hearts(1) else 1
    print(f"\n=== {names[winner]} win{'' if names[winner] == 'You' else 's'}! ===")


if __name__ == "__main__":
    main()
