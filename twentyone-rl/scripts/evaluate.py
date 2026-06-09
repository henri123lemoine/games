"""Evaluate a trained Twenty-One solver against baseline agents.

Reports head-to-head win rates (draws counted as half) with 95% Wilson
confidence intervals, plus the solver's exact/sampled exploitability.

    uv run scripts/evaluate.py --solver data/solver_6h.bin --games 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import twentyone
from agents import Agent, RandomAgent, SolverAgent, ThresholdAgent
from arena import run_match
from loguru import logger


def baselines(seed: int) -> list[Agent]:
    agents: list[Agent] = [RandomAgent(seed=seed)]
    agents += [ThresholdAgent(t) for t in (14, 15, 16, 17, 18, 19)]
    return agents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True, help="path to a saved solver .bin")
    parser.add_argument("--games", type=int, default=2000, help="games per matchup")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--eval-deals", type=int, default=0, help="exploitability deals (0=exact)")
    args = parser.parse_args()

    solver = twentyone.Solver.load(str(args.solver))
    hero = SolverAgent(solver, seed=args.seed, name="Solver")
    logger.info(
        f"Loaded {args.solver}: {solver.iterations()} iters/subgame, "
        f"{solver.num_infosets()} infosets"
    )

    br0, br1, nashconv = solver.exploitability(args.eval_deals, args.seed)
    logger.info(
        f"Exploitability={nashconv / 2:.4f} (nashconv={nashconv:.4f}, "
        f"br0={br0:+.4f} br1={br1:+.4f}); utilities in [-1, 1]"
    )

    print(f"\n{'Opponent':<16}{'Solver win%':>14}{'95% CI':>20}{'record (W-L-D)':>18}")
    print("-" * 68)
    for opp in baselines(args.seed):
        res = run_match(hero, opp, args.games, seed=args.seed)
        lo, hi = res.ci0
        print(
            f"{opp.name:<16}{res.score0 * 100:>13.1f}%"
            f"{f'[{lo * 100:.1f}, {hi * 100:.1f}]':>20}"
            f"{f'{res.wins0}-{res.wins1}-{res.draws}':>18}"
        )
    print("-" * 68)


if __name__ == "__main__":
    main()
