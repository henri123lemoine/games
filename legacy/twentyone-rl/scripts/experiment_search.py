"""Does inference-time search beat the raw table strategy, significantly?

Plays the search agent head-to-head against the greedy table agent (same solver),
and also runs both against the threshold ladder, reporting win rates with 95%
Wilson intervals. A head-to-head interval that excludes 50% is significant.

    uv run scripts/experiment_search.py --solver data/solver_6h.bin --games 3000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import twentyone
from agents import SolverAgent, ThresholdAgent
from arena import run_match
from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, default=Path("data/solver_6h.bin"))
    parser.add_argument("--games", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    solver = twentyone.Solver.load(str(args.solver))
    logger.info(f"Loaded {args.solver}: {solver.num_infosets()} infosets")
    greedy = SolverAgent(solver, seed=args.seed, mode="greedy")
    search = SolverAgent(solver, seed=args.seed, mode="search")

    # Time the search overhead per move.
    env = twentyone.Env(seed=1)
    env.start_new_round()
    t0 = time.perf_counter()
    for _ in range(2000):
        solver.search_draw(env, env.current_player())
    logger.info(f"search ~{(time.perf_counter() - t0) / 2000 * 1e3:.3f} ms/move")

    print(f"\nHead-to-head ({args.games} games, draws = half):")
    hh = run_match(search, greedy, args.games, seed=args.seed)
    lo, hi = hh.ci0
    sig = "SIGNIFICANT (excludes 50%)" if lo > 0.5 or hi < 0.5 else "not significant"
    print(f"  search vs greedy: {hh.score0 * 100:.1f}%")
    print(f"  95% CI [{lo * 100:.1f}, {hi * 100:.1f}]  -> {sig}")
    print(f"  record (search W-L-D): {hh.wins0}-{hh.wins1}-{hh.draws}")

    print(f"\nVs threshold ladder ({args.games} games each):")
    print(f"  {'opponent':<14}{'greedy':>16}{'search':>16}")
    for t in (15, 16, 17):
        opp_g = ThresholdAgent(t)
        opp_s = ThresholdAgent(t)
        g = run_match(greedy, opp_g, args.games, seed=args.seed)
        s = run_match(search, opp_s, args.games, seed=args.seed)
        glo, ghi = g.ci0
        slo, shi = s.ci0
        print(
            f"  Threshold({t}) "
            f"{g.score0 * 100:>6.1f}% [{glo * 100:.0f},{ghi * 100:.0f}]"
            f"{s.score0 * 100:>8.1f}% [{slo * 100:.0f},{shi * 100:.0f}]"
        )


if __name__ == "__main__":
    main()
