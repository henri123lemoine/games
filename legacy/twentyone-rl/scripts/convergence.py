"""Demonstrate convergence to Nash on small, exactly-measurable Twenty-One variants.

For each starting-hearts variant the solver is trained for increasing iteration
budgets and scored by *exact* best-response exploitability (NashConv / 2, with
utilities in [-1, 1]). Exploitability falls monotonically toward zero, which is
the defining property of a Nash equilibrium — evidence the solver is correct and
converging, on variants small enough to enumerate the best response exactly.

    uv run scripts/convergence.py --hearts 1 --iters 1000 10000 100000 1000000
"""

from __future__ import annotations

import argparse
import time

import twentyone
from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hearts", type=int, default=1, help="starting hearts (keep small)")
    parser.add_argument(
        "--iters",
        type=int,
        nargs="+",
        default=[1000, 10_000, 100_000, 1_000_000],
        help="iteration budgets to evaluate (each trains a fresh solver)",
    )
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    logger.info(f"{args.hearts}-heart variant, exact best-response exploitability")
    print(
        f"\n{'iters/subgame':>14}{'infosets':>12}{'exploitability':>16}{'nashconv':>12}{'time':>9}"
    )
    print("-" * 63)
    for iters in args.iters:
        if args.hearts == 6:
            solver = twentyone.Solver(args.seed)
        else:
            solver = twentyone.Solver.with_hearts(args.seed, args.hearts)
        t0 = time.perf_counter()
        solver.solve(iters)
        _, _, nashconv = solver.exploitability(0, args.seed)
        dt = time.perf_counter() - t0
        print(
            f"{iters:>14}{solver.num_infosets():>12}"
            f"{nashconv / 2:>16.4f}{nashconv:>12.4f}{dt:>8.0f}s"
        )
    print("-" * 63)


if __name__ == "__main__":
    main()
