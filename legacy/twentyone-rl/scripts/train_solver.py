"""Train a Twenty-One Nash strategy with the Rust external-sampling MCCFR+ solver.

Trains in chunks so convergence can be tracked by exact/sampled best-response
exploitability, then saves the strategy for evaluation and play.

    uv run scripts/train_solver.py --hearts 6 --iters 200000 --out data/solver_6h.bin
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import twentyone
from loguru import logger


def build_solver(hearts: int, seed: int, abstract: bool) -> twentyone.Solver:
    if abstract:
        return twentyone.Solver.abstracted(seed, hearts)
    return twentyone.Solver.with_hearts(seed, hearts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hearts", type=int, default=6, help="starting hearts per player")
    parser.add_argument("--iters", type=int, default=200_000, help="total iters per subgame")
    parser.add_argument("--chunks", type=int, default=10, help="train/measure checkpoints")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument(
        "--eval-deals",
        type=int,
        default=2000,
        help="opening deals sampled for exploitability (0 = exact, -1 = never measure)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="measure exploitability every N chunks (0 = only after the final chunk)",
    )
    parser.add_argument("--out", type=Path, default=Path("data/solver.bin"))
    parser.add_argument(
        "--abstract",
        action="store_true",
        help="key information sets by the unseen-set abstraction (for the full game)",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=0.0,
        help="train to a wall-clock budget instead of a fixed iteration count (fair comparison)",
    )
    parser.add_argument(
        "--chunk-iters",
        type=int,
        default=10_000,
        help="iters/subgame per training step in budget mode",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="train on the whole multi-round game (solve_full) instead of decomposing",
    )
    args = parser.parse_args()

    solver = build_solver(args.hearts, args.seed, args.abstract)
    step_fn = solver.solve_full if args.full else solver.solve

    def measure(final: bool) -> None:
        if args.eval_deals < 0:
            return
        t1 = time.perf_counter()
        br0, br1, nashconv = solver.exploitability(args.eval_deals, args.seed)
        eval_s = time.perf_counter() - t1
        logger.info(
            f"    exploitability={nashconv / 2:.4f} nashconv={nashconv:.4f} "
            f"(br0={br0:+.4f} br1={br1:+.4f}) eval={eval_s:.1f}s"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.budget_seconds > 0:
        logger.info(
            f"Training {args.hearts}-heart solver to a {args.budget_seconds:.0f}s budget "
            f"({args.chunk_iters} iters/subgame per step, abstract={args.abstract})"
        )
        start = time.perf_counter()
        step = 0
        while time.perf_counter() - start < args.budget_seconds:
            step += 1
            t0 = time.perf_counter()
            step_fn(args.chunk_iters)
            elapsed = time.perf_counter() - start
            logger.info(
                f"[{elapsed:6.1f}s] step={step} iters={solver.iterations():>9} "
                f"infosets={solver.num_infosets():>9} (+{time.perf_counter() - t0:.1f}s)"
            )
            solver.save(str(args.out))
    else:
        chunk = max(1, args.iters // args.chunks)
        logger.info(f"Training {args.hearts}-heart solver: {args.chunks} x {chunk} iters/subgame")
        for step in range(1, args.chunks + 1):
            t0 = time.perf_counter()
            step_fn(chunk)
            logger.info(
                f"[{step:>2}/{args.chunks}] iters={solver.iterations():>9} "
                f"infosets={solver.num_infosets():>9} train={time.perf_counter() - t0:.1f}s"
            )
            if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.chunks):
                measure(step == args.chunks)
            solver.save(str(args.out))

    measure(True)
    logger.info(f"Saved solver to {args.out} ({solver.num_infosets()} infosets)")


if __name__ == "__main__":
    main()
