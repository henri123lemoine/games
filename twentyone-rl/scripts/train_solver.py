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
    args = parser.parse_args()

    solver = build_solver(args.hearts, args.seed, args.abstract)
    chunk = max(1, args.iters // args.chunks)
    logger.info(
        f"Training {args.hearts}-heart solver: {args.chunks} x {chunk} iters/subgame "
        f"(eval_deals={args.eval_deals}, eval_every={args.eval_every})"
    )

    def measure(step: int) -> None:
        if args.eval_deals < 0:
            return
        is_final = step == args.chunks
        due = args.eval_every > 0 and step % args.eval_every == 0
        if not (is_final or due):
            return
        t1 = time.perf_counter()
        br0, br1, nashconv = solver.exploitability(args.eval_deals, args.seed)
        eval_s = time.perf_counter() - t1
        logger.info(
            f"    exploitability={nashconv / 2:.4f} nashconv={nashconv:.4f} "
            f"(br0={br0:+.4f} br1={br1:+.4f}) eval={eval_s:.1f}s"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.chunks + 1):
        t0 = time.perf_counter()
        solver.solve(chunk)
        train_s = time.perf_counter() - t0
        logger.info(
            f"[{step:>2}/{args.chunks}] iters={solver.iterations():>9} "
            f"infosets={solver.num_infosets():>9} train={train_s:.1f}s"
        )
        measure(step)
        solver.save(str(args.out))

    logger.info(f"Saved solver to {args.out} ({solver.num_infosets()} infosets)")


if __name__ == "__main__":
    main()
