"""Round-robin gauntlet: prove whether a new model beats all previous ones.

Every competitor (saved solvers + the Random and threshold baselines) plays
every other over `--games` games with alternating seats, scoring win rate with
draws as half. The output is a standings table and a pairwise matrix, and for the
top model it reports how many opponents it beats with a 95% Wilson interval that
excludes 50% — i.e. a statistically clear win, not noise.

Fairness: each solver's training budget (iterations/subgame) is shown in the
standings. A model that wins only because it trained longer is not a real
improvement — to compare *algorithms*, give every candidate the same `--iters`
when training (see train_solver.py) and check the budgets line up here.

    uv run scripts/gauntlet.py --solvers data/*.bin --games 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import twentyone
from agents import Agent, RandomAgent, SolverAgent, ThresholdAgent
from arena import MatchResult, run_match
from loguru import logger


def build_competitors(solver_paths: list[Path], mode: str, seed: int) -> list[Agent]:
    competitors: list[Agent] = []
    for path in solver_paths:
        solver = twentyone.Solver.load(str(path))
        name = f"{path.stem}@{solver.iterations() // 1000}k"
        competitors.append(SolverAgent(solver, seed=seed, name=name, mode=mode))
    competitors.append(RandomAgent(seed=seed))
    competitors += [ThresholdAgent(t) for t in (15, 17)]
    return competitors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solvers", type=Path, nargs="*", default=[], help="model .bin files")
    parser.add_argument("--games", type=int, default=2000, help="games per pairing")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", default="greedy", choices=("greedy", "mixed", "search"))
    args = parser.parse_args()

    competitors = build_competitors(args.solvers, args.mode, args.seed)
    names = [a.name for a in competitors]
    logger.info(f"Gauntlet: {len(competitors)} competitors, {args.games} games/pairing")

    # score[i] accumulates i's average win rate over all its pairings; results
    # keeps each pairing for the matrix and the significance check.
    results: dict[tuple[int, int], MatchResult] = {}
    totals = [0.0] * len(competitors)
    pairings = [0] * len(competitors)
    for i in range(len(competitors)):
        for j in range(i + 1, len(competitors)):
            res = run_match(
                competitors[i], competitors[j], args.games, seed=args.seed + i * 100 + j
            )
            results[(i, j)] = res
            totals[i] += res.score0
            totals[j] += 1.0 - res.score0
            pairings[i] += 1
            pairings[j] += 1

    order = sorted(range(len(competitors)), key=lambda i: totals[i] / pairings[i], reverse=True)

    print(f"\n{'rank':<5}{'competitor':<22}{'avg win%':>10}")
    print("-" * 40)
    for rank, i in enumerate(order, 1):
        print(f"{rank:<5}{names[i]:<22}{totals[i] / pairings[i] * 100:>9.1f}%")

    def pair(i: int, j: int) -> tuple[float, tuple[float, float]]:
        if (i, j) in results:
            r = results[(i, j)]
            return r.score0, r.ci0
        r = results[(j, i)]
        return 1.0 - r.score0, (1.0 - r.ci0[1], 1.0 - r.ci0[0])

    champ = order[0]
    beats = 0
    for j in range(len(competitors)):
        if j == champ:
            continue
        _, (lo, _) = pair(champ, j)
        if lo > 0.5:
            beats += 1
    print(
        f"\nChampion: {names[champ]} — significantly beats "
        f"{beats}/{len(competitors) - 1} opponents (95% CI excludes 50%)."
    )


if __name__ == "__main__":
    main()
