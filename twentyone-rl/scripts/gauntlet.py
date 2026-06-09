"""Round-robin gauntlet: prove whether a new model beats all previous ones.

Every competitor (saved solvers + the Random and threshold baselines) plays
every other over `--games` games with alternating seats, scoring win rate with
draws as half. Pairings run in parallel across cores. The output is a standings
table and, for the top model, how many opponents it beats with a 95% Wilson
interval that excludes 50% — a statistically clear win, not noise.

Fairness: each solver's training budget (iterations/subgame) is shown in the
standings. A model that wins only because it trained longer is not a real
improvement — to compare *algorithms*, give every candidate the same training
budget (see train_solver.py --budget-seconds) and check the budgets line up.

    uv run scripts/gauntlet.py --solvers data/bake/*.bin --games 2000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import twentyone
from agents import Agent, RandomAgent, SolverAgent, ThresholdAgent
from arena import MatchResult, run_match
from loguru import logger


@dataclass(frozen=True)
class Spec:
    """Picklable description of a competitor, built into an Agent inside workers."""

    kind: str  # "solver" | "random" | "threshold"
    label: str
    path: str = ""
    mode: str = "greedy"
    threshold: int = 17


_SOLVER_CACHE: dict[str, twentyone.Solver] = {}


def build_agent(spec: Spec, seed: int) -> Agent:
    if spec.kind == "random":
        return RandomAgent(seed=seed)
    if spec.kind == "threshold":
        return ThresholdAgent(spec.threshold)
    solver = _SOLVER_CACHE.get(spec.path)
    if solver is None:
        solver = twentyone.Solver.load(spec.path)
        _SOLVER_CACHE[spec.path] = solver
    return SolverAgent(solver, seed=seed, name=spec.label, mode=spec.mode)


def play_pairing(args: tuple[int, int, Spec, Spec, int, int]) -> tuple[int, int, MatchResult]:
    i, j, spec_a, spec_b, games, seed = args
    res = run_match(build_agent(spec_a, seed), build_agent(spec_b, seed), games, seed=seed)
    return i, j, res


def build_specs(solver_paths: list[Path], mode: str) -> list[Spec]:
    specs: list[Spec] = []
    for path in solver_paths:
        solver = twentyone.Solver.load(str(path))
        label = f"{path.stem}@{solver.iterations() // 1000}k"
        if mode != "greedy":
            label += f"/{mode}"
        specs.append(Spec("solver", label, str(path), mode))
    specs.append(Spec("random", "Random"))
    specs += [Spec("threshold", f"Threshold({t})", threshold=t) for t in (15, 17)]
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solvers", type=Path, nargs="*", default=[], help="model .bin files")
    parser.add_argument("--games", type=int, default=2000, help="games per pairing")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--mode", default="greedy", choices=("greedy", "mixed", "search", "lookahead")
    )
    parser.add_argument("--workers", type=int, default=0, help="processes (0 = auto)")
    args = parser.parse_args()

    specs = build_specs(args.solvers, args.mode)
    n = len(specs)
    logger.info(
        f"Gauntlet: {n} competitors, {args.games} games/pairing, {n * (n - 1) // 2} pairings"
    )

    jobs = [
        (i, j, specs[i], specs[j], args.games, args.seed + i * 131 + j)
        for i in range(n)
        for j in range(i + 1, n)
    ]
    workers = args.workers or None
    with Pool(processes=workers) as pool:
        outcomes = pool.map(play_pairing, jobs)

    results: dict[tuple[int, int], MatchResult] = {(i, j): r for i, j, r in outcomes}
    totals = [0.0] * n
    played = [0] * n
    for (i, j), r in results.items():
        totals[i] += r.score0
        totals[j] += 1.0 - r.score0
        played[i] += 1
        played[j] += 1

    order = sorted(range(n), key=lambda i: totals[i] / played[i], reverse=True)
    print(f"\n{'rank':<5}{'competitor':<26}{'avg win%':>10}")
    print("-" * 42)
    for rank, i in enumerate(order, 1):
        print(f"{rank:<5}{specs[i].label:<26}{totals[i] / played[i] * 100:>9.1f}%")

    def champ_score(j: int) -> float:
        if (champ, j) in results:
            return results[(champ, j)].ci0[0]
        return 1.0 - results[(j, champ)].ci0[1]

    champ = order[0]
    beats = sum(1 for j in range(n) if j != champ and champ_score(j) > 0.5)
    print(
        f"\nChampion: {specs[champ].label} — significantly beats "
        f"{beats}/{n - 1} opponents (95% CI excludes 50%)."
    )


if __name__ == "__main__":
    main()
