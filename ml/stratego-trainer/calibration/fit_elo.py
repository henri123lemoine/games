"""Bradley-Terry / Elo MLE fit over calibration/ladder_results.jsonl.

Every line in the ladder file is one aggregated match result between two
entities (either two DoI levels, from doi_vs_doi.py, or a checkpoint vs a DoI
level, from eval_vs_doi.py-shaped records embedded by run_ladder.sh) with a
win/draw/loss count for each side. A draw counts as half a win for each side
in the Bradley-Terry likelihood. DoI level 0 is fixed as the zero point of the
internal Elo scale (rating = 0, not updated by the optimizer).

Fit by iterative Zermelo/minorization-maximization on Bradley-Terry strengths
`p_i = exp(elo_i / 400)` (the standard Elo <-> BT correspondence), which
converges to the MLE without needing any external optimizer dependency.
Standard errors come from bootstrap resampling of games (with replacement)
within each match line, refitting each resample.

Output calibration/elo_estimates.json: one entry per entity (DoI levels and
checkpoints seen in the data):

    {"<entity>": {"rating": <elo, level-0-anchored>, "stderr": <bootstrap sd>, "n_games": <total games played by this entity>}}
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANCHOR = "doi_l0"


def entity_name(kind: str, key) -> str:
    if kind == "doi":
        return f"doi_l{key}"
    return str(key)


def load_matches(path: Path) -> list[tuple[str, str, float, float]]:
    """Returns a list of (name_a, name_b, wins_a, wins_b) with draws folded in
    as 0.5 wins to each side, one tuple per ladder-file line."""
    matches = []
    if not path.exists():
        return matches
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "level_a" in rec and "level_b" in rec:
            a = entity_name("doi", rec["level_a"])
            b = entity_name("doi", rec["level_b"])
            wa = rec["wins_a"] + 0.5 * rec["draws"]
            wb = rec["wins_b"] + 0.5 * rec["draws"]
        elif "ckpt" in rec and "doi_level" in rec:
            a = rec["ckpt"]
            b = entity_name("doi", rec["doi_level"])
            wa = rec["wins"] + 0.5 * rec["draws"]
            wb = rec["losses"] + 0.5 * rec["draws"]
        else:
            raise RuntimeError(f"unrecognised ladder_results.jsonl record shape: {rec!r}")
        matches.append((a, b, wa, wb))
    return matches


def fit_bt(matches: list[tuple[str, str, float, float]], anchor: str, iters: int = 200) -> dict[str, float]:
    """Zermelo/MM iteration on Bradley-Terry strengths `pi = exp(elo/400)`.
    Returns elo ratings anchored so `elo[anchor] == 0`."""
    entities = sorted({a for a, b, wa, wb in matches} | {b for a, b, wa, wb in matches})
    if anchor not in entities:
        entities.append(anchor)
    p = {e: 1.0 for e in entities}

    for _ in range(iters):
        num = {e: 0.0 for e in entities}
        den = {e: 0.0 for e in entities}
        for a, b, wa, wb in matches:
            s = p[a] + p[b]
            if s <= 0:
                continue
            num[a] += wa
            num[b] += wb
            den[a] += (wa + wb) * p[a] / s
            den[b] += (wa + wb) * p[b] / s
        new_p = {}
        for e in entities:
            new_p[e] = num[e] / den[e] if den[e] > 0 else p[e]
        anchor_scale = new_p[anchor] if new_p[anchor] > 0 else 1.0
        for e in entities:
            new_p[e] = max(new_p[e] / anchor_scale, 1e-9)
        p = new_p

    import math

    return {e: 400.0 * math.log10(pv) for e, pv in p.items()}


def bootstrap_stderr(matches: list[tuple[str, str, float, float]], anchor: str,
                      entities: list[str], n_boot: int, rng) -> dict[str, float]:
    samples: dict[str, list[float]] = {e: [] for e in entities}
    for _ in range(n_boot):
        resampled = []
        for a, b, wa, wb in matches:
            n = wa + wb
            if n <= 0:
                resampled.append((a, b, wa, wb))
                continue
            p_a_win = wa / n
            games = max(int(round(n)), 1)
            wins_a = rng.binomial(games, min(max(p_a_win, 0.0), 1.0))
            resampled.append((a, b, float(wins_a), float(games - wins_a)))
        fit = fit_bt(resampled, anchor)
        for e in entities:
            if e in fit:
                samples[e].append(fit[e])
    out = {}
    for e in entities:
        vals = samples[e]
        if len(vals) < 2:
            out[e] = 0.0
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        out[e] = var ** 0.5
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default=str(HERE / "ladder_results.jsonl"))
    ap.add_argument("--out", default=str(HERE / "elo_estimates.json"))
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    matches = load_matches(Path(args.ladder))
    if not matches:
        print(f"no match records found in {args.ladder}", file=sys.stderr)
        json.dump({}, open(args.out, "w"))
        return

    ratings = fit_bt(matches, ANCHOR)
    entities = sorted(ratings)

    import numpy as np

    rng = np.random.default_rng(args.seed)
    stderrs = bootstrap_stderr(matches, ANCHOR, entities, args.bootstrap, rng)

    n_games = {e: 0.0 for e in entities}
    for a, b, wa, wb in matches:
        n_games[a] += wa + wb
        n_games[b] += wa + wb

    out = {
        e: {"rating": ratings[e], "stderr": stderrs[e], "n_games": n_games[e]}
        for e in entities
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({"n_matches": len(matches), "n_entities": len(entities), "out": args.out}))


if __name__ == "__main__":
    main()
