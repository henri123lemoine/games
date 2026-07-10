"""Plots calibration/elo_progress.png from calibration/elo_estimates.json:
fitted Elo (DoI-level-0-anchored, from fit_elo.py) vs cumulative training
iteration for our checkpoints, with DoI rungs as horizontal reference lines
and coarse, documented human-skill-tier reference bands.

x-axis mapping (see calibration/README.md for the full justification):
  - runs/marathon1c/ckpt_100.safetensors is the x=0 anchor.
  - runs/marathon_r1/ckpt_<N>.safetensors -> x = N.
  - runs/marathon_r2/ckpt_<N>.safetensors -> x = N + 2200 (marathon_r2 picks
    up training after marathon_r1's ~2200-iteration run, so its own
    iteration counter is offset to stay cumulative on this plot).

Any other checkpoint path found in elo_estimates.json is plotted at its
basename's `ckpt_<N>` iteration number, unmapped (x = N), with a warning to
stderr, so sparse smoke-test data (a lone checkpoint not from the
marathon_r1/marathon_r2/anchor set) still produces a plot rather than
crashing.
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

MARATHON1C_ANCHOR = "runs/marathon1c/ckpt_100.safetensors"
R2_OFFSET = 2200

# Coarse, documented Gravon-rating reference bands. There is no measured
# mapping from this pipeline's internal (DoI-level-0-anchored) Elo scale to
# Gravon rating -- see calibration/README.md "Elo-to-Gravon caveat" for the
# assumption this plot makes to place these bands on the same axis as our
# fitted checkpoint Elo, and treat the band positions as illustrative only.
GRAVON_BANDS = [
    ("beginner human", 1000, 1200, "#8888ff"),
    ("club amateur", 1200, 1500, "#66aaff"),
    ("strong human", 1500, 1700, "#33cc99"),
    ("DeepNash / Ataraxos tier", 1750, 1850, "#ffaa33"),
    ("best human", 1850, 2000, "#ff5555"),
]


def ckpt_iteration(path: str) -> int:
    m = re.search(r"ckpt_(\d+)\.safetensors$", path)
    if not m:
        raise ValueError(f"cannot parse checkpoint iteration from {path!r}")
    return int(m.group(1))


def ckpt_x(path: str) -> float:
    if path.endswith(MARATHON1C_ANCHOR) or path == MARATHON1C_ANCHOR:
        return 0.0
    if "marathon_r1/" in path:
        return float(ckpt_iteration(path))
    if "marathon_r2/" in path:
        return float(ckpt_iteration(path) + R2_OFFSET)
    print(f"warning: {path!r} is not under marathon1c/marathon_r1/marathon_r2, "
          f"plotting at its own raw iteration number unmapped", file=sys.stderr)
    return float(ckpt_iteration(path))


def is_doi_entity(name: str) -> bool:
    return name.startswith("doi_l")


def doi_level(name: str) -> int:
    return int(name[len("doi_l"):])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimates", default=str(HERE / "elo_estimates.json"))
    ap.add_argument("--out", default=str(HERE / "elo_progress.png"))
    ap.add_argument("--gravon-anchor-doi-level", type=int, default=12,
                     help="which DoI rung's fitted Elo is assumed to sit near the "
                          "DeepNash/Ataraxos Gravon-rating band for axis placement")
    ap.add_argument("--gravon-anchor-rating", type=float, default=1790.0)
    args = ap.parse_args()

    estimates = json.loads(Path(args.estimates).read_text()) if Path(args.estimates).exists() else {}

    ckpt_entries = [(name, est) for name, est in estimates.items() if not is_doi_entity(name)]
    doi_entries = [(name, est) for name, est in estimates.items() if is_doi_entity(name)]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 7))

    anchor_level_name = f"doi_l{args.gravon_anchor_doi_level}"
    gravon_offset = None
    for name, est in doi_entries:
        if name == anchor_level_name:
            gravon_offset = args.gravon_anchor_rating - est["rating"]
            break
    if gravon_offset is None and doi_entries:
        # Fall back to the highest-rated DoI rung actually present in the data.
        name, est = max(doi_entries, key=lambda kv: kv[1]["rating"])
        gravon_offset = args.gravon_anchor_rating - est["rating"]
        print(f"warning: {anchor_level_name!r} not in elo_estimates.json, "
              f"anchoring Gravon axis to {name!r} instead", file=sys.stderr)
    if gravon_offset is None:
        gravon_offset = 0.0

    def to_gravon(elo: float) -> float:
        return elo + gravon_offset

    for label, lo, hi, color in GRAVON_BANDS:
        ax.axhspan(lo - gravon_offset, hi - gravon_offset, color=color, alpha=0.12, zorder=0)
        ax.text(0.995, (lo + hi) / 2 - gravon_offset, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8, color=color, alpha=0.9,
                bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.6))

    for name, est in sorted(doi_entries, key=lambda kv: kv[1]["rating"]):
        ax.axhline(est["rating"], linestyle=":", color="grey", linewidth=1.2, zorder=1)
        ax.text(0.01, est["rating"], f"DoI level {doi_level(name)}  (Elo {est['rating']:.0f})",
                transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=8, color="dimgrey")

    points = sorted(((ckpt_x(name), est["rating"], est["stderr"], name) for name, est in ckpt_entries),
                     key=lambda t: t[0])
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        errs = [p[2] for p in points]
        ax.errorbar(xs, ys, yerr=errs, marker="o", color="#d1495b", ecolor="#d1495b",
                     capsize=3, linewidth=1.8, markersize=5, zorder=3, label="RL checkpoint")
        ax.legend(loc="upper left")
    else:
        print("warning: no checkpoint entries in elo_estimates.json -- plotting DoI rungs only", file=sys.stderr)

    ax.set_xlabel("cumulative training iteration (marathon1c anchor=0, "
                  "marathon_r1 as-is, marathon_r2 + 2200)")
    ax.set_ylabel(f"fitted Elo (DoI level 0 = 0)  /  assumed Gravon rating "
                  f"(DoI level {args.gravon_anchor_doi_level} <-> {args.gravon_anchor_rating:.0f})")
    ax.set_title("Stratego RL checkpoint strength vs DoI-anchored Elo ladder")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(json.dumps({"out": args.out, "n_ckpt_points": len(points), "n_doi_rungs": len(doi_entries)}))


if __name__ == "__main__":
    main()
