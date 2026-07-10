"""External-only Elo trajectory: each checkpoint rated SOLELY by its games
against Demon of Ignorance (the fixed external opponent), sidestepping the
self-play non-transitivity that compresses the Bradley-Terry fit over
checkpoint-vs-checkpoint games (relatives draw with each other; the BT fit
showed +34 Elo for a week the external ruler scores at ~+400).

Elo vs DoI = 400*log10(p/(1-p)) with a (w+0.5)/(n+1) continuity correction so
perfect scores stay finite; error bars are the same transform applied to a
Wilson 95% interval. Gravon bands use the documented DoI ~= 1200 assumption
(README.md).

Points are hardcoded from measured matches (calibration/doi_trajectory.log and
ladder_results.jsonl DoI lines) — rerun those matches and update here to
extend.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOI_GRAVON = 1200.0

# (cumulative iter, label, wins, draws, losses) — 20 games each
POINTS = [
    (0,    "marathon1c/ckpt_100", 12, 1, 7),
    (600,  "r1/ckpt_600",         17, 0, 3),
    (1200, "r1/ckpt_1200",        13, 2, 5),
    (2100, "r1/ckpt_2100",        17, 3, 0),
    (2800, "r2/ckpt_600",         20, 0, 0),
    (3400, "r2/ckpt_1200",        19, 0, 1),
    (3500, "r2/ckpt_1300",        19, 0, 1),
]

BANDS = [
    ("beginner human", 900, 1300, "tab:purple"),
    ("club amateur", 1300, 1500, "tab:blue"),
    ("strong human", 1500, 1700, "tab:green"),
    ("DeepNash / Ataraxos tier", 1750, 1850, "tab:orange"),
    ("best human", 1850, 2000, "tab:red"),
]


def elo(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return 400.0 * math.log10(p / (1.0 - p))


def wilson(w: float, n: int, z: float = 1.96):
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d, (c + s) / d


def main():
    xs, ys, lo_err, hi_err = [], [], [], []
    for it, _label, w, d, l in POINTS:
        n = w + d + l
        score = w + 0.5 * d
        p = (score + 0.5) / (n + 1)
        p_lo, p_hi = wilson(score, n)
        r = DOI_GRAVON + elo(p)
        xs.append(it)
        ys.append(r)
        lo_err.append(r - (DOI_GRAVON + elo(max(p_lo, 0.02))))
        hi_err.append((DOI_GRAVON + elo(min(p_hi, 0.98))) - r)

    fig, ax = plt.subplots(figsize=(12, 7))
    for label, lo, hi, color in BANDS:
        ax.axhspan(lo, hi, color=color, alpha=0.12, zorder=0)
        ax.text(0.995, (lo + hi) / 2, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", color=color, alpha=0.8)
    ax.axhline(DOI_GRAVON, ls=":", color="gray")
    ax.text(0.005, DOI_GRAVON, "Demon of Ignorance (assumed ~1200 Gravon)",
            transform=ax.get_yaxis_transform(), va="bottom", color="gray")
    ax.errorbar(xs, ys, yerr=[lo_err, hi_err], marker="o", color="crimson",
                capsize=4, lw=2, label="RL checkpoint (rated by DoI games only, n=20)")
    ax.set_xlabel("cumulative training iteration (marathon week; x=0 is the pre-marathon resume point)")
    ax.set_ylabel("assumed Gravon rating (DoI-anchored, external games only)")
    ax.set_title("Stratego RL strength vs the external DoI anchor — self-play games excluded")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    out = "calibration/elo_progress_external.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
