"""Checkpoint round-robin for the Elo fit.

DoI's -l level knob proved to be a no-op in test mode (a weak checkpoint
scored 0.625 vs -l0 and 0.65 vs -l12; DoI self-play at any level pairing is
~100% fast repetition draws), so an external multi-rung ladder does not
exist. The Elo scale is instead built from checkpoint-vs-checkpoint matches
through the Rust sim (precise relative spacing) pinned to the single DoI
external anchor via the eval_vs_doi.py results.

Plays adjacent + stride-3 pairs over a subsampled checkpoint chain spanning
marathon1c -> marathon_r1 -> marathon_r2, 200 games each (EMA weights both
sides, temperature 0.25), and appends ckpt_a/ckpt_b records to
ladder_results.jsonl. Idempotent: pairs already present are skipped.

Run from ml/stratego-trainer: .venv/bin/python calibration/run_roundrobin.py
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "ladder_results.jsonl"
GAMES = 200

CHAIN = ["runs/marathon1c/ckpt_100.safetensors"] + [
    f"runs/marathon_r1/ckpt_{n}.safetensors" for n in range(300, 2101, 300)
] + [
    f"runs/marathon_r2/ckpt_{n}.safetensors" for n in range(300, 1500, 300)
]


def pairs(chain):
    for i in range(len(chain) - 1):
        yield chain[i], chain[i + 1]
    for i in range(len(chain) - 3):
        yield chain[i], chain[i + 3]


def existing_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "ckpt_a" in rec:
                keys.add((rec["ckpt_a"], rec["ckpt_b"]))
    return keys


def main():
    done = existing_keys()
    todo = [(a, b) for a, b in pairs(CHAIN) if (a, b) not in done]
    for i, (a, b) in enumerate(todo, 1):
        missing = [p for p in (a, b) if not (ROOT / p).exists()]
        if missing:
            print(f"skip (missing ckpt): {missing}", file=sys.stderr)
            continue
        cmd = [str(ROOT / ".venv/bin/python"), "-m", "stratego_trainer.eval_ckpt",
               "--ckpt", a, "--opponent", b, "--games", str(GAMES), "--seed", "0"]
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        try:
            res = json.loads(out.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            print(f"match {a} vs {b} FAILED:\n{out.stderr[-500:]}", file=sys.stderr)
            continue
        ws = res["ema_ws_ckpt"]
        rec = {"ckpt_a": a, "ckpt_b": b,
               "wins_a": round(ws * GAMES, 1), "wins_b": round((1 - ws) * GAMES, 1),
               "games": GAMES, "ema": True}
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{i}/{len(todo)}] {a} vs {b}: ws_a={ws:.3f}")


if __name__ == "__main__":
    main()
