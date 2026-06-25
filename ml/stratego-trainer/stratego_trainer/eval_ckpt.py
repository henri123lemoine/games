"""Standalone checkpoint eval — runs in a SEPARATE process.

The in-training eval (`win_share`'s heavy self-play forwards) corrupts the
trainer's MLX/Metal runtime and drives the learner non-finite about one
iteration later. This is proven by a controlled pair of runs that share a
byte-identical training trajectory and diverge only at the eval: eval-off
survives, eval-on dies. No in-process isolation (snapshotting the eval's inputs,
or snapshotting+restoring the learner around the eval) stops it, because the
damage is to the shared runtime, not the param values.

Running the eval in its own process gives it an isolated MLX runtime that
physically cannot touch the trainer's. Usage:

    python -m stratego_trainer.eval_ckpt --ckpt <latest.safetensors> [...]

prints one JSON line: ``{"ws_rand": float, "ema_ws_rand": float}``.
"""

import argparse
import json

import stratego_nets as S

from .eval import win_share
from .rundir import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--move-cap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.25)
    args = ap.parse_args()

    def load(prefer_ema: bool):
        move = S.MoveTransformer.from_config(S.MoveConfig())
        setup = S.ArrangementTransformer.from_config(S.SetupConfig())
        load_checkpoint(args.ckpt, move=move, setup=setup, prefer_ema=prefer_ema)
        return move, setup

    def winrate(move, setup):
        return win_share(
            move, setup, None, None,
            num_envs=args.num_envs, games=args.games, move_cap=args.move_cap,
            seed=args.seed, hero_temperature=args.temperature,
        )

    ws_rand = winrate(*load(prefer_ema=False))
    ema_ws_rand = winrate(*load(prefer_ema=True))
    print(json.dumps({"ws_rand": ws_rand, "ema_ws_rand": ema_ws_rand}))


if __name__ == "__main__":
    main()
