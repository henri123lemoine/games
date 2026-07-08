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

prints one JSON line. With ``--opponent random`` (default) it is
``{"ws_rand": float, "ema_ws_rand": float}``; with ``--opponent heuristic`` it
is ``{"ws_heur": float, "ema_ws_heur": float}`` (win share of the working and
EMA nets vs the codebase `HeuristicBot` baseline); with ``--opponent
<path.safetensors>`` it is ``{"ws_ckpt": ..., "ema_ws_ckpt": ...,
"opponent_ckpt": ...}`` (direct net-vs-net vs that checkpoint's EMA weights —
the only measurement that shows self-play improvement when win share against
fixed scripted baselines has saturated or drifted). The net size is read back
from the checkpoint's own metadata (`RunDir` stamps `net_size` at save time),
so evaluating a `mid`/`ref`-size run needs no extra flag.

Gate-decision noise: win share at n=200 games carries measured seed-to-seed
spread of ~0.04-0.08 (three same-checkpoint evals at different seeds gave
0.135/0.105/0.085) despite the pipeline itself being fully deterministic (the
same checkpoint + same seed reproduces bit-for-bit identical win/draw/loss
counts — verified directly, not assumed). That spread is real sampling noise,
not a bug, and it is large enough to flip a close-to-threshold gate call. Pass
``--seeds`` (comma-separated) to average several independent seeds and report
the spread alongside the mean for exactly that reason.
"""

import argparse
import json
import statistics

import mlx.core as mx

import stratego_nets as S

from .eval import win_share
from .rundir import load_checkpoint


def _net_size_of(ckpt: str) -> str:
    _, meta = mx.load(ckpt, return_metadata=True)
    return meta.get("net_size", "default")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--games", type=int, default=200)
    # Reference parity (`final_run/train.log`'s `max_num_moves: 4000`) — a GATE
    # decision should measure real, decisive-outcome strength, not a truncated
    # proxy. The periodic in-run telemetry eval (train.py) overrides this to a
    # much cheaper profile; see `TrainConfig.eval_move_cap`.
    ap.add_argument("--move-cap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=str, default=None,
                     help="Comma-separated seed list; overrides --seed and averages "
                          "win share across all of them, reporting the stdev too.")
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--opponent", default="random",
                     help="random | heuristic | path to an opponent checkpoint "
                          "(.safetensors, EMA weights preferred) for a direct "
                          "net-vs-net win share.")
    args = ap.parse_args()

    heuristic = args.opponent == "heuristic"
    opp_ckpt = None if args.opponent in ("random", "heuristic") else args.opponent
    move_cfg, setup_cfg = S.NET_SIZES[_net_size_of(args.ckpt)]

    def load(ckpt, cfgs, prefer_ema: bool):
        move = S.MoveTransformer.from_config(cfgs[0])
        setup = S.ArrangementTransformer.from_config(cfgs[1])
        load_checkpoint(ckpt, move=move, setup=setup, prefer_ema=prefer_ema)
        return move, setup

    opp_move, opp_setup = None, None
    if opp_ckpt is not None:
        opp_cfgs = S.NET_SIZES[_net_size_of(opp_ckpt)]
        opp_move, opp_setup = load(opp_ckpt, opp_cfgs, prefer_ema=True)

    def winrate(move, setup, seed):
        return win_share(
            move, setup, opp_move, opp_setup,
            num_envs=args.num_envs, games=args.games, move_cap=args.move_cap,
            seed=seed, hero_temperature=args.temperature, heuristic=heuristic,
        )

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    hero_cfgs = (move_cfg, setup_cfg)
    working_net = load(args.ckpt, hero_cfgs, prefer_ema=False)
    ema_net = load(args.ckpt, hero_cfgs, prefer_ema=True)
    ws_per_seed = [winrate(*working_net, seed) for seed in seeds]
    ema_ws_per_seed = [winrate(*ema_net, seed) for seed in seeds]
    ws = statistics.fmean(ws_per_seed)
    ema_ws = statistics.fmean(ema_ws_per_seed)

    if heuristic:
        key, ema_key = "ws_heur", "ema_ws_heur"
    elif opp_ckpt is not None:
        key, ema_key = "ws_ckpt", "ema_ws_ckpt"
    else:
        key, ema_key = "ws_rand", "ema_ws_rand"
    out = {key: ws, ema_key: ema_ws}
    if opp_ckpt is not None:
        out["opponent_ckpt"] = opp_ckpt
    if len(seeds) > 1:
        out[f"{key}_stdev"] = statistics.pstdev(ws_per_seed)
        out[f"{ema_key}_stdev"] = statistics.pstdev(ema_ws_per_seed)
        out["seeds"] = seeds
    print(json.dumps(out))


if __name__ == "__main__":
    main()
