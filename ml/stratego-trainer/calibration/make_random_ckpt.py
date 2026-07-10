"""Writes a randomly-initialized "default"-size checkpoint for calibration
smoke tests. Not a trained network — its only purpose is to give
`eval_vs_doi.py` something that loads through the real `eval_ckpt.py` /
`rundir.load_checkpoint` path (EMA + working weights both present) without
touching anything under any `runs/` directory.

    .venv/bin/python calibration/make_random_ckpt.py calibration/random_ckpt.safetensors
"""

import sys

import mlx.core as mx
from mlx.utils import tree_flatten

import stratego_nets as S


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "calibration/random_ckpt.safetensors"
    move_cfg, setup_cfg = S.NET_SIZES["default"]
    move = S.MoveTransformer.from_config(move_cfg)
    setup = S.ArrangementTransformer.from_config(setup_cfg)
    mx.eval(move.parameters(), setup.parameters())

    flat = {}
    for prefix, tree in (
        ("move.model", move.parameters()),
        ("move.ema", move.parameters()),
        ("setup.model", setup.parameters()),
        ("setup.ema", setup.parameters()),
    ):
        for k, v in tree_flatten(tree):
            flat[f"{prefix}.{k}"] = v
    mx.save_safetensors(out_path, flat, metadata={"step": "0", "net_size": "default"})
    print(f"wrote random checkpoint to {out_path}")


if __name__ == "__main__":
    main()
