"""Export a RunDir checkpoint to the browser inference artifact (`ATRX1`).

Writes the move + setup net weights as the fp16 container that
`nn_infer::stratego::StrategoNet::parse` consumes (see `ml/nn-infer/src/
stratego.rs` for the authoritative layout), plus an optional parity fixture:
deterministic pseudo-random inputs and the MLX forward outputs computed with
the same fp16-roundtripped weights the artifact stores, so the Rust test
tolerance covers accumulation order only, never quantization.

The exported copy is the checkpoint's *raw* (`.model`) weights, not the EMA
shadow: the pure_r4 end-of-run net-vs-net eval measured raw-7600 beating its
own EMA (the fully-decayed LR leaves the EMA ~1k iterations behind). Pass
--ema to export the shadow instead when a future run measures the other way.

Usage:
    python export_web.py --ckpt runs/<run>/ckpt_XXXX.safetensors \
        --artifact ../../web/app/public/artifacts/ataraxios.bin \
        --fixture ../../games/stratego/tests/fixtures/net_parity.json
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from stratego_nets.config import MOVE_REF, SETUP_REF
from stratego_nets.nets import ArrangementTransformer, MoveTransformer, _causal_mask
from stratego_nets.spec import CLASSIC_PIECE_COUNTS, MOVE_IN_DIM, N_PIECE_TYPE

MAGIC = b"ATRX1\0\0\0"
VERSION = 1
N_MOVE_TOKENS = 92
N_SETUP_TOKENS = 40


def lcg_stream(seed: int):
    """The fixture PRNG, mirrored bit-for-bit by the Rust parity test."""
    x = seed
    while True:
        x = (x * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        yield (x >> 40) / float(1 << 24)


def header_bytes() -> bytes:
    words = [
        VERSION,
        MOVE_REF.depth, MOVE_REF.embed_dim, MOVE_REF.n_head, MOVE_IN_DIM, N_MOVE_TOKENS,
        SETUP_REF.depth, SETUP_REF.embed_dim, SETUP_REF.n_head, N_PIECE_TYPE, N_SETUP_TOKENS,
        0,
    ]
    return MAGIC + struct.pack("<12I", *words)


def linear_tensors(params: dict, path: str) -> list:
    node = params
    for part in path.split("."):
        node = node[int(part)] if part.isdigit() else node[part]
    return [node["weight"], node["bias"]]


def trunk_tensors(params: dict, depth: int) -> list:
    out = linear_tensors(params, "embedder")
    out.append(params["positional_encoding"])
    for i in range(depth):
        out += linear_tensors(params, f"layers.{i}.layer_norm1")
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            out += linear_tensors(params, f"layers.{i}.mha.{proj}")
        out += linear_tensors(params, f"layers.{i}.layer_norm2")
        out += linear_tensors(params, f"layers.{i}.ff.linear1")
        out += linear_tensors(params, f"layers.{i}.ff.linear2")
    out += linear_tensors(params, "norm_out")
    return out


def export_tensors(move_params: dict, setup_params: dict) -> list:
    out = trunk_tensors(move_params, MOVE_REF.depth)
    out += linear_tensors(move_params, "q_proj")
    out += linear_tensors(move_params, "k_proj")
    out += linear_tensors(move_params, "value_head")
    out.append(setup_params["start_token"])
    out += trunk_tensors(setup_params, SETUP_REF.depth)
    out += linear_tensors(setup_params, "policy_out")
    out += linear_tensors(setup_params, "value_out")
    out += linear_tensors(setup_params, "ent_out")
    return out


def collect(flat: dict, prefix: str) -> dict:
    plen = len(prefix) + 1
    sub = {k[plen:]: v for k, v in flat.items() if k.startswith(prefix + ".")}
    if not sub:
        raise SystemExit(f"checkpoint has no '{prefix}.*' weights")
    return tree_unflatten(list(sub.items()))


def roundtrip_f16(params):
    return tree_unflatten(
        [
            (k, mx.array(np.asarray(v, dtype=np.float32).astype(np.float16).astype(np.float32)))
            for k, v in tree_flatten(params)
        ]
    )


def setup_raw_logits(net: ArrangementTransformer, seq: mx.array) -> mx.array:
    """The policy head without the internal legal mask, last position."""
    start = mx.broadcast_to(net.start_token, (1, 1, N_PIECE_TYPE))
    full = mx.concatenate([start, seq], axis=1)[:, :N_SETUP_TOKENS]
    x = net.embedder(full) + net.positional_encoding[:, : full.shape[1]]
    mask = _causal_mask(x.shape[1])
    for layer in net.layers:
        x = layer(x, mask)
    return net.policy_out(net.norm_out(x))[0, -1]


def build_fixture(move_params: dict, setup_params: dict) -> bytes:
    """Binary fixture (`ATRXFIX1`): expected outputs only — the Rust test
    regenerates the LCG inputs. Two move cases (1800 logits + 3 value_logp
    each), then two setup cases (u32 prefix length, then 14 raw logits)."""
    move_net = MoveTransformer.from_config(MOVE_REF)
    move_net.update(move_params)
    setup_net = ArrangementTransformer.from_config(SETUP_REF)
    setup_net.update(setup_params)

    stream = lcg_stream(0x243F6A8885A308D3)
    out = bytearray(b"ATRXFIX1")
    out += struct.pack("<I", 2)
    for _ in range(2):
        obs = np.fromiter(
            (next(stream) for _ in range(N_MOVE_TOKENS * MOVE_IN_DIM)),
            dtype=np.float32,
        ).reshape(1, N_MOVE_TOKENS, MOVE_IN_DIM)
        fwd = move_net(mx.array(obs))
        out += np.asarray(fwd["move_logits"][0], dtype="<f4").tobytes()
        out += np.asarray(fwd["value_logp"][0], dtype="<f4").tobytes()

    supply_order = [t for t, c in enumerate(CLASSIC_PIECE_COUNTS) for _ in range(c)]
    prefixes = (0, 17)
    out += struct.pack("<I", len(prefixes))
    for prefix_len in prefixes:
        seq = np.zeros((1, prefix_len, N_PIECE_TYPE), dtype=np.float32)
        for slot, kind in enumerate(supply_order[:prefix_len]):
            seq[0, slot, kind] = 1.0
        logits = setup_raw_logits(setup_net, mx.array(seq))
        out += struct.pack("<I", prefix_len)
        out += np.asarray(logits, dtype="<f4").tobytes()
    return bytes(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--ema", action="store_true", help="export the EMA shadow instead of raw")
    args = ap.parse_args()

    flat = mx.load(args.ckpt)
    kind = "ema" if args.ema else "model"
    move_params = collect(flat, f"move.{kind}")
    setup_params = collect(flat, f"setup.{kind}")

    body = bytearray()
    for t in export_tensors(move_params, setup_params):
        body += np.asarray(t, dtype=np.float32).astype("<f2").tobytes()
    artifact = Path(args.artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(header_bytes() + bytes(body))
    print(f"wrote {artifact} ({artifact.stat().st_size / 1e6:.1f} MB, {kind} weights)")

    if args.fixture:
        blob = build_fixture(roundtrip_f16(move_params), roundtrip_f16(setup_params))
        fixture = Path(args.fixture)
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_bytes(blob)
        print(f"wrote {fixture} ({len(blob)} bytes)")


if __name__ == "__main__":
    main()
