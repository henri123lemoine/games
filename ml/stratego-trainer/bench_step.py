"""Measure synthetic train-step time + peak memory for the three nets.

Forward + backward + AdamW.update() + EMA.update(), one mx.eval per step.
Synthetic random inputs of the correct shapes. Default batch 1024 (the milestone
sanity target; cf. BENCHMARK.md ~190-450 ms for the move net at b=1024).
"""

import argparse
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

import stratego_nets as S
from stratego_nets.spec import (
    BELIEF_IN_DIM,
    MOVE_IN_DIM,
    N_OCCUPIABLE_CELL,
    N_PIECE_TYPE,
)

WARMUP = 10
STEPS = 30


def _nparams(net):
    return sum(p.size for _, p in tree_flatten(net.parameters()))


def _time(net, loss_fn, ema_decay, bf16):
    if bf16:
        net.set_dtype(mx.bfloat16)
    opt = optim.AdamW(learning_rate=1e-4)
    ema = S.EMA(net, decay=ema_decay)
    lg = nn.value_and_grad(net, loss_fn)

    def step():
        loss, grads = lg(net)
        opt.update(net, grads)
        ema.update(net)
        return loss

    for _ in range(WARMUP):
        loss = step()
        mx.eval(loss, net.parameters(), opt.state, ema.shadow)

    times = []
    for _ in range(STEPS):
        t0 = time.perf_counter()
        loss = step()
        mx.eval(loss, net.parameters(), opt.state, ema.shadow)
        times.append((time.perf_counter() - t0) * 1000.0)
    try:
        peak = mx.get_peak_memory() / 1e9
    except Exception:
        peak = float("nan")
    return statistics.median(times), peak


def bench_move(batch, bf16):
    net = S.MoveTransformer.from_config(S.MoveConfig())
    dt = mx.bfloat16 if bf16 else mx.float32
    obs = mx.random.normal((batch, N_OCCUPIABLE_CELL, MOVE_IN_DIM)).astype(dt)
    tgt = mx.array(np.random.randint(0, 3, (batch,)))
    pol = mx.softmax(mx.random.normal((batch, 1800)) * 0.1, axis=-1)
    mx.eval(obs, tgt, pol)

    def loss_fn(net):
        out = net(obs)
        ml = out["move_logits"].astype(mx.float32)
        logp = ml - mx.logsumexp(ml, axis=-1, keepdims=True)
        pol_loss = -(pol * mx.where(mx.isfinite(logp), logp, 0.0)).sum(-1).mean()
        return pol_loss + nn.losses.cross_entropy(out["value_logp"], tgt, reduction="mean")

    return _nparams(net), _time(net, loss_fn, S.spec.EMA_DECAY_MOVE, bf16)


def bench_setup(batch, bf16):
    net = S.ArrangementTransformer.from_config(S.SetupConfig())
    pc = mx.array(list(S.spec.CLASSIC_PIECE_COUNTS), dtype=mx.float32)
    seq = mx.zeros((batch, 39, N_PIECE_TYPE))
    legal = np.array(net(seq, pc)["logits"]) > np.finfo(np.float32).min
    tgt = mx.array(np.argmax(legal, axis=-1))
    mx.eval(seq, tgt)

    def loss_fn(net):
        out = net(seq, pc)
        logits = out["logits"].astype(mx.float32)
        return nn.losses.cross_entropy(
            logits.reshape(-1, N_PIECE_TYPE), tgt.reshape(-1), reduction="mean"
        )

    return _nparams(net), _time(net, loss_fn, S.spec.EMA_DECAY_SETUP, bf16)


def bench_belief(batch, bf16):
    net = S.BeliefTransformer.from_config(S.BeliefConfig())
    dt = mx.bfloat16 if bf16 else mx.float32
    obs = mx.random.normal((batch, N_OCCUPIABLE_CELL, BELIEF_IN_DIM)).astype(dt)
    pos_np = np.zeros((batch, 40, N_OCCUPIABLE_CELL), dtype=np.float32)
    for b in range(batch):
        cells = np.random.choice(N_OCCUPIABLE_CELL, 10, replace=False)
        pos_np[b, np.arange(10), cells] = 1.0
    pos = mx.array(pos_np)
    idx = np.random.randint(0, 12, (batch, 40))
    typ = mx.array(np.eye(N_PIECE_TYPE, dtype=np.float32)[idx])
    tgt = mx.array(idx)
    active = mx.array((pos_np.sum(-1) > 0).astype(np.float32))
    mx.eval(obs, pos, typ, tgt, active)

    def loss_fn(net):
        logits = net(obs, pos, typ).astype(mx.float32)
        ce = nn.losses.cross_entropy(
            logits.reshape(-1, N_PIECE_TYPE), tgt.reshape(-1), reduction="none"
        ).reshape(batch, 40)
        return (ce * active).sum() / active.sum()

    return _nparams(net), _time(net, loss_fn, S.spec.EMA_DECAY_BELIEF, bf16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--fp32", action="store_true", help="run fp32 instead of bf16")
    args = ap.parse_args()
    bf16 = not args.fp32
    dtype = "bf16" if bf16 else "fp32"
    print(f"batch={args.batch} dtype={dtype}  (fwd+bwd+AdamW+EMA, median of {STEPS} steps)")
    for name, fn in [("move", bench_move), ("setup", bench_setup), ("belief", bench_belief)]:
        mx.random.seed(0)
        nparams, (ms, peak) = fn(args.batch, bf16)
        print(f"  {name:7s} {nparams/1e6:6.2f}M  {ms:8.1f} ms/step  peak {peak:5.2f} GB")


if __name__ == "__main__":
    main()
