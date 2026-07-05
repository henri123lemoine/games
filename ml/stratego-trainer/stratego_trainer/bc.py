"""Move-net behavior cloning from HeuristicBot self-play — the RL warm start.

A from-scratch net can't learn to attack fast enough for the reference's tight
100-ply no-attack clock to ever see a decisive result (measured 2026-07-04:
HeuristicBot attacks 24.5/100 plies, draw_frac 0.03; an iter-100 RL net is
already down to 0.6/100, draw_frac 0.99 — self-play collapses into universal
passivity before the value head ever gets a real signal). BC starts the move
net on the far side of that trap: supervised cross-entropy on HeuristicBot's
own chosen actions plus a categorical CE of the value head against the
teacher's true game outcome (exact, not bootstrapped — see `bc_rs`'s doc).

Streams chunks from `stratego_sim.generate_bc_games` (generation is ~137k
rows/sec, far cheaper than training) rather than materializing one huge
dataset — a full obs tensor for the hundreds of thousands of decisions a
real BC pass wants would be tens of GB.

The setup net is untouched (HeuristicBot's deployment is a uniform-random
legal fill — nothing to clone) but is still saved alongside the BC-trained
move net in the same combined-checkpoint format `train.py`'s `--resume`
expects, so a BC output loads there transparently.
"""

import argparse
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import stratego_nets as S
import stratego_sim as sim

from .config import TrainConfig
from .rundir import RunDir
from .train import build


@dataclass
class BcConfig:
    net_size: str = "default"
    seed: int = 0
    # Games generated per chunk (a few thousand move decisions each — see the
    # module docstring on why this streams instead of one big dataset).
    games_per_chunk: int = 60
    epsilon: float = 0.05
    lr: float = 3e-4
    batch_size: int = 512
    # One pass over each freshly-generated chunk before it's discarded.
    chunks: int = 200
    max_grad_norm: float = 1.0
    eval_games: int = 200
    run_name: str = "bc"
    runs_root: str = "runs"
    work_seconds: float = 0.0


def bc_loss_and_stats(net, obs, legal_mask, action, outcome):
    """CE on the chosen action (over the legal-masked policy) + categorical CE
    of the value head against the teacher's true outcome one-hot/near-one-hot.
    """
    out = net(obs, legal_mask=legal_mask)
    move_logits = out["move_logits"].astype(mx.float32)
    log_probs = move_logits - mx.logsumexp(move_logits, axis=-1, keepdims=True)
    chosen_logp = mx.take_along_axis(log_probs, action[:, None], axis=-1).squeeze(-1)
    policy_loss = -chosen_logp.mean()

    value_logp = out["value_logp"].astype(mx.float32)
    value_loss = -(outcome * value_logp).sum(-1).mean()

    loss = policy_loss + value_loss
    top1 = (mx.argmax(move_logits, axis=-1) == action).astype(mx.float32).mean()
    return loss, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "top1_acc": top1,
    }


def _minibatches(n, batch_size, rng):
    perm = rng.permutation(n)
    for i in range(0, n, batch_size):
        yield perm[i : i + batch_size]


def _to_mx(data, idx):
    return (
        mx.array(data["obs"][idx]),
        mx.array(data["legal_mask"][idx]),
        mx.array(data["action"][idx].astype(np.int64)),
        mx.array(data["outcome"][idx]),
    )


def evaluate(move, cfg: BcConfig, seed_offset: int) -> dict:
    """Held-out CE + top-1 accuracy on a freshly-generated (unseen-seed) chunk."""
    data = sim.generate_bc_games(
        num_games=max(20, cfg.eval_games // 4), seed=cfg.seed + 900_000 + seed_offset, epsilon=0.0
    )
    n = data["action"].shape[0]
    obs, legal_mask, action, outcome = _to_mx(data, np.arange(n))
    loss, stats = bc_loss_and_stats(move, obs, legal_mask, action, outcome)
    mx.eval(loss)
    return {"eval/n": n, "eval/loss": float(loss), "eval/top1_acc": float(stats["top1_acc"])}


def train_bc(cfg: BcConfig) -> str:
    """Runs the streaming BC loop; returns the saved checkpoint path."""
    mx.random.seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    run = RunDir(cfg.runs_root, cfg.run_name, cfg.work_seconds, net_size=cfg.net_size)
    move, setup, move_opt, setup_opt, move_ema, setup_ema = build(TrainConfig(net_size=cfg.net_size))
    move_opt.learning_rate = cfg.lr

    def loss_fn(net, obs, legal_mask, action, outcome):
        loss, stats = bc_loss_and_stats(net, obs, legal_mask, action, outcome)
        return loss, stats

    print(f"[bc] {run.path} net_size={cfg.net_size} chunks={cfg.chunks} "
          f"games_per_chunk={cfg.games_per_chunk} lr={cfg.lr}")

    step = 0
    for c in range(cfg.chunks):
        if run.should_stop():
            print(f"[stop] STOP/work-budget at chunk {c}")
            break
        t0 = time.time()
        data = sim.generate_bc_games(
            num_games=cfg.games_per_chunk, seed=cfg.seed + 1 + c, epsilon=cfg.epsilon
        )
        n = data["action"].shape[0]
        t_gen = time.time() - t0

        chunk_loss = 0.0
        chunk_top1 = 0.0
        n_batches = 0
        for idx in _minibatches(n, cfg.batch_size, np_rng):
            obs, legal_mask, action, outcome = _to_mx(data, idx)
            (loss, stats), grads = nn.value_and_grad(move, loss_fn)(
                move, obs, legal_mask, action, outcome
            )
            grads, _ = optim.clip_grad_norm(grads, cfg.max_grad_norm)
            move_opt.update(move, grads)
            mx.eval(move.parameters(), move_opt.state)
            chunk_loss += float(loss)
            chunk_top1 += float(stats["top1_acc"])
            n_batches += 1
            step += 1

        rec = {
            "chunk": c,
            "step": step,
            "rows": n,
            "t_gen": round(t_gen, 3),
            "t_chunk": round(time.time() - t0, 3),
            "loss": chunk_loss / max(1, n_batches),
            "top1_acc": chunk_top1 / max(1, n_batches),
        }
        if c % 10 == 0 or c == cfg.chunks - 1:
            rec.update(evaluate(move, cfg, seed_offset=c))
            print(f"[bc] chunk={c} rows={n} loss={rec['loss']:.4f} "
                  f"top1={rec['top1_acc']:.3f} eval_loss={rec.get('eval/loss', float('nan')):.4f} "
                  f"eval_top1={rec.get('eval/top1_acc', float('nan')):.3f}")
        run.log(rec)

    # A fresh EMA/optimizer pair for the saved checkpoint's setup half — the
    # setup net was never trained (see module docstring), so its EMA shadow is
    # just its own (random-init) params; the move EMA is likewise a fresh shadow
    # seeded from the just-BC'd move params (not an average over BC training —
    # BC has no analogue of RL's slow-moving-target need for one).
    move_ema = S.EMA(move, decay=move_ema.decay)
    setup_ema = S.EMA(setup, decay=setup_ema.decay)
    path = run.save_named(
        "bc.safetensors", move, move_opt, move_ema, setup, setup_opt, setup_ema, step=step
    )
    print(f"[bc] saved {path}")
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--net-size", choices=tuple(S.NET_SIZES), default=BcConfig.net_size)
    p.add_argument("--seed", type=int, default=BcConfig.seed)
    p.add_argument("--games-per-chunk", type=int, default=BcConfig.games_per_chunk)
    p.add_argument("--epsilon", type=float, default=BcConfig.epsilon)
    p.add_argument("--lr", type=float, default=BcConfig.lr)
    p.add_argument("--batch-size", type=int, default=BcConfig.batch_size)
    p.add_argument("--chunks", type=int, default=BcConfig.chunks)
    p.add_argument("--eval-games", type=int, default=BcConfig.eval_games)
    p.add_argument("--run-name", type=str, default=BcConfig.run_name)
    p.add_argument("--runs-root", type=str, default=BcConfig.runs_root)
    p.add_argument("--work-seconds", type=float, default=BcConfig.work_seconds)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    cfg = BcConfig(
        net_size=a.net_size,
        seed=a.seed,
        games_per_chunk=a.games_per_chunk,
        epsilon=a.epsilon,
        lr=a.lr,
        batch_size=a.batch_size,
        chunks=a.chunks,
        eval_games=a.eval_games,
        run_name=a.run_name,
        runs_root=a.runs_root,
        work_seconds=a.work_seconds,
    )
    train_bc(cfg)


if __name__ == "__main__":
    main()
