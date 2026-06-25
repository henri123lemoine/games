"""The Ataraxos move-RL + co-trained setup self-play training loop in MLX.

Faithful to ATARAXOS_SPEC §4.1 (move-RL) and §4.2 (setup), driven by the verified
Rust sim via the `stratego_sim` bridge. Each iteration:

  1. Self-play: collect ragged per-phase decisions, run the move net on move-phase
     envs and the setup net on deploy-phase envs (reducing the value head
     softmax(W/L/D)@[-1,0,1] to a scalar), commit (the verified Rust sampler).
  2. Drain the move-RL arrays (λ-returns / GAE done sim-side) and the completed
     setup trajectories (full 40 placements + the game's MC outcome).
  3. Train ONE move pass (advantage-filtered PPO-clip + magnet-KL + value-CE +
     data-KL, dynamic-damping LR + magnet schedule) and 5 setup epochs (pure-MC
     PPO-clip + value-CE + conditional-entropy MSE + data-KL).
  4. EMA update both nets; periodically checkpoint, eval (win share vs random and
     vs a frozen earlier EMA), and log to metrics.jsonl.

CLI:
  python -m stratego_trainer.train --envs 1024 --iters 400 --run-name smoke
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import stratego_nets as S
import stratego_sim as sim

from .config import TrainConfig
from .eval import win_share
from .move_loss import advantage_filter_mask, move_loss_and_stats
from .rundir import RunDir
from .setup_loss import setup_baseline, setup_loss_and_stats

CATS = np.array(S.spec.CATEGORICAL_AGGREGATION, dtype=np.float32)  # [-1,0,1]


def _bucket(n, mult):
    """Smallest multiple of `mult` that is >= n (the MPS shape-cache fix)."""
    if mult <= 1 or n == 0:
        return n
    return ((n + mult - 1) // mult) * mult


def _move_forward(net, obs_np, legal_np):
    """Net forward on a move-phase batch -> (logits_np, scalar_values_np)."""
    if obs_np.shape[0] == 0:
        return (np.zeros((0, sim.N_ACTION), np.float32), np.zeros(0, np.float32))
    out = net(mx.array(obs_np), legal_mask=mx.array(legal_np))
    logits = np.array(out["move_logits"].astype(mx.float32))
    vlogp = np.array(out["value_logp"])
    vals = (np.exp(vlogp) * CATS).sum(-1).astype(np.float32)
    return logits, vals


def _setup_forward(net, obs_np):
    """Setup net forward on a deploy-phase batch -> (logits_np(B,14), scalar_values)."""
    if obs_np.shape[0] == 0:
        return (np.zeros((0, sim.DEPLOY_WIDTH), np.float32), np.zeros(0, np.float32))
    pc = mx.array(list(S.spec.CLASSIC_PIECE_COUNTS), dtype=mx.float32)
    out = net(mx.array(obs_np), pc)
    n_placed = obs_np.reshape(obs_np.shape[0], 40, 14).sum(axis=(1, 2)).astype(int)
    slot = np.clip(n_placed, 0, 39)
    all_logits = np.array(out["logits"].astype(mx.float32))  # (B,40,14)
    logits = all_logits[np.arange(obs_np.shape[0]), slot]
    vlogp = np.array(out["value"].astype(mx.float32))
    vlogp = vlogp - np.log(np.exp(vlogp).sum(-1, keepdims=True))
    vals = (np.exp(vlogp[np.arange(obs_np.shape[0]), slot]) * CATS).sum(-1).astype(np.float32)
    return logits, vals


def collect_iter(s, move_net, setup_net, steps):
    """Run `steps` self-play decision steps; return (env_steps, n_terminals, reward_pl0_sum)."""
    n_term = 0
    rsum = 0.0
    for _ in range(steps):
        b = s.collect()
        m_logits, m_vals = _move_forward(move_net, b["move_obs"], b["move_legal"])
        d_logits, d_vals = _setup_forward(setup_net, b["deploy_obs"])
        out = s.commit(m_logits, m_vals, d_logits, d_vals)
        n_term += int(out["terminal"].sum())
        rsum += float(out["reward_pl0"].sum())
    return steps * s.num_envs, n_term, rsum


def train_move_pass(move_net, opt, ema, data, magnet_coef, lr, cfg):
    """One advantage-filtered PPO pass over the drained move-RL transitions."""
    n = data["action"].shape[0]
    keep = advantage_filter_mask(data["advantage"], cfg.adv_filt_rate, cfg.adv_filt_thresh)
    # Always surface the attempt's counts so move training is visible every
    # iteration even when nothing passed the advantage filter (would otherwise
    # look like move training silently stopped).
    if n == 0 or keep.sum() == 0:
        return {"move/n_kept": int(keep.sum()), "move/n_total": int(n), "move/skipped": True}

    opt.learning_rate = lr
    idx = np.nonzero(keep)[0]
    # Bound the per-pass batch: a large kept set would blow up peak Metal memory
    # (obs is 92*F floats/row) and can trigger a GPU page-fault on a shared box.
    # Subsample the kept rows to at most `max_train_batch` before bucketing.
    if idx.size > cfg.max_train_batch:
        idx = np.random.choice(idx, cfg.max_train_batch, replace=False)
    # Pad to a bucket multiple by repeating filtered rows (keeps MPS shape stable;
    # repeated rows only rescale the mean, which advantage filtering already does).
    target = _bucket(idx.size, cfg.bucket)
    if target > idx.size:
        pad = np.random.choice(idx, target - idx.size, replace=True)
        idx = np.concatenate([idx, pad])

    batch = {
        "obs": mx.array(data["obs"][idx]),
        "legal": mx.array(data["legal_mask"][idx]),
        "action": mx.array(data["action"][idx].astype(np.int32)),
        "old_log_prob": mx.array(data["old_log_prob"][idx]),
        "data_log_prob": mx.array(np.where(np.isfinite(data["data_log_prob"][idx]),
                                           data["data_log_prob"][idx], -1e30).astype(np.float32)),
        "advantage": mx.array(data["advantage"][idx]),
        "ret": mx.array(data["ret"][idx]),
    }

    def loss_fn(net):
        loss, stats = move_loss_and_stats(net, batch, magnet_coef, cfg)
        return loss, stats

    (loss, stats), grads = nn.value_and_grad(move_net, loss_fn)(move_net)
    grads, gnorm = optim.clip_grad_norm(grads, cfg.max_grad_norm)
    loss_v = float(loss)
    gnorm_v = float(gnorm)
    # Skip a rare non-finite update rather than corrupt the net / crash the run.
    if not (np.isfinite(loss_v) and np.isfinite(gnorm_v)):
        return {"move/n_kept": int(keep.sum()), "move/n_total": int(n), "move/skipped": True}
    opt.update(move_net, grads)
    ema.update(move_net)
    mx.eval(move_net.parameters(), opt.state, ema.shadow)
    return {
        "move/policy_loss": float(stats["policy_loss"]),
        "move/value_loss": float(stats["value_loss"]),
        "move/kl_loss": float(stats["kl_loss"]),
        "move/magnet_kl": float(stats["magnet_kl"]),
        "move/entropy": float(stats["entropy"]),
        "move/loss": float(loss),
        "move/grad_norm": float(gnorm),
        "move/n_kept": int(keep.sum()),
        "move/n_total": int(n),
    }


def train_setup_pass(setup_net, opt, ema, setup_data, reg_temp, cfg):
    """5 epochs of the pure-MC setup loss over the drained arrangements."""
    m = setup_data["seq"].shape[0]
    if m == 0:
        return {}
    seq_all = mx.array(setup_data["seq"])
    outcome_all = mx.array(setup_data["outcome"])
    # Freeze the behavior baseline once (the generation-time actor).
    base = setup_baseline(setup_net, seq_all, cfg.arr_reg_norm)
    mx.eval(base["log_probs"], base["value_scalar"], base["reg_returns"], base["reg_adv"])

    last = {}
    n_skipped = 0
    bs = cfg.arr_batch_size
    for _ in range(cfg.arr_num_epoch_per_train):
        perm = np.random.permutation(m)
        for i in range(0, m, bs):
            bi = perm[i:i + bs]
            bidx = mx.array(bi)
            batch = {
                "seq": seq_all[bidx],
                "outcome": outcome_all[bidx],
                "reg_temp": reg_temp,
                "old_log_probs": base["log_probs"][bidx],
                "old_value_scalar": base["value_scalar"][bidx],
                "reg_returns": base["reg_returns"][bidx],
                "reg_adv": base["reg_adv"][bidx],
            }

            def loss_fn(net):
                loss, stats = setup_loss_and_stats(net, batch, cfg)
                return loss, stats

            (loss, stats), grads = nn.value_and_grad(setup_net, loss_fn)(setup_net)
            # The setup net reuses `start_token` inside its legal-mask cumsum, so
            # it must stay the fixed-zero start it is documented to be — training
            # it would corrupt the per-slot legality. Freeze its gradient.
            if "start_token" in grads:
                grads["start_token"] = mx.zeros_like(grads["start_token"])
            grads, gnorm = optim.clip_grad_norm(grads, cfg.arr_max_grad_norm)
            loss_v = float(loss)
            gnorm_v = float(gnorm)
            # A rare degenerate minibatch (e.g. a tiny all-draw setup batch once the
            # move policy plays both sides well) can still produce a non-finite
            # gradient. Skip that single update rather than corrupt the net or
            # crash the run — the next minibatch resumes cleanly.
            if not (np.isfinite(loss_v) and np.isfinite(gnorm_v)):
                n_skipped += 1
                continue
            opt.update(setup_net, grads)
            mx.eval(setup_net.parameters(), opt.state)
            last = {
                "setup/policy_loss": float(stats["policy_loss"]),
                "setup/value_loss": float(stats["value_loss"]),
                "setup/entropy_loss": float(stats["entropy_loss"]),
                "setup/kl_loss": float(stats["kl_loss"]),
                "setup/loss": float(loss),
                "setup/grad_norm": float(gnorm),
            }
    ema.update(setup_net)
    mx.eval(ema.shadow)
    last["setup/n_games"] = int(m)
    last["setup/n_skipped"] = n_skipped
    return last


def build(cfg):
    move = S.MoveTransformer.from_config(cfg.move_net_config())
    setup = S.ArrangementTransformer.from_config(cfg.setup_net_config())
    move_opt = optim.AdamW(learning_rate=cfg.lr(0), betas=[cfg.adam_b1, cfg.adam_b2],
                           eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
    setup_opt = optim.AdamW(learning_rate=cfg.arr_lr, betas=[cfg.adam_b1, cfg.adam_b2],
                            eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
    move_ema = S.EMA(move, decay=cfg.ema_decay)
    setup_ema = S.EMA(setup, decay=cfg.ema_decay)
    mx.eval(move.parameters(), setup.parameters())
    return move, setup, move_opt, setup_opt, move_ema, setup_ema


def ema_net(template_net, ema):
    """A net instance loaded with the EMA shadow params (for eval)."""
    template_net.update(ema.shadow_params())
    mx.eval(template_net.parameters())
    return template_net


def train(cfg: TrainConfig):
    mx.random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    # Be a good citizen on the shared 64 GB box: cap MLX's working set and its
    # allocator cache so a transient spike can't page-fault the GPU or starve
    # other processes. `set_memory_limit` makes over-allocation raise (a catchable
    # Python error) instead of a fatal Metal command-buffer fault.
    if cfg.mlx_memory_limit_gb > 0:
        mx.set_memory_limit(int(cfg.mlx_memory_limit_gb * 1024**3))
    if cfg.mlx_cache_limit_gb > 0:
        mx.set_cache_limit(int(cfg.mlx_cache_limit_gb * 1024**3))
    run = RunDir(cfg.runs_root, cfg.run_name, cfg.work_seconds)
    move, setup, move_opt, setup_opt, move_ema, setup_ema = build(cfg)
    # eval template nets (separate instances so loading EMA doesn't disturb the learners)
    eval_move = S.MoveTransformer.from_config(cfg.move_net_config())
    eval_setup = S.ArrangementTransformer.from_config(cfg.setup_net_config())
    # frozen earlier-EMA snapshot for self-play-improvement eval (set on first eval)
    frozen_move = S.MoveTransformer.from_config(cfg.move_net_config())
    frozen_setup = S.ArrangementTransformer.from_config(cfg.setup_net_config())
    have_frozen = False

    s = sim.BatchSim(num_envs=cfg.num_envs, move_cap=cfg.move_cap, seed=cfg.seed,
                     buffer_capacity=cfg.buffer_capacity)

    total_env_steps = 0
    print(f"[run] {run.path}  envs={cfg.num_envs} iters={cfg.iters} "
          f"move={cfg.move_net_config()} setup={cfg.setup_net_config()}")

    for t in range(cfg.iters):
        if run.should_stop():
            print(f"[stop] STOP/work-budget at iter {t}")
            break
        it_start = time.time()

        # 1-2: self-play + drain
        env_steps, n_term, rsum = collect_iter(s, move, setup, cfg.collect_steps)
        total_env_steps += env_steps
        move_data = s.drain_training_batch(cfg.td_lambda, cfg.gae_lambda)
        setup_data = s.drain_setup_batch()

        # 3: train
        lr = cfg.lr(t)
        magnet_coef = cfg.magnet_coef(t)
        reg_temp = cfg.setup_temperature(t)
        m_stats = train_move_pass(move, move_opt, move_ema, move_data, magnet_coef, lr, cfg)
        a_stats = train_setup_pass(setup, setup_opt, setup_ema, setup_data, reg_temp, cfg)

        # Drop the large drained numpy batches and flush the MLX buffer cache each
        # iteration — without this the Metal allocator's cache grows unbounded
        # (iter time balloons, then OOM on a shared 64 GB box).
        del move_data, setup_data
        mx.clear_cache()

        rec = {
            "iter": t,
            "env_steps": total_env_steps,
            "work_seconds": round(run.elapsed(), 1),
            "iter_seconds": round(time.time() - it_start, 2),
            "lr": lr,
            "magnet_coef": magnet_coef,
            "setup_temp": reg_temp,
            "n_terminals": n_term,
            "reward_pl0_mean": (rsum / n_term) if n_term else 0.0,
            **m_stats,
            **a_stats,
        }

        # 5: eval — win share vs uniform-random (the milestone proof) and vs a
        # frozen earlier policy (self-play improvement). We log BOTH the live
        # working policy (`winrate_vs_random`, the direct learning signal) and the
        # EMA/"magnet" policy (`ema_winrate_vs_random`, the spec-faithful deployed
        # net). Over a short smoke the 0.999 EMA lags the working net heavily, so
        # the working-net curve is the legible learning proof; the EMA catches up
        # over the full run. Hero sampling is sharpened (eval_temperature) so a
        # still-exploratory policy's learned preferences show through.
        do_eval = (t > 0 and t % cfg.eval_every == 0) or t == cfg.iters - 1
        if do_eval:
            eval_envs = min(128, cfg.num_envs)
            ws_rand = win_share(move, setup, None, None, num_envs=eval_envs,
                                games=cfg.eval_games, move_cap=cfg.move_cap,
                                seed=cfg.seed + 999, hero_temperature=cfg.eval_temperature)
            rec["eval/winrate_vs_random"] = round(ws_rand, 4)

            em = ema_net(eval_move, move_ema)
            es = ema_net(eval_setup, setup_ema)
            ema_ws_rand = win_share(em, es, None, None, num_envs=eval_envs,
                                    games=cfg.eval_games, move_cap=cfg.move_cap,
                                    seed=cfg.seed + 999, hero_temperature=cfg.eval_temperature)
            rec["eval/ema_winrate_vs_random"] = round(ema_ws_rand, 4)

            if have_frozen:
                ws_self = win_share(move, setup, frozen_move, frozen_setup,
                                    num_envs=eval_envs, games=cfg.eval_games,
                                    move_cap=cfg.move_cap, seed=cfg.seed + 7,
                                    hero_temperature=cfg.eval_temperature)
                rec["eval/winrate_vs_frozen"] = round(ws_self, 4)
            # snapshot the current WORKING policy as the next frozen reference
            frozen_move.update(move.parameters())
            frozen_setup.update(setup.parameters())
            mx.eval(frozen_move.parameters(), frozen_setup.parameters())
            have_frozen = True
            run.maybe_save_best(ws_rand, move, move_opt, move_ema, setup, setup_opt,
                                setup_ema, step=t)
            mx.clear_cache()

        run.log(rec)
        if t % 10 == 0 or "eval/winrate_vs_random" in rec:
            msg = (f"it {t:4d}  Lp={rec.get('move/policy_loss', 0):+.3f} "
                   f"Lv={rec.get('move/value_loss', 0):.3f} "
                   f"H={rec.get('move/entropy', 0):.3f} "
                   f"mag={rec.get('move/magnet_kl', 0):+.3f} lr={lr:.1e} "
                   f"setupL={rec.get('setup/loss', 0):.3f} "
                   f"term={n_term} {rec.get('iter_seconds', 0):.1f}s")
            if "eval/winrate_vs_random" in rec:
                msg += (f"  WSrand={rec['eval/winrate_vs_random']:.3f}"
                        f" (ema {rec.get('eval/ema_winrate_vs_random', 0):.3f})")
            if "eval/winrate_vs_frozen" in rec:
                msg += f" WSfrozen={rec['eval/winrate_vs_frozen']:.3f}"
            print(msg)

        if t > 0 and t % cfg.save_every == 0:
            run.save_periodic(move, move_opt, move_ema, setup, setup_opt, setup_ema, step=t)

    run.save_latest(move, move_opt, move_ema, setup, setup_opt, setup_ema, step=cfg.iters)
    print(f"[done] {total_env_steps} env-steps, best WSrand={run.best_eval:.3f}, "
          f"metrics -> {run.metrics_path}")
    return run


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ataraxos Stratego self-play trainer (MLX)")
    p.add_argument("--envs", type=int, default=TrainConfig.num_envs)
    p.add_argument("--iters", type=int, default=TrainConfig.iters)
    p.add_argument("--collect-steps", type=int, default=TrainConfig.collect_steps)
    p.add_argument("--buffer-capacity", type=int, default=TrainConfig.buffer_capacity)
    p.add_argument("--move-cap", type=int, default=TrainConfig.move_cap)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--run-name", type=str, default="run")
    p.add_argument("--runs-root", type=str, default=TrainConfig.runs_root)
    p.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    p.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    p.add_argument("--eval-games", type=int, default=TrainConfig.eval_games)
    p.add_argument("--eval-temperature", type=float, default=TrainConfig.eval_temperature)
    p.add_argument("--work-seconds", type=float, default=TrainConfig.work_seconds)
    # The magnet-KL strength is the shipped-default 0.05 (spec §4.1) but "make it
    # configurable" — at smoke scale the magnet otherwise overwhelms the weaker
    # advantage signal and pins the policy at uniform, so the smoke lowers it.
    p.add_argument("--magnet-coef", type=float, default=TrainConfig.temperature_coef)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    cfg = TrainConfig(
        num_envs=a.envs,
        iters=a.iters,
        collect_steps=a.collect_steps,
        buffer_capacity=a.buffer_capacity,
        move_cap=a.move_cap,
        seed=a.seed,
        run_name=a.run_name,
        runs_root=a.runs_root,
        save_every=a.save_every,
        eval_every=a.eval_every,
        eval_games=a.eval_games,
        eval_temperature=a.eval_temperature,
        temperature_coef=a.magnet_coef,
        work_seconds=a.work_seconds,
    )
    train(cfg)


if __name__ == "__main__":
    main()
