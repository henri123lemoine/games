"""Head-to-head eval: the §5 test-time search vs the raw move-net policy.

Plays full games where the *hero* seat resolves each of its move-phase decisions
with `search(...)` (depth-D belief rollouts + the MMD update) while the opponent
seat samples the raw move net. Both seats deploy with the same setup net (so the
comparison isolates the move-phase search). Hero is rotated through both seats and
win share is averaged, per the repo convention. Search should beat the raw policy
by a measurable margin (the spec's +120 Elo at full scale).

The bridge samples the logits we hand it, so we force the search action by
handing a one-hot-ish logit vector (the chosen slot at +30, the rest at -1e30).
The opponent and the deploy phase are handled by the batched net forward; only the
hero's move rows are resolved one env at a time through a `Searcher`.
"""

from __future__ import annotations

import numpy as np

import stratego_sim as sim

import mlx.core as mx

import stratego_nets as S

from .search import MarginalizedBelief, search

CATS = np.array(S.spec.CATEGORICAL_AGGREGATION, dtype=np.float32)
_FORCE_HI = 30.0
_FORCE_LO = -1e30


def _move_logits(move_net, obs, legal, temperature=1.0):
    if obs.shape[0] == 0:
        return (np.zeros((0, sim.N_ACTION), np.float32), np.zeros(0, np.float32),
                np.zeros((0, 3), np.float32))
    out = move_net(mx.array(obs), legal_mask=mx.array(legal))
    inv_t = 1.0 / max(temperature, 1e-6)
    logits = np.clip(np.array(out["move_logits"].astype(mx.float32)), -1e30, None) * inv_t
    vlogp = np.array(out["value_logp"])
    probs = np.exp(vlogp)
    vals = (probs * CATS).sum(-1).astype(np.float32)
    return logits, vals, probs.astype(np.float32)


def _setup_logits(setup_net, obs, temperature=1.0):
    if obs.shape[0] == 0:
        return (np.zeros((0, sim.DEPLOY_WIDTH), np.float32), np.zeros(0, np.float32),
                np.zeros((0, 3), np.float32))
    pc = mx.array(list(S.spec.CLASSIC_PIECE_COUNTS), dtype=mx.float32)
    out = setup_net(mx.array(obs), pc)
    n_placed = obs.reshape(obs.shape[0], 40, 14).sum(axis=(1, 2)).astype(int)
    slot = np.clip(n_placed, 0, 39)
    inv_t = 1.0 / max(temperature, 1e-6)
    all_logits = np.clip(np.array(out["logits"].astype(mx.float32)), -1e30, None)
    logits = all_logits[np.arange(obs.shape[0]), slot] * inv_t
    vlogp = np.array(out["value"].astype(mx.float32))
    vlogp = vlogp - np.log(np.exp(vlogp).sum(-1, keepdims=True))
    probs = np.exp(vlogp[np.arange(obs.shape[0]), slot])
    vals = (probs * CATS).sum(-1).astype(np.float32)
    return logits, vals, probs.astype(np.float32)


def _play_matches(move_net, setup_net, hero_seat, num_envs, games, move_cap, seed,
                  depth, stepsize, temperature, max_samples, search_envs, belief,
                  opp_temperature):
    """Play until `games` complete; the hero seat searches, the opponent samples
    the raw policy. Returns (wins, draws, losses) from the hero's POV."""
    s = sim.BatchSim(num_envs=num_envs, move_cap=move_cap, seed=seed)
    wins = draws = losses = 0
    completed = 0
    rng = np.random.default_rng(seed + 1)
    max_steps = games * move_cap + 200000
    step = 0
    while completed < games and step < max_steps:
        step += 1
        b = s.collect()
        m_obs, m_legal, m_env, m_pl = (b["move_obs"], b["move_legal"],
                                       b["move_env"], b["move_player"])
        d_obs = b["deploy_obs"]

        m_logits = np.zeros((m_obs.shape[0], sim.N_ACTION), np.float32)
        m_vals = np.zeros(m_obs.shape[0], np.float32)
        m_probs = np.full((m_obs.shape[0], 3), 1.0 / 3.0, np.float32)
        d_logits = np.zeros((d_obs.shape[0], sim.DEPLOY_WIDTH), np.float32)
        d_vals = np.zeros(d_obs.shape[0], np.float32)
        d_probs = np.full((d_obs.shape[0], 3), 1.0 / 3.0, np.float32)

        # Deploy: both seats use the setup net.
        if d_obs.shape[0] > 0:
            d_logits, d_vals, d_probs = _setup_logits(setup_net, d_obs)

        # Opponent move rows: raw policy (sampled).
        opp_rows = np.nonzero(m_pl != hero_seat)[0]
        if opp_rows.size > 0:
            lg, vl, vp = _move_logits(move_net, m_obs[opp_rows], m_legal[opp_rows], opp_temperature)
            m_logits[opp_rows] = lg
            m_vals[opp_rows] = vl
            m_probs[opp_rows] = vp

        # Hero move rows: resolve by search, one env at a time, forcing the action.
        hero_rows = np.nonzero(m_pl == hero_seat)[0]
        for row in hero_rows:
            env = int(m_env[row])
            srch = s.search_root(env)
            res = search(move_net, srch, depth=depth, stepsize=stepsize,
                         temperature=temperature, max_samples=max_samples,
                         num_envs=search_envs, belief=belief,
                         seed=int(rng.integers(1 << 30)))
            forced = np.full(sim.N_ACTION, _FORCE_LO, np.float32)
            forced[res.action] = _FORCE_HI
            m_logits[row] = forced
            m_vals[row] = 0.0
            m_probs[row] = (0.0, 1.0, 0.0)

        out = s.commit(m_logits, m_vals, m_probs, d_logits, d_vals, d_probs)
        mx.clear_cache()
        term = out["terminal"]
        if term.any():
            r0 = out["reward_pl0"]
            for e in np.nonzero(term)[0]:
                completed += 1
                hr = r0[e] if hero_seat == 0 else -r0[e]
                if hr > 0:
                    wins += 1
                elif hr < 0:
                    losses += 1
                else:
                    draws += 1
                if completed >= games:
                    break
    return wins, draws, losses


def search_vs_policy(move_net, setup_net, games=40, num_envs=64, move_cap=400,
                     seed=0, depth=10, stepsize=10.0, temperature=1e-3,
                     max_samples=100, search_envs=512,
                     belief=MarginalizedBelief(), opp_temperature=1.0):
    """Win share of the depth-`D` search hero vs the raw move-net policy, hero
    rotated through both seats. Returns a dict with the win/draw/loss tallies, the
    win share, and a Wald 95% CI half-width on the win share."""
    half = max(1, games // 2)
    w0, dr0, l0 = _play_matches(move_net, setup_net, 0, num_envs, half, move_cap,
                                seed, depth, stepsize, temperature, max_samples,
                                search_envs, belief, opp_temperature)
    w1, dr1, l1 = _play_matches(move_net, setup_net, 1, num_envs, half, move_cap,
                                seed + 4242, depth, stepsize, temperature, max_samples,
                                search_envs, belief, opp_temperature)
    w, dr, ll = w0 + w1, dr0 + dr1, l0 + l1
    total = w + dr + ll
    if total == 0:
        return {"win_share": 0.5, "wins": 0, "draws": 0, "losses": 0, "n": 0, "ci95": 0.0}
    score = w + 0.5 * dr
    ws = score / total
    # Wald CI on the per-game score in {0, .5, 1}.
    p = ws
    var = max(p * (1 - p), 1e-9) / total
    ci = 1.96 * np.sqrt(var)
    return {"win_share": ws, "wins": w, "draws": dr, "losses": ll, "n": total,
            "ci95": ci}


def _main(argv=None):
    """`python -m stratego_trainer.search_eval --ckpt runs/smoke/latest.safetensors`
    — load a run checkpoint (EMA move + setup nets) and report the §5 search win
    share vs the raw move-net policy. The milestone demonstration: search should
    beat the raw policy by a measurable margin (the spec's +120 Elo at scale)."""
    import argparse
    import math

    from .rundir import load_checkpoint

    p = argparse.ArgumentParser(description="search vs raw-policy eval (ATARAXOS §5)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--search-envs", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=48)
    p.add_argument("--move-cap", type=int, default=400)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--working", action="store_true",
                   help="use the working (non-EMA) net instead of the EMA/magnet net")
    a = p.parse_args(argv)

    move = S.MoveTransformer.from_config(S.MoveConfig())
    setup = S.ArrangementTransformer.from_config(S.SetupConfig())
    load_checkpoint(a.ckpt, move=move, setup=setup, prefer_ema=not a.working)
    res = search_vs_policy(move, setup, games=a.games, num_envs=a.num_envs,
                           move_cap=a.move_cap, seed=a.seed, depth=a.depth,
                           max_samples=a.max_samples, search_envs=a.search_envs)
    ws, n = res["win_share"], res["n"]
    margin = ws - 0.5
    se = math.sqrt(max(ws * (1 - ws), 1e-9) / max(n, 1))
    z = margin / se if se > 0 else 0.0
    print(f"search(depth={a.depth}) vs raw policy: win_share={ws:.3f} +/- {res['ci95']:.3f}  "
          f"(W{res['wins']}/D{res['draws']}/L{res['losses']}, n={n})")
    print(f"margin over policy = {margin:+.3f}  (z={z:.2f} vs the 0.5 null)")


if __name__ == "__main__":
    _main()
