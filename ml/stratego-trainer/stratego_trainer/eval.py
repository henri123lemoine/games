"""Eval hook: win share of a hero policy (move + setup nets) vs an opponent.

The bridge drives BOTH players through one net forward, tagging every decision
row with its acting player. We exploit that to run asymmetric matches: hero rows
(the seat under test) get the hero net's logits; opponent rows get the opponent
policy's logits (uniform-random, or a frozen earlier net for self-play
improvement). Hero is rotated through both seats and win share is averaged, per
the repo convention ("hero rotated through every seat; fair = 1/players").

Win share counts a win as 1, draw as 0.5, loss as 0 (so 0.5 is parity).
"""

import numpy as np

import stratego_sim as sim

import mlx.core as mx

import stratego_nets as S

CATS = np.array(S.spec.CATEGORICAL_AGGREGATION, dtype=np.float32)  # [-1,0,1]


def _net_logits(move_net, setup_net, obs, legal, kind, temperature=1.0):
    """Return (logits, scalar_values) for a batch of rows of one phase.

    The sim softmax-samples the returned logits, so `temperature < 1` sharpens the
    sampled play toward the policy's preferred moves. A learned policy still
    carries high entropy early (the magnet keeps exploration alive), so a low eval
    temperature surfaces what it actually prefers rather than its exploration
    noise — the standard "is it learning" measurement (illegal slots are -inf, so
    scaling preserves the mask)."""
    if obs.shape[0] == 0:
        if kind == "move":
            return np.zeros((0, sim.N_ACTION), np.float32), np.zeros(0, np.float32)
        return np.zeros((0, sim.DEPLOY_WIDTH), np.float32), np.zeros(0, np.float32)
    obs_mx = mx.array(obs)
    inv_t = 1.0 / max(temperature, 1e-6)
    # Illegal slots are filled with finfo.min; scaling that by inv_t overflows to
    # -inf (a numpy warning). The sim only reads legal slots, so clamp to a finite
    # floor first — illegal stays maximally negative without overflowing.
    floor = -1e30

    if kind == "move":
        out = move_net(obs_mx, legal_mask=mx.array(legal))
        logits = np.clip(np.array(out["move_logits"].astype(mx.float32)), floor, None) * inv_t
        vlogp = np.array(out["value_logp"])
        vals = (np.exp(vlogp) * CATS).sum(-1)
        return logits, vals.astype(np.float32)
    else:
        pc = mx.array(list(S.spec.CLASSIC_PIECE_COUNTS), dtype=mx.float32)
        # Build the running placement prefix for each row from the deploy obs
        # (already the (40,14) one-hot the setup net consumes).
        out = setup_net(obs_mx, pc)
        # The setup net emits per-slot logits; the decision is for the NEXT empty
        # slot = number of placed pieces so far (sum over the one-hot prefix).
        n_placed = np.array(obs).reshape(obs.shape[0], 40, 14).sum(axis=(1, 2)).astype(int)
        all_logits = np.clip(np.array(out["logits"].astype(mx.float32)), floor, None)  # (B,40,14)
        slot = np.clip(n_placed, 0, 39)
        logits = all_logits[np.arange(obs.shape[0]), slot] * inv_t  # (B,14)
        vlogp = np.array(out["value"].astype(mx.float32))
        vlogp = vlogp - np.log(np.exp(vlogp).sum(-1, keepdims=True))
        vals = (np.exp(vlogp[np.arange(obs.shape[0]), slot]) * CATS).sum(-1)
        return logits, vals.astype(np.float32)


def _uniform_logits(obs, kind):
    n = obs.shape[0]
    w = sim.N_ACTION if kind == "move" else sim.DEPLOY_WIDTH
    return np.zeros((n, w), np.float32), np.zeros(n, np.float32)


# A near-one-hot logit on the heuristic's chosen action: the sim gathers logits
# down to the legal set and softmax-samples, so a large gap collapses the sample
# onto that (always-legal) move while leaving every illegal slot untouched.
_HEURISTIC_LOGIT = 30.0


def _heuristic_move_logits(s, env_ids):
    """Opponent move-row logits driven by the Rust `HeuristicBot` (see
    `BatchSim.heuristic_move_actions`). `env_ids` are the env ids of the rows to
    fill, in row order. A `-1` (mid-deploy / no move) leaves the row uniform."""
    n = int(env_ids.shape[0])
    logits = np.zeros((n, sim.N_ACTION), np.float32)
    if n == 0:
        return logits, np.zeros(0, np.float32)
    acts = np.asarray(s.heuristic_move_actions([int(e) for e in env_ids]), dtype=np.int64)
    rows = np.nonzero(acts >= 0)[0]
    logits[rows, acts[rows]] = _HEURISTIC_LOGIT
    return logits, np.zeros(n, np.float32)


def play_matches(hero_move, hero_setup, opp_move, opp_setup, hero_seat,
                 num_envs, games, move_cap, seed, hero_temperature=1.0,
                 heuristic=False):
    """Play until `games` complete; hero acts on rows whose player == hero_seat.

    The opponent (non-hero) rows are filled by, in priority order:
    `heuristic=True` -> the Rust `HeuristicBot` on move rows (uniform deploy, as
    the bot itself deploys uniformly); else `opp_move`/`opp_setup` net if given;
    else uniform-random. `hero_temperature` sharpens only the hero's sampling
    (the opponent net, if any, plays at temperature 1.0). Returns
    (hero_wins, hero_draws, hero_losses)."""
    s = sim.BatchSim(num_envs=num_envs, move_cap=move_cap, seed=seed)
    wins = draws = losses = 0
    completed = 0
    max_steps = games * move_cap + 200000
    step = 0
    while completed < games and step < max_steps:
        step += 1
        b = s.collect()
        m_obs, m_legal, m_pl = b["move_obs"], b["move_legal"], b["move_player"]
        m_env = b["move_env"]
        d_obs, d_legal, d_pl = b["deploy_obs"], b["deploy_legal"], b["deploy_player"]

        m_logits = np.zeros((m_obs.shape[0], sim.N_ACTION), np.float32)
        m_vals = np.zeros(m_obs.shape[0], np.float32)
        d_logits = np.zeros((d_obs.shape[0], sim.DEPLOY_WIDTH), np.float32)
        d_vals = np.zeros(d_obs.shape[0], np.float32)

        for seat in (0, 1):
            is_hero = seat == hero_seat
            mm = m_pl == seat
            dd = d_pl == seat
            temp = hero_temperature if is_hero else 1.0
            if mm.any():
                if is_hero:
                    lg, vl = _net_logits(hero_move, hero_setup, m_obs[mm], m_legal[mm], "move", temp)
                elif heuristic:
                    lg, vl = _heuristic_move_logits(s, m_env[mm])
                elif opp_move is not None:
                    lg, vl = _net_logits(opp_move, opp_setup, m_obs[mm], m_legal[mm], "move")
                else:
                    lg, vl = _uniform_logits(m_obs[mm], "move")
                m_logits[mm] = lg
                m_vals[mm] = vl
            if dd.any():
                if is_hero:
                    lg, vl = _net_logits(hero_move, hero_setup, d_obs[dd], d_legal[dd], "deploy", temp)
                elif heuristic:
                    lg, vl = _uniform_logits(d_obs[dd], "deploy")
                elif opp_move is not None:
                    lg, vl = _net_logits(opp_move, opp_setup, d_obs[dd], d_legal[dd], "deploy")
                else:
                    lg, vl = _uniform_logits(d_obs[dd], "deploy")
                d_logits[dd] = lg
                d_vals[dd] = vl

        out = s.commit(m_logits, m_vals, d_logits, d_vals)
        # The net forwards above already materialized to numpy (np.array), so no
        # MLX graph is retained across steps; flush the Metal cache periodically
        # so a long eval cannot accumulate allocator pressure.
        if step % 64 == 0:
            mx.clear_cache()
        term = out["terminal"]
        r0 = out["reward_pl0"]
        if term.any():
            idx = np.nonzero(term)[0]
            for e in idx:
                completed += 1
                # reward in hero's POV: +r0 if hero is player 0 else -r0.
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


def win_share(hero_move, hero_setup, opp_move, opp_setup, num_envs, games,
              move_cap, seed, hero_temperature=1.0, heuristic=False):
    """Hero win share vs opponent, hero rotated through both seats.

    `heuristic=True` pits the hero against the Rust `HeuristicBot` baseline
    instead of `opp_move`/uniform-random."""
    half = max(1, games // 2)
    w0, dr0, l0 = play_matches(hero_move, hero_setup, opp_move, opp_setup, 0,
                               num_envs, half, move_cap, seed, hero_temperature,
                               heuristic)
    w1, dr1, l1 = play_matches(hero_move, hero_setup, opp_move, opp_setup, 1,
                               num_envs, half, move_cap, seed + 12345, hero_temperature,
                               heuristic)
    w, dr, ll = w0 + w1, dr0 + dr1, l0 + l1
    total = w + dr + ll
    if total == 0:
        return 0.5
    return (w + 0.5 * dr) / total
